#!/usr/bin/env python3
"""
Hyperparameter-tuned Random Forest T-learner for mediator CATE.

A drop-in replacement for `RandomForestMediator` that performs an
internal `RandomizedSearchCV` over (max_depth, max_features,
min_samples_leaf, n_estimators) inside each treatment arm.

Same fit(X, T, M) / predict_cate(X) interface as the other T-learner
estimators.
"""

import time
import tracemalloc

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from .base import (
    MediatorCATEEstimator,
    check_inputs,
    check_prediction_inputs,
)


# Search grid favouring deeper, decorrelated forests that preserve the
# treatment-effect contrast recovered by the T-learner difference mu1_hat - mu0_hat.
# Shallower trees with larger leaves tend to over-smooth and erase the CATE signal.
_DEFAULT_PARAM_DIST = {
    'max_depth':        [10, 20, None],
    'max_features':     [0.2, 0.33, 0.5],
    'min_samples_leaf': [1, 3, 5],
    'n_estimators':     [200, 300],
}


class RandomForestTunedMediator(MediatorCATEEstimator):
    """
    T-learner Random Forest with per-arm RandomizedSearchCV.

    Parameters
    ----------
    n_iter : int, default=15
        Number of draws in the RandomizedSearchCV per arm.
    cv : int, default=3
        Number of CV folds inside the search.
    param_dist : dict, optional
        Override the default search grid. Defaults to
        `_DEFAULT_PARAM_DIST`.
    n_jobs : int, default=1
        Parallelism for tree fitting. Must be 1 inside a ProcessPoolExecutor
        worker to avoid fork contention.
    seed : int, default=0
        Random seed used for both the RandomizedSearchCV split and the
        per-arm RF random_state.

    Attributes after fit
    --------------------
    model_0_ : RandomForestRegressor — best-tuned RF for T=0
    model_1_ : RandomForestRegressor — best-tuned RF for T=1
    best_params_0_ / best_params_1_ : dict — chosen hyperparameters per arm
    """

    def __init__(
        self,
        n_iter: int = 15,
        cv: int = 3,
        param_dist: dict = None,
        n_jobs: int = 1,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(
            name="RandomForestTuned",
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            seed=seed,
            **kwargs,
        )
        self.n_iter      = int(n_iter)
        self.cv          = int(cv)
        self.param_dist  = param_dist if param_dist is not None else _DEFAULT_PARAM_DIST
        self.n_jobs      = int(n_jobs)
        self.random_state = int(seed)

        self.model_0_ = None
        self.model_1_ = None
        self.best_params_0_ = None
        self.best_params_1_ = None

    def _fit_arm(self, X_arm: np.ndarray, M_arm: np.ndarray,
                 rs_offset: int) -> tuple:
        base = RandomForestRegressor(
            random_state=self.random_state + rs_offset,
            n_jobs=self.n_jobs,
        )
        search = RandomizedSearchCV(
            base,
            self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='neg_mean_squared_error',
            random_state=self.random_state + rs_offset,
            n_jobs=1,        # nested parallelism is hazardous in pool workers
            refit=True,
        )
        search.fit(X_arm, M_arm)
        return search.best_estimator_, search.best_params_

    def fit(self, X: np.ndarray, T: np.ndarray, M: np.ndarray,
            **kwargs) -> 'RandomForestTunedMediator':
        check_inputs(X, T, M)

        tracemalloc.start()
        start_time = time.time()

        self.feature_dim = X.shape[1]
        idx_0 = (T == 0)
        idx_1 = (T == 1)

        self.model_0_, self.best_params_0_ = self._fit_arm(
            X[idx_0], M[idx_0], rs_offset=0
        )
        self.model_1_, self.best_params_1_ = self._fit_arm(
            X[idx_1], M[idx_1], rs_offset=1
        )

        self.is_fitted = True
        self.training_time = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / 1024 / 1024
        tracemalloc.stop()
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        check_prediction_inputs(X, expected_dim=self.feature_dim)
        return self.model_1_.predict(X) - self.model_0_.predict(X)

    def predict_potential_outcomes(self, X: np.ndarray) -> tuple:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        check_prediction_inputs(X, expected_dim=self.feature_dim)
        return self.model_0_.predict(X), self.model_1_.predict(X)
