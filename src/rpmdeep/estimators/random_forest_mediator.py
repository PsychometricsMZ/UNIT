#!/usr/bin/env python
"""
Random Forest T-learner estimator for mediator CATE.

Fits separate RandomForestRegressor models for T=0 and T=1:
  - μ₀(X) = RF(X | T=0)
  - μ₁(X) = RF(X | T=1)
  - τ_M(X) = μ₁(X) - μ₀(X)
"""

import numpy as np
import time
import tracemalloc
from sklearn.ensemble import RandomForestRegressor

from .base import (
    MediatorCATEEstimator,
    check_inputs,
    check_prediction_inputs
)


class RandomForestMediator(MediatorCATEEstimator):
    """
    Random Forest T-learner for mediator CATE estimation.

    Fits two separate RandomForestRegressor models:
    - model_0: M ~ X for control group (T=0)
    - model_1: M ~ X for treated group (T=1)

    CATE prediction: τ̂_M(X) = model_1.predict(X) - model_0.predict(X)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        min_samples_leaf: int = 10,
        max_features: float = 0.5,
        n_jobs: int = 1,
        seed: int = 0,
        **kwargs
    ):
        """
        Parameters
        ----------
        n_estimators : int, default=200
            Number of trees in the forest.
        max_depth : int, default=6
            Maximum depth of each tree.  Kept shallow to avoid over-fitting.
        min_samples_leaf : int, default=10
            Minimum samples per leaf.
        max_features : float, default=0.5
            Fraction of features considered at each split.
        n_jobs : int, default=1
            Parallelism for tree fitting.  Set to 1 in multiprocessing contexts
            to avoid fork/thread contention.
        seed : int, default=0
            Random seed for reproducibility.
        """
        super().__init__(
            name="RandomForest",
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            seed=seed,
            **kwargs
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = seed
        self.model_0 = None
        self.model_1 = None

    def _make_model(self, rs_offset: int = 0) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state + rs_offset,
        )

    def fit(self, X: np.ndarray, T: np.ndarray, M: np.ndarray, **kwargs) -> 'RandomForestMediator':
        """
        Fit separate RF models for each treatment group.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        T : np.ndarray, shape (n_samples,)
        M : np.ndarray, shape (n_samples,)
        """
        check_inputs(X, T, M)

        tracemalloc.start()
        start_time = time.time()

        self.feature_dim = X.shape[1]

        idx_0 = (T == 0)
        idx_1 = (T == 1)

        self.model_0 = self._make_model(rs_offset=0)
        self.model_0.fit(X[idx_0], M[idx_0])

        self.model_1 = self._make_model(rs_offset=1)
        self.model_1.fit(X[idx_1], M[idx_1])

        self.is_fitted = True

        self.training_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / 1024 / 1024
        tracemalloc.stop()

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mediator CATE: τ̂_M(X) = μ̂₁(X) - μ̂₀(X).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        check_prediction_inputs(X, expected_dim=self.feature_dim)

        return self.model_1.predict(X) - self.model_0.predict(X)

    def predict_potential_outcomes(self, X: np.ndarray) -> tuple:
        """Return (μ̂₀(X), μ̂₁(X))."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        check_prediction_inputs(X, expected_dim=self.feature_dim)

        return self.model_0.predict(X), self.model_1.predict(X)
