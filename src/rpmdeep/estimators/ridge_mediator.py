#!/usr/bin/env python
"""
Ridge regression estimator for mediator CATE with cross-validated alpha selection.

Fits separate Ridge models for T=0 and T=1 with automatic alpha tuning:
  - μ₀(X) = X·β₀ + b₀  (fit on control samples)
  - μ₁(X) = X·β₁ + b₁  (fit on treated samples)
  - τ_M(X) = μ₁(X) - μ₀(X)
"""

import numpy as np
import time
import tracemalloc
from sklearn.linear_model import RidgeCV

from .base import (
    MediatorCATEEstimator,
    check_inputs,
    check_prediction_inputs
)


class RidgeMediator(MediatorCATEEstimator):
    """
    Ridge regression estimator for mediator CATE (T-learner with L2 regularization).

    Fits two separate Ridge regression models with cross-validation for alpha:
    - model_0: M ~ X for control group (T=0)
    - model_1: M ~ X for treated group (T=1)

    Alpha is selected via K-fold CV from a predefined grid.

    CATE prediction: τ̂_M(X) = model_1.predict(X) - model_0.predict(X)
    """

    def __init__(
        self,
        alphas: tuple = (0.01, 0.1, 1.0, 10.0),
        cv: int = 5,
        fit_intercept: bool = True,
        **kwargs
    ):
        """
        Initialize Ridge Mediator estimator.

        Parameters
        ----------
        alphas : tuple of floats, default=(0.01, 0.1, 1.0, 10.0)
            Grid of alpha values to try in cross-validation
        cv : int, default=5
            Number of folds for cross-validation
        fit_intercept : bool, default=True
            Whether to fit an intercept term
        **kwargs : dict
            Additional parameters (for compatibility)
        """
        super().__init__(
            name="Ridge",
            alphas=alphas,
            cv=cv,
            fit_intercept=fit_intercept,
            **kwargs
        )
        self.alphas = alphas
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.model_0 = None
        self.model_1 = None
        self.alpha_0 = None
        self.alpha_1 = None

    def fit(self, X: np.ndarray, T: np.ndarray, M: np.ndarray, **kwargs) -> 'RidgeMediator':
        """
        Fit separate Ridge models for each treatment group.

        Alpha is selected independently for each group via CV.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix
        T : np.ndarray, shape (n_samples,)
            Treatment assignment (0 or 1)
        M : np.ndarray, shape (n_samples,)
            Mediator values

        Returns
        -------
        self : RidgeMediator
            Fitted estimator
        """
        # Validate inputs
        check_inputs(X, T, M)

        # Track performance
        tracemalloc.start()
        start_time = time.time()

        # Store feature dimension
        self.feature_dim = X.shape[1]

        # Split data by treatment
        idx_0 = (T == 0)
        idx_1 = (T == 1)

        X_0, M_0 = X[idx_0], M[idx_0]
        X_1, M_1 = X[idx_1], M[idx_1]

        # Fit Ridge model for control group (T=0) with CV for alpha
        self.model_0 = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            fit_intercept=self.fit_intercept
        )
        self.model_0.fit(X_0, M_0)
        self.alpha_0 = self.model_0.alpha_

        # Fit Ridge model for treated group (T=1) with CV for alpha
        self.model_1 = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            fit_intercept=self.fit_intercept
        )
        self.model_1.fit(X_1, M_1)
        self.alpha_1 = self.model_1.alpha_

        # Mark as fitted
        self.is_fitted = True

        # Track performance
        self.training_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        self.memory_usage = peak / 1024 / 1024  # Convert to MB
        tracemalloc.stop()

        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mediator CATE: τ̂_M(X) = μ̂₁(X) - μ̂₀(X).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
            Predicted mediator CATE
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        check_prediction_inputs(X, expected_dim=self.feature_dim)

        # Predict potential outcomes
        mu_0 = self.model_0.predict(X)
        mu_1 = self.model_1.predict(X)

        # CATE is the difference
        cate = mu_1 - mu_0

        return cate

    def predict_potential_outcomes(self, X: np.ndarray) -> tuple:
        """
        Predict potential outcomes μ̂₀(X) and μ̂₁(X).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix

        Returns
        -------
        mu_0 : np.ndarray, shape (n_samples,)
            Predicted E[M|T=0,X]
        mu_1 : np.ndarray, shape (n_samples,)
            Predicted E[M|T=1,X]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        check_prediction_inputs(X, expected_dim=self.feature_dim)

        mu_0 = self.model_0.predict(X)
        mu_1 = self.model_1.predict(X)

        return mu_0, mu_1

    def get_selected_alphas(self) -> dict:
        """
        Get the selected alpha values from cross-validation.

        Returns
        -------
        alphas : dict
            Dictionary with 'alpha_0' and 'alpha_1'
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        return {
            'alpha_0': self.alpha_0,
            'alpha_1': self.alpha_1
        }

    def get_coefficients(self) -> dict:
        """
        Get fitted coefficients for both models.

        Returns
        -------
        coefficients : dict
            Dictionary with 'beta_0', 'beta_1', 'intercept_0', 'intercept_1',
            'alpha_0', 'alpha_1'
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        coeffs = {
            'beta_0': self.model_0.coef_,
            'beta_1': self.model_1.coef_,
            'alpha_0': self.alpha_0,
            'alpha_1': self.alpha_1,
        }

        if self.fit_intercept:
            coeffs['intercept_0'] = self.model_0.intercept_
            coeffs['intercept_1'] = self.model_1.intercept_

        return coeffs
