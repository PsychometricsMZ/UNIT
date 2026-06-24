#!/usr/bin/env python
"""
TARNet estimator for mediator CATE using CATENets SNet1.

Wraps CATENets' SNet1 (TARNet/CFRNet) implementation for mediator CATE estimation.
TARNet uses shared representation layers with separate hypothesis heads for T=0 and T=1.
"""

import numpy as np
import time
import tracemalloc
import warnings
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add vendored CATENets to path.
# Layout: src/rpmdeep/estimators/tarnet_mediator.py
#   parents[0] = estimators/
#   parents[1] = rpmdeep/
#   parents[2] = src/
#   parents[3] = repo root → third_party/catenets/
catenets_path = Path(__file__).resolve().parents[3] / 'third_party' / 'catenets'
if catenets_path.exists() and str(catenets_path) not in sys.path:
    sys.path.insert(0, str(catenets_path))

try:
    from catenets.models.jax import SNet1
    CATENETS_AVAILABLE = True
except ImportError as e:
    CATENETS_AVAILABLE = False
    warnings.warn(f"CATENets not available: {e}. TARNet estimator will not work.")

from .base import (
    MediatorCATEEstimator,
    check_inputs,
    check_prediction_inputs
)


class TARNetMediator(MediatorCATEEstimator):
    """
    TARNet estimator for mediator CATE (using CATENets SNet1).

    Architecture:
    - Shared representation layers: n_layers_r × n_units_r
    - Separate hypothesis heads for μ₀(X) and μ₁(X): n_layers_out × n_units_out

    CATE prediction: τ̂_M(X) = μ̂₁(X) - μ̂₀(X)

    Reference:
    Shalit et al. (2017). Estimating individual treatment effect:
    generalization bounds and algorithms. ICML.
    """

    def __init__(
        self,
        n_layers_r: int = 3,
        n_units_r: int = 200,
        n_layers_out: int = 2,
        n_units_out: int = 100,
        penalty_l2: float = 0.01,
        step_size: float = 0.001,
        n_iter: int = 100,
        batch_size: int = 100,
        val_split_prop: float = 0.2,
        early_stopping: bool = True,
        patience: int = 10,
        n_iter_min: int = 10,
        nonlin: str = 'elu',
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize TARNet Mediator estimator.

        Parameters
        ----------
        n_layers_r : int, default=3
            Number of shared representation layers
        n_units_r : int, default=200
            Number of units in each representation layer
        n_layers_out : int, default=2
            Number of hypothesis layers (per head)
        n_units_out : int, default=100
            Number of units in each hypothesis layer
        penalty_l2 : float, default=0.01
            L2 regularization penalty
        step_size : float, default=0.001
            Learning rate
        n_iter : int, default=100
            Maximum number of training iterations
        batch_size : int, default=100
            Batch size for training
        val_split_prop : float, default=0.2
            Proportion of data to use for validation
        early_stopping : bool, default=True
            Whether to use early stopping
        patience : int, default=10
            Patience for early stopping
        n_iter_min : int, default=10
            Minimum number of iterations before early stopping
        nonlin : str, default='elu'
            Nonlinearity ('elu', 'relu', 'sigmoid')
        seed : int, default=42
            Random seed
        """
        if not CATENETS_AVAILABLE:
            raise ImportError("CATENets is required for TARNet estimator")

        super().__init__(
            name="TARNet",
            n_layers_r=n_layers_r,
            n_units_r=n_units_r,
            n_layers_out=n_layers_out,
            n_units_out=n_units_out,
            penalty_l2=penalty_l2,
            step_size=step_size,
            n_iter=n_iter,
            batch_size=batch_size,
            val_split_prop=val_split_prop,
            early_stopping=early_stopping,
            patience=patience,
            n_iter_min=n_iter_min,
            nonlin=nonlin,
            seed=seed,
            **kwargs
        )

        self.n_layers_r = n_layers_r
        self.n_units_r = n_units_r
        self.n_layers_out = n_layers_out
        self.n_units_out = n_units_out
        self.penalty_l2 = penalty_l2
        self.step_size = step_size
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.val_split_prop = val_split_prop
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_iter_min = n_iter_min
        self.nonlin = nonlin
        self.seed = seed

        self.model = None
        # Scalers set during fit
        self.x_scaler_ = None
        self.m_mean_   = None
        self.m_std_    = None

    def fit(self, X: np.ndarray, T: np.ndarray, M: np.ndarray, **kwargs) -> 'TARNetMediator':
        """
        Fit TARNet model.

        Preprocessing applied before passing to CATENets (which does not normalize):
          - X: StandardScaler (zero mean, unit variance per feature).
            CATENets uses L2 regularization on weights; non-unit-variance features
            force large weights to compensate, effectively over-regularizing.
          - M: standardized to zero mean / unit variance before training,
            then denormalized at prediction. CATE = (μ̂₁ - μ̂₀) × m_std_, so the
            mean cancels and only the scale matters for the CATE prediction.

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
        self : TARNetMediator
            Fitted estimator
        """
        # Validate inputs
        check_inputs(X, T, M)

        # Track performance
        tracemalloc.start()
        start_time = time.time()

        # Store feature dimension
        self.feature_dim = X.shape[1]

        # ── Preprocessing ──
        # Standardize X (CATENets has no internal normalization)
        self.x_scaler_ = StandardScaler()
        X_fit = self.x_scaler_.fit_transform(X)

        # Standardize M target (reduces gradient scale sensitivity)
        self.m_mean_ = float(np.mean(M))
        self.m_std_  = float(np.std(M))
        if self.m_std_ < 1e-8:
            self.m_std_ = 1.0
        M_fit = (M - self.m_mean_) / self.m_std_

        # Create SNet1 model
        self.model = SNet1(
            binary_y=False,  # Continuous mediator
            n_layers_r=self.n_layers_r,
            n_units_r=self.n_units_r,
            n_layers_out=self.n_layers_out,
            n_units_out=self.n_units_out,
            penalty_l2=self.penalty_l2,
            step_size=self.step_size,
            n_iter=self.n_iter,
            batch_size=self.batch_size,
            val_split_prop=self.val_split_prop,
            early_stopping=self.early_stopping,
            patience=self.patience,
            n_iter_min=self.n_iter_min,
            nonlin=self.nonlin,
            seed=self.seed
        )

        # Fit model — CATENets expects fit(X, y, w) where y=outcome, w=treatment
        self.model.fit(X=X_fit, y=M_fit, w=T)

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

        The model was trained on standardized M, so the raw CATE is in M/m_std
        units. Multiply by m_std_ to recover the original M scale.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
            Predicted mediator CATE in original M scale
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        check_prediction_inputs(X, expected_dim=self.feature_dim)

        # Apply same X transformation used at fit time
        X_pred = self.x_scaler_.transform(X)

        # CATENets' predict method returns CATE directly (in standardized M units)
        cate = self.model.predict(X_pred)

        # Convert to numpy array (might be JAX array) and rescale to M units
        cate = np.array(cate).flatten() * self.m_std_

        return cate

    def predict_potential_outcomes(self, X: np.ndarray) -> tuple:
        """
        Predict potential outcomes μ̂₀(X) and μ̂₁(X) in original M scale.

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

        X_pred = self.x_scaler_.transform(X)

        # CATENets' predict method with return_po=True returns (cate, mu_0, mu_1)
        cate, mu_0, mu_1 = self.model.predict(X_pred, return_po=True)

        # Convert to numpy arrays and rescale to original M units
        mu_0 = np.array(mu_0).flatten() * self.m_std_ + self.m_mean_
        mu_1 = np.array(mu_1).flatten() * self.m_std_ + self.m_mean_

        return mu_0, mu_1
