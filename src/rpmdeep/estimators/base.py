#!/usr/bin/env python
"""
Base class for mediator CATE estimators.

Defines the interface that all estimators must implement for consistent
evaluation and comparison.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional


class MediatorCATEEstimator(ABC):
    """
    Abstract base class for estimators of mediator CATE: τ_M(X) = E[M|T=1,X] - E[M|T=0,X]

    All estimators must implement:
    - fit(X, T, M): Train the model
    - predict_cate(X): Predict τ_M(X) for new samples

    Optional methods:
    - predict_potential_outcomes(X): Return (μ_0(X), μ_1(X))
    """

    def __init__(self, name: str = "BaseEstimator", **kwargs):
        """
        Initialize estimator.

        Parameters
        ----------
        name : str
            Name of the estimator for reporting
        **kwargs : dict
            Additional hyperparameters specific to the estimator
        """
        self.name = name
        self.is_fitted = False
        self.hyperparameters = kwargs
        self.feature_dim = None
        self.training_time = None
        self.memory_usage = None

    @abstractmethod
    def fit(self, X: np.ndarray, T: np.ndarray, M: np.ndarray, **kwargs) -> 'MediatorCATEEstimator':
        """
        Fit the estimator on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix
        T : np.ndarray, shape (n_samples,)
            Treatment assignment (0 or 1)
        M : np.ndarray, shape (n_samples,)
            Mediator values
        **kwargs : dict
            Additional fitting parameters

        Returns
        -------
        self : MediatorCATEEstimator
            Fitted estimator
        """
        pass

    @abstractmethod
    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        """
        Predict mediator CATE for new samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix

        Returns
        -------
        cate : np.ndarray, shape (n_samples,)
            Predicted τ_M(X) = E[M|T=1,X] - E[M|T=0,X]
        """
        pass

    def predict_potential_outcomes(self, X: np.ndarray) -> tuple:
        """
        Predict potential outcomes μ_0(X) and μ_1(X).

        Optional method. If not implemented by subclass, raises NotImplementedError.

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
        raise NotImplementedError(
            f"{self.name} does not implement predict_potential_outcomes()"
        )

    def get_params(self) -> Dict[str, Any]:
        """
        Get estimator hyperparameters.

        Returns
        -------
        params : dict
            Dictionary of hyperparameters
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_dim': self.feature_dim,
            **self.hyperparameters
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """
        Get performance information (time, memory).

        Returns
        -------
        info : dict
            Dictionary with training_time and memory_usage
        """
        return {
            'training_time': self.training_time,
            'memory_usage': self.memory_usage
        }

    def __repr__(self) -> str:
        params_str = ', '.join(f'{k}={v}' for k, v in self.hyperparameters.items())
        return f"{self.name}({params_str})"


def check_inputs(X: np.ndarray, T: np.ndarray, M: np.ndarray) -> None:
    """
    Validate input data for estimator fitting.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix
    T : np.ndarray
        Treatment assignment
    M : np.ndarray
        Mediator values

    Raises
    ------
    ValueError
        If inputs have invalid shapes or values
    """
    # Check dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    n = X.shape[0]

    if T.ndim != 1 or len(T) != n:
        raise ValueError(f"T must be 1D with length {n}, got shape {T.shape}")

    if M.ndim != 1 or len(M) != n:
        raise ValueError(f"M must be 1D with length {n}, got shape {M.shape}")

    # Check treatment is binary
    unique_t = np.unique(T)
    if not np.all(np.isin(unique_t, [0, 1])):
        raise ValueError(f"T must be binary (0/1), got unique values: {unique_t}")

    # Check for NaN/Inf
    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values")

    if np.any(~np.isfinite(T)):
        raise ValueError("T contains NaN or Inf values")

    if np.any(~np.isfinite(M)):
        raise ValueError("M contains NaN or Inf values")

    # Check we have both treatment groups
    if len(unique_t) != 2:
        raise ValueError(f"T must contain both 0 and 1, got: {unique_t}")


def check_prediction_inputs(X: np.ndarray, expected_dim: Optional[int] = None) -> None:
    """
    Validate input data for prediction.

    Parameters
    ----------
    X : np.ndarray
        Covariate matrix
    expected_dim : int, optional
        Expected number of features (from training)

    Raises
    ------
    ValueError
        If inputs have invalid shapes or values
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    if expected_dim is not None and X.shape[1] != expected_dim:
        raise ValueError(
            f"X has {X.shape[1]} features, expected {expected_dim}"
        )

    if np.any(~np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values")
