"""
Naive Baron-Kenny estimator for structural parameters.

Two-regression approach:
1) M ~ T + X
2) Y ~ T + M + X

Mapping to structural parameters:
    theta_1 = coefficient on T
    theta_2 = coefficient on M
"""

from typing import Dict, Tuple
import numpy as np


class BaronKennyEstimator:
    """
    Naive Baron-Kenny estimator for θ using linear regressions.

    Attributes
    ----------
    theta_ : np.ndarray
        Estimated [theta_1, theta_2] from Y ~ T + M + X
    theta_se_ : np.ndarray
        Standard errors for theta estimates (OLS)
    theta_var_ : np.ndarray
        Variance-covariance matrix for theta
    coef_mediator_ : np.ndarray
        Coefficients from M ~ T + X (intercept, T, X...)
    coef_outcome_ : np.ndarray
        Coefficients from Y ~ T + M + X (intercept, T, M, X...)
    """

    def __init__(self, include_intercept: bool = True):
        self.include_intercept = include_intercept
        self.theta_ = None
        self.theta_se_ = None
        self.theta_var_ = None
        self.coef_mediator_ = None
        self.coef_outcome_ = None
        self.n_samples_ = None
        self.n_features_ = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray
    ) -> "BaronKennyEstimator":
        X = np.asarray(X)
        T = np.asarray(T).ravel()
        M = np.asarray(M).ravel()
        Y = np.asarray(Y).ravel()

        n = X.shape[0]
        if not all(arr.shape[0] == n for arr in [T, M, Y]):
            raise ValueError("All inputs must have the same number of samples")

        self.n_samples_ = n
        self.n_features_ = X.shape[1]

        # Stage 1: M ~ T + X
        Z_m = self._design_matrix_mediator(X, T)
        self.coef_mediator_ = self._ols_fit(Z_m, M)

        # Stage 2: Y ~ T + M + X
        Z_y = self._design_matrix_outcome(X, T, M)
        self.coef_outcome_, cov_beta = self._ols_fit_with_cov(Z_y, Y)

        # theta_1 = coef on T, theta_2 = coef on M
        idx_t, idx_m = self._outcome_coef_indices()
        theta_1 = self.coef_outcome_[idx_t]
        theta_2 = self.coef_outcome_[idx_m]
        self.theta_ = np.array([theta_1, theta_2], dtype=float)

        # Extract variance/covariance for theta entries
        self.theta_var_ = cov_beta[np.ix_([idx_t, idx_m], [idx_t, idx_m])]
        self.theta_se_ = np.sqrt(np.diag(self.theta_var_))

        return self

    def _design_matrix_mediator(self, X: np.ndarray, T: np.ndarray) -> np.ndarray:
        if self.include_intercept:
            return np.column_stack([np.ones(X.shape[0]), T, X])
        return np.column_stack([T, X])

    def _design_matrix_outcome(self, X: np.ndarray, T: np.ndarray, M: np.ndarray) -> np.ndarray:
        if self.include_intercept:
            return np.column_stack([np.ones(X.shape[0]), T, M, X])
        return np.column_stack([T, M, X])

    def _outcome_coef_indices(self) -> Tuple[int, int]:
        if self.include_intercept:
            idx_t = 1
            idx_m = 2
        else:
            idx_t = 0
            idx_m = 1
        return idx_t, idx_m

    def _ols_fit(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        return coef

    def _ols_fit_with_cov(self, Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coef, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
        residuals = y - Z @ coef
        n, p = Z.shape
        dof = max(n - p, 1)
        sigma2 = (residuals @ residuals) / dof
        XtX_inv = np.linalg.inv(Z.T @ Z)
        cov_beta = sigma2 * XtX_inv
        return coef, cov_beta

    def get_confidence_intervals(self, alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        if self.theta_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        z = _z_critical(alpha)
        return {
            'theta_1': (self.theta_[0] - z * self.theta_se_[0],
                        self.theta_[0] + z * self.theta_se_[0]),
            'theta_2': (self.theta_[1] - z * self.theta_se_[1],
                        self.theta_[1] + z * self.theta_se_[1]),
        }

    def summary(self) -> str:
        if self.theta_ is None:
            return "Model not fitted. Call fit() first."
        ci = self.get_confidence_intervals()
        lines = [
            "=" * 70,
            "Naive Baron-Kenny Results: Y ~ T + M + X",
            "=" * 70,
            f"theta_1 (T):   {self.theta_[0]:>10.4f}  SE={self.theta_se_[0]:.4f}  "
            f"CI=[{ci['theta_1'][0]:.3f}, {ci['theta_1'][1]:.3f}]",
            f"theta_2 (M):   {self.theta_[1]:>10.4f}  SE={self.theta_se_[1]:.4f}  "
            f"CI=[{ci['theta_2'][0]:.3f}, {ci['theta_2'][1]:.3f}]",
            "=" * 70,
        ]
        return "\n".join(lines)


def _z_critical(alpha: float) -> float:
    # Exact normal quantile for the common alpha=0.05 case
    if alpha == 0.05:
        return 1.959963984540054
    # Normal quantile for other alpha values, computed as sqrt(2) * erf^-1(1-alpha)
    return np.sqrt(2) * _erf_inv(1 - alpha)


def _erf_inv(x: float) -> float:
    # Approximation of inverse error function
    a = 0.147
    sign = 1 if x >= 0 else -1
    ln = np.log(1 - x * x)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sign * np.sqrt(np.sqrt(first * first - second) - first)
