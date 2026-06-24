"""
Core data generating process components for mediation analysis simulation.

Implements covariate generation using Gaussian copulas with GMM marginals
and AR(1) correlation structures.
"""

import numpy as np
from scipy import stats
from scipy.linalg import cholesky
from typing import Optional, List
from .gaussian_mixture import GaussianMixture1D, create_random_gmm


class CovariateGenerator:
    """
    Generate covariates X using Gaussian copula with GMM marginals and AR(1) correlation.

    Covariate structure:
    - AR(1) correlation: Σ_ij = ρ^|i-j|
    - Non-Gaussian marginals via Gaussian Mixture Models
    - Gaussian copula to preserve correlation structure

    The generation process:
    1. Sample from multivariate normal with AR(1) correlation matrix
    2. Transform to uniform [0,1] via Gaussian CDF
    3. Transform to target distribution via inverse CDF of GMM

    Parameters
    ----------
    n_features : int
        Dimensionality of covariates (p)
    correlation_rho : float, default 0.3
        AR(1) correlation parameter ρ (typically in [0, 0.5])
    n_gmm_components_range : tuple, default (3, 7)
        Range for number of GMM components per covariate
    scale_range : tuple, default (0, 5)
        Range to scale covariate values to
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> cov_gen = CovariateGenerator(n_features=25, correlation_rho=0.3, seed=42)
    >>> X = cov_gen.generate(n_samples=1000)
    >>> print(X.shape)  # (1000, 25)
    >>> print(np.corrcoef(X.T)[0, 1])  # Should be close to ρ
    """

    def __init__(
        self,
        n_features: int,
        correlation_rho: float = 0.3,
        n_gmm_components_range: tuple = (3, 7),
        scale_range: tuple = (0, 5),
        seed: Optional[int] = None
    ):
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if not 0 <= correlation_rho < 1:
            raise ValueError("correlation_rho must be in [0, 1)")

        self.n_features = n_features
        self.correlation_rho = correlation_rho
        self.n_gmm_components_range = n_gmm_components_range
        self.scale_range = scale_range
        self.seed = seed

        # Create AR(1) correlation matrix
        self.correlation_matrix = self._create_ar1_correlation()

        # Compute Cholesky decomposition for efficient sampling
        self.cholesky_factor = cholesky(self.correlation_matrix, lower=True)

        # Create GMM for each feature
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        self.gmm_models = []
        for i in range(n_features):
            # Random number of components for each feature
            n_comp = rng.randint(
                n_gmm_components_range[0],
                n_gmm_components_range[1] + 1
            )

            # Create GMM with feature-specific seed
            feature_seed = None if seed is None else seed + i
            gmm = create_random_gmm(
                n_components=n_comp,
                scale_range=scale_range,
                seed=feature_seed
            )
            self.gmm_models.append(gmm)

    def _create_ar1_correlation(self) -> np.ndarray:
        """
        Create AR(1) correlation matrix: Σ_ij = ρ^|i-j|.

        Returns
        -------
        corr_matrix : np.ndarray, shape (n_features, n_features)
            AR(1) correlation matrix
        """
        indices = np.arange(self.n_features)
        diff_matrix = np.abs(indices[:, None] - indices[None, :])
        corr_matrix = self.correlation_rho ** diff_matrix
        return corr_matrix

    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate covariate matrix X.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate (n)
        seed : int, optional
            Random seed (overrides object seed if provided)

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
            Generated covariates with AR(1) correlation and GMM marginals
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        if seed is not None:
            rng = np.random.RandomState(seed)
        elif self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random

        # Step 1: Sample from multivariate standard normal with AR(1) correlation
        # Z ~ N(0, Σ) where Σ_ij = ρ^|i-j|
        standard_normal = rng.randn(n_samples, self.n_features)
        correlated_normal = standard_normal @ self.cholesky_factor.T

        # Step 2: Transform to uniform [0, 1] via Gaussian CDF
        uniform = stats.norm.cdf(correlated_normal)

        # Step 3: Transform to target GMM distributions via inverse CDF
        # This is done by quantile matching using the GMM samples
        X = np.zeros((n_samples, self.n_features))

        for j in range(self.n_features):
            # Generate a large sample from the GMM to use for quantile matching
            gmm_sample = self.gmm_models[j].sample(n_samples=10000, seed=seed)
            gmm_sample_sorted = np.sort(gmm_sample)

            # For each uniform value, find the corresponding quantile in the GMM
            quantile_indices = (uniform[:, j] * (len(gmm_sample_sorted) - 1)).astype(int)
            quantile_indices = np.clip(quantile_indices, 0, len(gmm_sample_sorted) - 1)

            X[:, j] = gmm_sample_sorted[quantile_indices]

        return X

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get the theoretical correlation matrix.

        Returns
        -------
        corr_matrix : np.ndarray, shape (n_features, n_features)
            AR(1) correlation matrix
        """
        return self.correlation_matrix.copy()

    def get_gmm_params(self, feature_idx: int) -> dict:
        """
        Get GMM parameters for a specific feature.

        Parameters
        ----------
        feature_idx : int
            Index of the feature (0 to n_features-1)

        Returns
        -------
        params : dict
            GMM parameters for the specified feature
        """
        if not 0 <= feature_idx < self.n_features:
            raise ValueError(f"feature_idx must be in [0, {self.n_features})")

        return self.gmm_models[feature_idx].get_params()

    def __repr__(self) -> str:
        return (
            f"CovariateGenerator(n_features={self.n_features}, "
            f"correlation_rho={self.correlation_rho})"
        )


def verify_correlation_structure(X: np.ndarray, target_rho: float) -> dict:
    """
    Verify that generated covariates have the expected correlation structure.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Generated covariate matrix
    target_rho : float
        Target AR(1) correlation parameter

    Returns
    -------
    diagnostics : dict
        Dictionary containing:
        - empirical_corr_matrix: Observed correlation matrix
        - target_corr_matrix: Expected AR(1) correlation matrix
        - max_abs_error: Maximum absolute error in correlations
        - mean_abs_error: Mean absolute error in correlations
        - is_valid: Boolean indicating if correlations are reasonable (MAE < 0.1)
    """
    n_features = X.shape[1]

    # Compute empirical correlation
    empirical_corr = np.corrcoef(X.T)

    # Compute target correlation
    indices = np.arange(n_features)
    diff_matrix = np.abs(indices[:, None] - indices[None, :])
    target_corr = target_rho ** diff_matrix

    # Compute errors
    errors = np.abs(empirical_corr - target_corr)

    # Exclude diagonal (which is always 1)
    off_diagonal_mask = ~np.eye(n_features, dtype=bool)
    off_diagonal_errors = errors[off_diagonal_mask]

    diagnostics = {
        'empirical_corr_matrix': empirical_corr,
        'target_corr_matrix': target_corr,
        'max_abs_error': off_diagonal_errors.max(),
        'mean_abs_error': off_diagonal_errors.mean(),
        'is_valid': off_diagonal_errors.mean() < 0.1  # Reasonable threshold
    }

    return diagnostics
