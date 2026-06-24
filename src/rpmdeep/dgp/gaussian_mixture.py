"""
1D Gaussian Mixture Models for generating non-Gaussian marginal distributions
(used with Gaussian copulas to build multivariate covariate distributions).
"""

import numpy as np
from typing import Optional, List


class GaussianMixture1D:
    """
    One-dimensional Gaussian Mixture Model for generating non-Gaussian marginals.

    Used in conjunction with Gaussian copulas to create multivariate distributions
    with complex marginal structures and specified correlation patterns.

    Parameters
    ----------
    n_components : int
        Number of mixture components (typically 3-7 for complex distributions)
    means : np.ndarray or list, optional
        Mean of each component. If None, randomly generated in [-2, 2]
    sds : np.ndarray or list, optional
        Standard deviation of each component. If None, randomly generated in [0.5, 1.5]
    weights : np.ndarray or list, optional
        Mixing weights (must sum to 1). If None, randomly generated and normalized
    scale_range : tuple, default (0, 5)
        Range to scale the output samples to
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> gmm = GaussianMixture1D(n_components=5, seed=42)
    >>> samples = gmm.sample(1000)
    >>> print(samples.min(), samples.max())  # Should be roughly in [0, 5]
    """

    def __init__(
        self,
        n_components: int,
        means: Optional[np.ndarray] = None,
        sds: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
        scale_range: tuple = (0, 5),
        seed: Optional[int] = None
    ):
        if n_components < 1:
            raise ValueError("n_components must be at least 1")

        self.n_components = n_components
        self.scale_range = scale_range
        self.seed = seed

        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Initialize parameters
        if means is None:
            # Random means spread across the space
            self.means = rng.uniform(-2, 2, size=n_components)
        else:
            self.means = np.array(means)
            if len(self.means) != n_components:
                raise ValueError(f"means must have length {n_components}")

        if sds is None:
            # Random standard deviations (not too small, not too large)
            self.sds = rng.uniform(0.5, 1.5, size=n_components)
        else:
            self.sds = np.array(sds)
            if len(self.sds) != n_components:
                raise ValueError(f"sds must have length {n_components}")
            if np.any(self.sds <= 0):
                raise ValueError("All standard deviations must be positive")

        if weights is None:
            # Random weights, normalized to sum to 1
            raw_weights = rng.uniform(0.5, 2.0, size=n_components)
            self.weights = raw_weights / raw_weights.sum()
        else:
            self.weights = np.array(weights)
            if len(self.weights) != n_components:
                raise ValueError(f"weights must have length {n_components}")
            if not np.isclose(self.weights.sum(), 1.0):
                raise ValueError("Weights must sum to 1")
            if np.any(self.weights < 0):
                raise ValueError("All weights must be non-negative")

        self._fitted = True

    def sample(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Draw samples from the Gaussian mixture.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        seed : int, optional
            Random seed (overrides the object's seed if provided)

        Returns
        -------
        samples : np.ndarray, shape (n_samples,)
            Samples scaled to the specified range
        """
        if n_samples < 1:
            raise ValueError("n_samples must be at least 1")

        if seed is not None:
            rng = np.random.RandomState(seed)
        elif self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random

        # Sample component assignments
        components = rng.choice(
            self.n_components,
            size=n_samples,
            p=self.weights
        )

        # Sample from assigned components
        samples = np.zeros(n_samples)
        for i in range(self.n_components):
            mask = (components == i)
            n_i = mask.sum()
            if n_i > 0:
                samples[mask] = rng.normal(
                    loc=self.means[i],
                    scale=self.sds[i],
                    size=n_i
                )

        # Scale to target range
        samples_scaled = self._scale_to_range(samples)

        return samples_scaled

    def _scale_to_range(self, samples: np.ndarray) -> np.ndarray:
        """
        Scale samples to the target range using min-max scaling.

        Parameters
        ----------
        samples : np.ndarray
            Raw samples from the mixture

        Returns
        -------
        scaled : np.ndarray
            Samples scaled to self.scale_range
        """
        # Min-max scaling
        s_min, s_max = samples.min(), samples.max()

        if s_max - s_min < 1e-10:
            # All values are the same, return middle of range
            return np.full_like(samples, np.mean(self.scale_range))

        # Scale to [0, 1]
        normalized = (samples - s_min) / (s_max - s_min)

        # Scale to target range
        range_min, range_max = self.scale_range
        scaled = normalized * (range_max - range_min) + range_min

        return scaled

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the probability density function.

        Note: This returns the PDF before scaling. For the scaled distribution,
        this is only approximate.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the PDF

        Returns
        -------
        density : np.ndarray
            PDF values at each point in x
        """
        x = np.atleast_1d(x)
        density = np.zeros_like(x, dtype=float)

        for i in range(self.n_components):
            # Gaussian PDF: (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))
            component_density = (
                1.0 / (self.sds[i] * np.sqrt(2 * np.pi)) *
                np.exp(-0.5 * ((x - self.means[i]) / self.sds[i])**2)
            )
            density += self.weights[i] * component_density

        return density

    def get_params(self) -> dict:
        """
        Get the parameters of the mixture model.

        Returns
        -------
        params : dict
            Dictionary containing means, sds, weights, and other parameters
        """
        return {
            'n_components': self.n_components,
            'means': self.means.copy(),
            'sds': self.sds.copy(),
            'weights': self.weights.copy(),
            'scale_range': self.scale_range
        }

    def __repr__(self) -> str:
        return (
            f"GaussianMixture1D(n_components={self.n_components}, "
            f"scale_range={self.scale_range})"
        )


def create_random_gmm(
    n_components: int = 5,
    scale_range: tuple = (0, 5),
    seed: Optional[int] = None
) -> GaussianMixture1D:
    """
    Factory function to create a random Gaussian mixture model.

    Parameters
    ----------
    n_components : int, default 5
        Number of mixture components
    scale_range : tuple, default (0, 5)
        Range to scale samples to
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    gmm : GaussianMixture1D
        A randomly initialized Gaussian mixture model

    Examples
    --------
    >>> gmm = create_random_gmm(n_components=7, seed=123)
    >>> samples = gmm.sample(1000)
    """
    return GaussianMixture1D(
        n_components=n_components,
        means=None,
        sds=None,
        weights=None,
        scale_range=scale_range,
        seed=seed
    )
