"""
Stage 2 G-estimation for structural mean model parameters via alternating
block coordinate descent, using Stage 1 mediator CATE estimates as weights.

References
----------
- Zheng & Zhou (2015): Causal mediation analysis in the multilevel
  intervention and multicomponent mediator case
"""

import numpy as np
from typing import Optional, Dict, Tuple, Union
from scipy import stats
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings


class GEstimator:
    """
    G-estimation for structural mean model parameters.

    Estimates θ in the outcome model:
        Y = g(X) + θ₁·T + θ₂·M + ε_Y

    Using alternating block coordinate descent:
    1. Update g(X) by fitting regression on residuals Y - H@θ
    2. Update θ by solving weighted estimating equation

    Key insight from Zheng-Zhou (2015):
    - use_centering=True: Uses proper G-estimation with E[A|V] = 0
    - This handles unmeasured M-Y confounding correctly
    - The centered weight A(Z,V) = (Z - e(V)) · W(V) where W(V) = (1, τ(V))

    Parameters
    ----------
    g_model_type : str, default='ridge'
        Type of model for baseline g(X):
        - 'ridge': RidgeCV with automatic regularization
        - 'neural_network': MLPRegressor with early stopping
    max_iterations : int, default=100
        Maximum number of alternating iterations
    convergence_tol : float, default=1e-4
        Convergence tolerance for ||θ^{t+1} - θ^{t}||.
        Note: For neural_network baseline, convergence may require
        a slightly higher tolerance (e.g., 1e-3) due to
        stochasticity in the optimization.
    n_structural_params : int, default=2
        Number of structural parameters (default: θ₁ for T, θ₂ for M)
    use_centering : bool, default=True
        Whether to use Zheng-Zhou centered weights (recommended).
        - True: A = (T - e(X)) · W(V) satisfies E[A|V] = 0
        - False: Use standard weighted least squares
    n_folds_g : int, default=1
        Number of folds for K-fold cross-fitting of the Stage 2 g-model
        (DML / Chernozhukov 2018 "partialing out"). When > 1, g(X) is fit
        on K-1 folds and predicts out-of-fold residuals, eliminating
        regularisation bias from the baseline model. Recommended: 5.
        Setting to 1 (default) uses the original in-sample g-model.
    normalize_centered_weights : bool, default=False
        Whether to normalize weights (tau_hat) before constructing
        centered weight matrix A. This can help when Stage 1 methods
        produce tau_hat on different scales. Only applies when use_centering=True.
    weight_transform : str, default='squared'
        How to transform Stage 1 weights (only used when use_centering=False):
        - 'squared': W = τ̂² (always positive)
        - 'absolute': W = |τ̂|
        - 'raw': W = τ̂ (can be negative)
    weight_clip_percentile : float, default=99
        Percentile for clipping extreme weights
    variance_method : str, default='sandwich'
        Variance estimation method:
        - 'sandwich': Analytical sandwich estimator
        - 'bootstrap': Bootstrap variance (more robust)
    n_bootstrap : int, default=500
        Number of bootstrap samples (only used if variance_method='bootstrap')
    seed : int, default=42
        Random seed for reproducibility

    Attributes
    ----------
    theta_ : np.ndarray
        Estimated structural parameters
    theta_se_ : np.ndarray
        Standard errors for theta estimates
    theta_var_ : np.ndarray
        Variance-covariance matrix for theta
    g_model_ : object
        Fitted baseline model for g(X)
    n_iterations_ : int
        Number of iterations until convergence
    converged_ : bool
        Whether algorithm converged
    convergence_history_ : list
        History of ||θ^{t+1} - θ^{t}|| at each iteration
    propensity_ : np.ndarray
        Estimated propensity scores P(T=1|X)
    """

    def __init__(
        self,
        g_model_type: str = 'ridge',
        max_iterations: int = 100,
        convergence_tol: float = 1e-4,
        n_structural_params: int = 2,
        use_centering: bool = True,
        n_folds_g: int = 1,
        normalize_centered_weights: bool = False,
        weight_transform: str = 'squared',
        weight_clip_percentile: float = 99,
        variance_method: str = 'sandwich',
        n_bootstrap: int = 500,
        seed: int = 42
    ):
        if g_model_type not in ('ridge', 'neural_network'):
            raise ValueError(f"g_model_type must be 'ridge' or 'neural_network', got {g_model_type}")
        if weight_transform not in ('squared', 'absolute', 'raw'):
            raise ValueError(f"weight_transform must be 'squared', 'absolute', or 'raw'")
        if variance_method not in ('sandwich', 'bootstrap'):
            raise ValueError(f"variance_method must be 'sandwich' or 'bootstrap'")

        self.g_model_type = g_model_type
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.n_structural_params = n_structural_params
        self.use_centering = use_centering
        self.n_folds_g = n_folds_g
        self.normalize_centered_weights = normalize_centered_weights
        self.weight_transform = weight_transform
        self.weight_clip_percentile = weight_clip_percentile
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        # Attributes set during fit
        self.theta_ = None
        self.theta_se_ = None
        self.theta_var_ = None
        self.g_model_ = None
        self.n_iterations_ = None
        self.converged_ = None
        self.weakly_identified_ = None
        self.convergence_history_ = None
        self.propensity_ = None
        self._scaler = None
        self._X = None
        self._T = None
        self._M = None
        self._Y = None
        self._weights = None

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        weights: np.ndarray,
        propensity: np.ndarray = None
    ) -> 'GEstimator':
        """
        Fit G-estimation model using alternating optimization.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Covariate matrix
        T : np.ndarray, shape (n_samples,)
            Treatment assignment (binary 0/1)
        M : np.ndarray, shape (n_samples,)
            Mediator values
        Y : np.ndarray, shape (n_samples,)
            Outcome values
        weights : np.ndarray, shape (n_samples,)
            Stage 1 mediator CATE estimates τ̂_M(X)
        propensity : np.ndarray, optional, shape (n_samples,)
            Propensity scores P(T=1|X). If None, estimated from data.

        Returns
        -------
        self : GEstimator
            Fitted estimator
        """
        X = np.asarray(X)
        T = np.asarray(T).ravel()
        M = np.asarray(M).ravel()
        Y = np.asarray(Y).ravel()
        weights = np.asarray(weights).ravel()

        n_samples = X.shape[0]

        # Validate inputs
        if not all(arr.shape[0] == n_samples for arr in [T, M, Y, weights]):
            raise ValueError("All inputs must have the same number of samples")

        self.weakly_identified_ = False

        # Store data for bootstrap
        self._X = X
        self._T = T
        self._M = M
        self._Y = Y
        self._weights = weights

        # Store weight diagnostics
        self.weight_stats_ = {
            'mean': float(weights.mean()),
            'std': float(weights.std()),
            'min': float(weights.min()),
            'max': float(weights.max()),
            'median': float(np.median(weights)),
            'n_negative': int(np.sum(weights < 0)),
            'n_zero': int(np.sum(np.abs(weights) < 1e-10))
        }

        # Compute basis functions H = [T, M, ...]
        H = self._basis_functions(T, M, X)

        # Scale X for g(X) estimation
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Estimate or use provided propensity scores
        if propensity is not None:
            self.propensity_ = np.asarray(propensity).ravel()
        else:
            self.propensity_ = self._estimate_propensity(X, T)

        if self.use_centering:
            # Zheng-Zhou G-estimation with centered weights
            theta, g_model, g_pred = self._fit_centered(
                X_scaled, T, M, Y, H, weights
            )
        else:
            # Standard weighted least squares (no centering)
            W = self._process_weights(weights)
            theta, g_model, g_pred = self._fit_wls(
                X_scaled, Y, H, W
            )

        self.theta_ = theta
        self.g_model_ = g_model

        # When weakly identified, theta is NaN — skip variance and return early
        if self.weakly_identified_:
            k = self.n_structural_params
            self.theta_var_ = np.full((k, k), np.nan)
            self.theta_se_ = np.full(k, np.nan)
            return self

        # Compute variance estimates
        if self.variance_method == 'bootstrap':
            self.theta_var_ = self._compute_bootstrap_variance()
        else:
            if self.use_centering:
                A = self._compute_centered_weights(T, weights)
                self.theta_var_ = self._compute_sandwich_variance_centered(
                    H, A, Y, theta, g_pred
                )
            else:
                W = self._process_weights(weights)
                self.theta_var_ = self._compute_sandwich_variance(
                    H, Y, W, theta, g_pred
                )

        self.theta_se_ = np.sqrt(np.diag(self.theta_var_))

        return self

    def _estimate_propensity(
        self,
        X: np.ndarray,
        T: np.ndarray
    ) -> np.ndarray:
        """
        Estimate propensity scores P(T=1|X).

        For RCT data, this is approximately constant.
        For observational data, fit logistic regression.
        """
        from sklearn.linear_model import LogisticRegressionCV

        # Check if treatment is (approximately) randomized
        treatment_prob = T.mean()
        if 0.3 < treatment_prob < 0.7:
            # Could be RCT with ~50% treatment probability
            # Use constant propensity as default
            return np.full(len(T), treatment_prob)

        # Otherwise, estimate from data
        model = LogisticRegressionCV(cv=5, max_iter=1000)
        model.fit(X, T)
        propensity = model.predict_proba(X)[:, 1]

        # Clip extreme values for stability
        propensity = np.clip(propensity, 0.01, 0.99)

        return propensity

    def _compute_centered_weights(
        self,
        T: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute Zheng-Zhou centered weight matrix A.

        A(Z, V) = (Z - e(V)) · W(V)
        where W(V) = (1, τ(V))

        This satisfies E[A|V] = 0, which is crucial for proper G-estimation.

        If normalize_centered_weights=True, weights are standardized before
        constructing A to ensure consistent scale across different Stage 1 methods.
        """
        # Centered treatment: T - e(X)
        T_centered = T - self.propensity_

        # Optionally normalize weights to consistent scale
        if self.normalize_centered_weights:
            # Standardize weights: (w - mean) / std
            w_mean = weights.mean()
            w_std = weights.std()
            if w_std > 1e-10:
                weights_norm = (weights - w_mean) / w_std
            else:
                weights_norm = weights - w_mean
        else:
            weights_norm = weights

        # Weight function W(V) = (1, τ(V))
        # A₁ = T_centered · 1 = T_centered
        # A₂ = T_centered · τ = T_centered * weights
        A = np.column_stack([T_centered, T_centered * weights_norm])

        return A

    def _fit_centered(
        self,
        X_scaled: np.ndarray,
        T: np.ndarray,
        M: np.ndarray,
        Y: np.ndarray,
        H: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Fit using Zheng-Zhou G-estimation with centered weights.

        Solves the estimating equation:
            Σᵢ A(Zᵢ,Vᵢ)' {Yᵢ - θ·Hᵢ - g(Vᵢ)} = 0

        where A(Z,V) = (Z - e(V)) · (1, τ(V)) has E[A|V] = 0.
        """
        # Compute centered weight matrix A (fixed for all iterations)
        A = self._compute_centered_weights(T, weights)

        # Pre-compute AtH once — it is constant across iterations because A and H
        # do not change. Check condition number before iterating: when τ̂ is
        # near-constant, column 2 of A ≈ c·column 1, making AtH rank-deficient
        # and θ₂ unidentified. A condition number above 1e6 signals this regime.
        AtH = A.T @ H
        cond = np.linalg.cond(AtH)
        if cond > 1e6:
            self.weakly_identified_ = True
            self.converged_ = False
            self.n_iterations_ = 0
            self.convergence_history_ = []
            k = self.n_structural_params
            n = len(Y)
            return (
                np.full(k, np.nan),
                None,
                np.full(n, np.nan),
            )

        # Initialize theta
        theta = np.zeros(self.n_structural_params)

        self.convergence_history_ = []
        self.converged_ = False

        # Pre-compute fold splits once so they are identical across iterations
        if self.n_folds_g > 1:
            kf = KFold(n_splits=self.n_folds_g, shuffle=True,
                       random_state=self.seed)
            fold_splits = list(kf.split(X_scaled))

        k = self.n_structural_params
        n = len(Y)

        for iteration in range(self.max_iterations):
            theta_old = theta.copy()

            # Step A: Update g(X) given current theta
            residuals = Y - H @ theta

            if self.n_folds_g > 1:
                # DML partialing-out: fit g on K-1 folds, predict out-of-fold.
                # Breaks correlation between g-model regularisation bias and θ
                # inference, restoring √n-consistent estimation (Chernozhukov 2018).
                g_pred = np.empty_like(Y)
                for train_idx, val_idx in fold_splits:
                    g_fold = self._fit_baseline(X_scaled[train_idx],
                                                residuals[train_idx])
                    g_pred[val_idx] = g_fold.predict(X_scaled[val_idx])
                # Full-data model kept only for predict_g(); not used for θ
                g_model = self._fit_baseline(X_scaled, residuals)
            else:
                g_model = self._fit_baseline(X_scaled, residuals)
                g_pred  = g_model.predict(X_scaled)

            # Step B: Update theta using centered weights A
            # Solve: Σ A'H · θ = Σ A'(Y - g)
            Y_tilde = Y - g_pred
            AtY = A.T @ Y_tilde

            # Use lstsq (robust to mild ill-conditioning) instead of solve
            theta = np.linalg.lstsq(AtH, AtY, rcond=1e-10)[0]

            # Divergence guard: the BCD map can have spectral radius > 1 for
            # certain data configurations (g absorbs H@theta through X→M→Y
            # correlation, feeding back into the next theta update). When
            # ||theta|| exceeds 1e4 the iteration has diverged; flag and
            # return NaN.
            if np.linalg.norm(theta) > 1e4:
                self.weakly_identified_ = True
                self.n_iterations_ = iteration + 1
                self.convergence_history_.append(np.linalg.norm(theta - theta_old))
                return (np.full(k, np.nan), None, np.full(n, np.nan))

            # Check convergence
            delta = np.linalg.norm(theta - theta_old)
            self.convergence_history_.append(delta)

            if delta < self.convergence_tol:
                self.converged_ = True
                break

        self.n_iterations_ = iteration + 1

        # g_pred at final iteration: OOF when n_folds_g>1, in-sample otherwise
        return theta, g_model, g_pred

    def _fit_wls(
        self,
        X_scaled: np.ndarray,
        Y: np.ndarray,
        H: np.ndarray,
        W: np.ndarray
    ) -> Tuple[np.ndarray, object, np.ndarray]:
        """
        Fit using standard weighted least squares.

        Note: This does not have the centering property E[A|V]=0,
        so it may have bias with unmeasured M-Y confounding.
        """
        # Initialize theta
        theta = np.zeros(self.n_structural_params)

        self.convergence_history_ = []
        self.converged_ = False

        for iteration in range(self.max_iterations):
            theta_old = theta.copy()

            # Step A: Update g(X) given current theta
            residuals = Y - H @ theta
            g_model = self._fit_baseline(X_scaled, residuals)
            g_pred = g_model.predict(X_scaled)

            # Step B: Update theta given g(X)
            Y_tilde = Y - g_pred
            theta = self._solve_theta(H, Y_tilde, W)

            # Check convergence
            delta = np.linalg.norm(theta - theta_old)
            self.convergence_history_.append(delta)

            if delta < self.convergence_tol:
                self.converged_ = True
                break

        self.n_iterations_ = iteration + 1

        return theta, g_model, g_pred

    def _basis_functions(
        self,
        T: np.ndarray,
        M: np.ndarray,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute basis functions H = [h₁, h₂, ...].

        Must match the DGP definition exactly:
        - h₁ = T (direct treatment effect)
        - h₂ = M (mediator main effect)
        - Additional: M * X_i for k > 2

        Parameters
        ----------
        T : np.ndarray, shape (n,)
            Treatment
        M : np.ndarray, shape (n,)
            Mediator
        X : np.ndarray, shape (n, p)
            Covariates

        Returns
        -------
        H : np.ndarray, shape (n, k)
            Basis function matrix
        """
        h1 = T                    # Direct treatment effect
        h2 = M                    # Mediator main effect

        basis_list = [h1, h2]

        # Add more basis functions if needed
        if self.n_structural_params > 2:
            n_features = X.shape[1]
            for i in range(self.n_structural_params - 2):
                idx = i % n_features
                basis_list.append(M * X[:, idx])

        H = np.column_stack(basis_list[:self.n_structural_params])
        return H

    def _process_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Process Stage 1 weights for use in weighted estimation.

        Parameters
        ----------
        weights : np.ndarray, shape (n,)
            Raw Stage 1 τ̂_M(X) estimates

        Returns
        -------
        W : np.ndarray, shape (n,)
            Processed weights (positive, clipped, normalized)
        """
        if self.weight_transform == 'squared':
            W = weights ** 2
        elif self.weight_transform == 'absolute':
            W = np.abs(weights)
        else:  # raw
            W = weights.copy()

        # Clip extreme values
        if self.weight_clip_percentile < 100:
            upper_clip = np.percentile(W, self.weight_clip_percentile)
            W = np.clip(W, 1e-10, upper_clip)

        # Ensure minimum weight
        W = np.maximum(W, 1e-10)

        # Normalize to mean 1
        W = W / np.mean(W)

        return W

    def _fit_baseline(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> object:
        """
        Fit baseline model g(X).

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Scaled covariates
        y : np.ndarray, shape (n,)
            Target values (residuals)

        Returns
        -------
        model : fitted sklearn model
        """
        if self.g_model_type == 'ridge':
            model = RidgeCV(
                alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                cv=5
            )
        else:  # neural_network
            model = MLPRegressor(
                hidden_layer_sizes=(64, 32),  # Smaller network for faster convergence
                activation='relu',
                solver='adam',
                alpha=0.01,  # More regularization for stability
                learning_rate='adaptive',  # Adaptive learning rate
                learning_rate_init=0.001,
                max_iter=1000,  # More iterations
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,  # More patience
                tol=1e-4,
                random_state=self.seed,
                warm_start=True  # Use warm start for alternating optimization
            )

        model.fit(X, y)
        return model

    def _solve_theta(
        self,
        H: np.ndarray,
        Y_tilde: np.ndarray,
        W: np.ndarray
    ) -> np.ndarray:
        """
        Solve weighted estimating equation for theta.

        Solves: θ = (H' diag(W) H)^{-1} H' diag(W) Ỹ

        This is weighted least squares with weight matrix diag(W).

        Parameters
        ----------
        H : np.ndarray, shape (n, k)
            Basis function matrix
        Y_tilde : np.ndarray, shape (n,)
            Outcome residuals after removing g(X)
        W : np.ndarray, shape (n,)
            Processed weights

        Returns
        -------
        theta : np.ndarray, shape (k,)
            Estimated structural parameters
        """
        # Weighted design matrix: sqrt(W) * H
        sqrt_W = np.sqrt(W)
        H_weighted = H * sqrt_W[:, np.newaxis]
        Y_weighted = Y_tilde * sqrt_W

        # Solve via normal equations with regularization for stability
        HtWH = H_weighted.T @ H_weighted
        HtWY = H_weighted.T @ Y_weighted

        # Add small regularization for numerical stability
        reg = 1e-8 * np.eye(HtWH.shape[0])
        theta = np.linalg.solve(HtWH + reg, HtWY)

        return theta

    def _compute_sandwich_variance(
        self,
        H: np.ndarray,
        Y: np.ndarray,
        W: np.ndarray,
        theta: np.ndarray,
        g_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute sandwich variance estimator.

        Var(θ̂) = A^{-1} B A^{-T}

        where:
            A = (1/n) H' diag(W) H
            B = (1/n) Σᵢ Wᵢ² ψᵢ² Hᵢ Hᵢ'
            ψᵢ = Yᵢ - g(Xᵢ) - Hᵢ @ θ

        Parameters
        ----------
        H : np.ndarray, shape (n, k)
            Basis function matrix
        Y : np.ndarray, shape (n,)
            Outcome values
        W : np.ndarray, shape (n,)
            Processed weights
        theta : np.ndarray, shape (k,)
            Estimated theta
        g_pred : np.ndarray, shape (n,)
            Predicted g(X)

        Returns
        -------
        var_matrix : np.ndarray, shape (k, k)
            Variance-covariance matrix for theta
        """
        n = H.shape[0]
        k = H.shape[1]

        # Residuals
        psi = Y - g_pred - H @ theta

        # A matrix: (1/n) H' diag(W) H
        A = (H * W[:, np.newaxis]).T @ H / n

        # B matrix: (1/n) Σᵢ Wᵢ² ψᵢ² Hᵢ Hᵢ'
        # This is the "meat" of the sandwich
        W_psi_sq = (W ** 2) * (psi ** 2)
        B = np.zeros((k, k))
        for i in range(n):
            B += W_psi_sq[i] * np.outer(H[i], H[i])
        B = B / n

        # Sandwich: A^{-1} B A^{-T}
        try:
            A_inv = np.linalg.inv(A)
            var_matrix = A_inv @ B @ A_inv.T
        except np.linalg.LinAlgError:
            # Fallback with regularization
            A_reg = A + 1e-8 * np.eye(k)
            A_inv = np.linalg.inv(A_reg)
            var_matrix = A_inv @ B @ A_inv.T

        # Scale by 1/n for finite sample variance
        # A and B are averages (O(1)), so A^{-1} B A^{-T} is O(1)
        # Dividing by n gives Var(θ̂) = O(1/n) which is correct
        var_matrix = var_matrix / n

        return var_matrix

    def _compute_sandwich_variance_centered(
        self,
        H: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        theta: np.ndarray,
        g_pred: np.ndarray
    ) -> np.ndarray:
        """
        Compute sandwich variance for centered G-estimation.

        For the Zheng-Zhou estimating equation:
            Σᵢ Aᵢ' {Yᵢ - θ·Hᵢ - g(Vᵢ)} = 0

        The variance is:
            Var(θ̂) = (A'H)⁻¹ Var(A'ψ) (A'H)⁻ᵀ

        where ψᵢ = Yᵢ - g(Xᵢ) - Hᵢ @ θ

        Parameters
        ----------
        H : np.ndarray, shape (n, k)
            Basis function matrix [T, M]
        A : np.ndarray, shape (n, k)
            Centered weight matrix [(T-e), (T-e)*τ]
        Y : np.ndarray, shape (n,)
            Outcome values
        theta : np.ndarray, shape (k,)
            Estimated theta
        g_pred : np.ndarray, shape (n,)
            Predicted g(X)

        Returns
        -------
        var_matrix : np.ndarray, shape (k, k)
            Variance-covariance matrix for theta
        """
        n = H.shape[0]
        k = H.shape[1]

        # Residuals
        psi = Y - g_pred - H @ theta

        # "Bread": (A'H) / n
        AtH = A.T @ H / n

        # "Meat": (1/n) Σᵢ ψᵢ² Aᵢ Aᵢ'
        psi_sq = psi ** 2
        B = np.zeros((k, k))
        for i in range(n):
            B += psi_sq[i] * np.outer(A[i], A[i])
        B = B / n

        # Sandwich: (A'H)⁻¹ B (A'H)⁻ᵀ
        try:
            AtH_inv = np.linalg.inv(AtH)
            var_matrix = AtH_inv @ B @ AtH_inv.T
        except np.linalg.LinAlgError:
            # Fallback with regularization
            AtH_reg = AtH + 1e-8 * np.eye(k)
            AtH_inv = np.linalg.inv(AtH_reg)
            var_matrix = AtH_inv @ B @ AtH_inv.T

        # Scale by 1/n for finite sample variance
        # AtH and B are averages (O(1)), so AtH^{-1} B AtH^{-T} is O(1)
        # Dividing by n gives Var(θ̂) = O(1/n) which is correct
        var_matrix = var_matrix / n

        return var_matrix

    def _compute_bootstrap_variance(self) -> np.ndarray:
        """
        Compute bootstrap variance for theta estimates.

        More robust than sandwich estimator, especially for
        two-stage estimation with complex Stage 1.

        Returns
        -------
        var_matrix : np.ndarray, shape (k, k)
            Variance-covariance matrix for theta
        """
        if self._X is None:
            raise ValueError("Data not stored. Call fit() first.")

        n = len(self._Y)
        rng = np.random.RandomState(self.seed)

        theta_boots = []

        for b in range(self.n_bootstrap):
            # Bootstrap sample
            idx = rng.choice(n, size=n, replace=True)

            # Fit on bootstrap sample
            try:
                g_boot = GEstimator(
                    g_model_type=self.g_model_type,
                    max_iterations=self.max_iterations,
                    convergence_tol=self.convergence_tol,
                    n_structural_params=self.n_structural_params,
                    use_centering=self.use_centering,
                    weight_transform=self.weight_transform,
                    weight_clip_percentile=self.weight_clip_percentile,
                    variance_method='sandwich',  # Don't recurse
                    seed=self.seed + b
                )
                g_boot.fit(
                    self._X[idx],
                    self._T[idx],
                    self._M[idx],
                    self._Y[idx],
                    self._weights[idx],
                    propensity=self.propensity_[idx] if self.propensity_ is not None else None
                )
                theta_boots.append(g_boot.theta_)
            except Exception:
                # Skip failed bootstrap samples
                continue

        if len(theta_boots) < 50:
            warnings.warn(
                f"Only {len(theta_boots)} bootstrap samples succeeded. "
                "Variance estimate may be unreliable."
            )

        theta_boots = np.array(theta_boots)
        var_matrix = np.cov(theta_boots.T)

        return var_matrix

    def predict_g(self, X: np.ndarray) -> np.ndarray:
        """
        Predict baseline g(X) for new data.

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Covariate matrix

        Returns
        -------
        g_pred : np.ndarray, shape (n,)
            Predicted baseline values
        """
        if self.g_model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self._scaler.transform(X)
        return self.g_model_.predict(X_scaled)

    def get_confidence_intervals(
        self,
        alpha: float = 0.05
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for theta parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level (0.05 for 95% CI)

        Returns
        -------
        ci : dict
            Dictionary with keys 'theta_1', 'theta_2', etc.
            Values are (lower, upper) tuples
        """
        if self.theta_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci = {}

        for i in range(len(self.theta_)):
            lower = self.theta_[i] - z_crit * self.theta_se_[i]
            upper = self.theta_[i] + z_crit * self.theta_se_[i]
            ci[f'theta_{i+1}'] = (lower, upper)

        return ci

    def get_pvalues(self) -> np.ndarray:
        """
        Compute two-sided p-values for H0: θ = 0.

        Returns
        -------
        pvalues : np.ndarray, shape (k,)
            P-values for each theta parameter
        """
        if self.theta_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        z_stats = self.theta_ / self.theta_se_
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
        return pvalues

    def summary(self) -> str:
        """
        Generate summary table of estimation results.

        Returns
        -------
        summary_str : str
            Formatted summary table
        """
        if self.theta_ is None:
            return "Model not fitted. Call fit() first."

        ci = self.get_confidence_intervals()
        pvalues = self.get_pvalues()

        centering_str = "Zheng-Zhou (centered)" if self.use_centering else "WLS (legacy)"
        variance_str = self.variance_method

        lines = [
            "=" * 70,
            "G-Estimation Results: Structural Mean Model",
            "=" * 70,
            f"Model: Y = g(X) + θ₁·T + θ₂·M + ε",
            f"Baseline g(X): {self.g_model_type}",
            f"Estimation: {centering_str}",
            f"Variance: {variance_str}",
            f"Converged: {self.converged_} (iterations: {self.n_iterations_})",
            "-" * 70,
            f"{'Parameter':<12} {'Estimate':>10} {'Std.Err':>10} {'z-value':>10} {'P>|z|':>10} {'95% CI':>20}",
            "-" * 70,
        ]

        param_names = ['θ₁ (T)', 'θ₂ (M)'] + [f'θ_{i+1}' for i in range(2, len(self.theta_))]

        for i, name in enumerate(param_names[:len(self.theta_)]):
            est = self.theta_[i]
            se = self.theta_se_[i]
            z = est / se if se > 0 else np.nan
            p = pvalues[i]
            ci_low, ci_high = ci[f'theta_{i+1}']

            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            ci_str = f"[{ci_low:.3f}, {ci_high:.3f}]"

            lines.append(
                f"{name:<12} {est:>10.4f} {se:>10.4f} {z:>10.3f} {p_str:>10} {ci_str:>20}"
            )

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.theta_ is None:
            return f"GEstimator(g_model_type='{self.g_model_type}', fitted=False)"
        return (
            f"GEstimator(g_model_type='{self.g_model_type}', "
            f"theta={self.theta_.round(3)}, converged={self.converged_})"
        )
