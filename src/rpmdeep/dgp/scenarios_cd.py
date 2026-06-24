#!/usr/bin/env python3
"""
Scenario C / D / E DGP cells — structural extensions of the DGP-A outcome model.

Scenario C — NEH holds, rank preservation fails (random outcome slope):
    Y_i = g(X_i) + θ₁ T_i + (θ₂ + ξ_i) M_i + λ_K K_i + η_i,
    ξ_i ~ N(0, s_ξ²) independent of (X, T, M, K, η). G-estimation stays
    consistent for θ₂ (predicted bias ≈ 0); ξ_i carries ``xi_share`` of Var(ε_Y).

Scenario D — essential heterogeneity, NEH violated (shared latent in both legs):
    L_i ~ N(0,1) independent of (X, T, K, ε_M, η) enters both structural legs:
        M_i = τ(X_i)·(1 + κ L_i)·T_i + μ₀(X_i) + K_i + ε_{M,i}
        Y_i = g(X_i) + θ₁ T_i + θ₂ M_i + γ L_i M_i + λ_K K_i + η_i
    The treatment channel of M is correlated with the outcome error γ L M, so the
    G-moment is violated: asymptotically bias(θ̂₂) → γ·κ, bias(θ̂₁) → 0, for any
    Stage-1 weight.

Scenario E — NEH holds (as in A), but τ is made mostly linear (tunable linearity):
    τ_mix blends an orthogonalised nonlinear and linear component; default weights
    target a linear-dominated CATE surface. Predicted G-estimation bias on θ₂ ≈ 0.
"""

import numpy as np

from .base import CovariateGenerator
from . import mediation as H


# ─────────────────────────────────────────────────────────────────────────────
# Shared base draw — identical to mediation.generate_mediation_data up to calibration
# ─────────────────────────────────────────────────────────────────────────────

def _common_draw(n_samples: int, seed: int) -> dict:
    """Draw X, T, latent scores, K, and ε_M for the shared base.

    RNG streams use offsets (seed, +101, +202, +303) so the covariates /
    treatment / confounder / mediator-noise match Scenario A at a given seed.
    Scenario-specific randomness (ξ in C, L in D) uses higher offsets to stay
    independent of the shared base.
    """
    rng_t   = np.random.RandomState(seed + 101)
    rng_k   = np.random.RandomState(seed + 202)
    rng_eps = np.random.RandomState(seed + 303)

    X = CovariateGenerator(
        n_features=H.N_FEATURES,
        correlation_rho=H._REF_RHO,
        n_gmm_components_range=H._REF_GMM_RANGE,
        seed=seed,
    ).generate(n_samples, seed=seed)

    T          = rng_t.binomial(1, 0.5, size=n_samples).astype(float)
    propensity = np.full(n_samples, 0.5)

    z      = H._standardize(X)
    scores = H._scores(z)

    tau_latent = H._tau_raw(scores)
    tau_latent = tau_latent - tau_latent.mean()
    mu0_latent = H._mu0_raw(scores)
    mu0_latent = mu0_latent - mu0_latent.mean()

    comp  = rng_k.choice(3, size=n_samples, p=H._GMM_WEIGHTS)
    K_gmm = rng_k.normal(H._GMM_MEANS[comp], H._GMM_STDS[comp])
    K_pre = 0.70 * K_gmm + 0.30 * rng_k.randn(n_samples)
    K_raw = K_pre / H._K_PRE_STD

    eps_raw = rng_eps.randn(n_samples)

    return {
        'X': X, 'T': T, 'propensity': propensity,
        'tau_latent': tau_latent, 'mu0_latent': mu0_latent,
        'K_raw': K_raw, 'eps_raw': eps_raw, 'rng_eps': rng_eps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario C — NEH holds, rank preservation fails (random outcome slope)
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_scenario_C(
    n_samples: int,
    seed: int,
    var_shares: dict | None = None,
    lambda_k: float = H.LAMBDA_K,
    sigma0_sq: float = H.SIGMA0_SQ,
    xi_share: float = 0.15,
) -> dict:
    """Scenario C: outcome slope on M is θ₂ + ξ_i, ξ_i ~ N(0, s_ξ²) i.i.d.

    Predicted G-estimation bias on θ₂ ≈ 0. ξ_i carries ``xi_share`` of the fixed
    Var(ε_Y) = sigma0_sq budget.
    """
    if var_shares is None:
        var_shares = H.VAR_SHARES_A.copy()

    b      = _common_draw(n_samples, seed)
    rng_xi = np.random.RandomState(seed + 404)

    comps = {'tau': b['tau_latent'] * b['T'], 'D': b['mu0_latent'],
             'K': b['K_raw'], 'noise': b['eps_raw']}
    scaled, scales, V_total, realized_var = H._calibrate(comps, var_shares)
    M = scaled['tau'] + scaled['D'] + scaled['K'] + scaled['noise']

    var_K_scaled = realized_var['K']

    # Random outcome slope: Var(ξ·M) = s_ξ²·E[M²] = xi_share·sigma0_sq
    EM2  = float(np.mean(M ** 2))
    s_xi = float(np.sqrt(xi_share * sigma0_sq / EM2))
    xi   = rng_xi.normal(0.0, s_xi, n_samples)

    # Hold Var(ε_Y) = sigma0_sq: subtract both the K-channel and the ξM-channel
    sigma_eta_sq = max(1e-8,
                       sigma0_sq - (lambda_k ** 2) * var_K_scaled - xi_share * sigma0_sq)
    eta = b['rng_eps'].randn(n_samples) * np.sqrt(sigma_eta_sq)

    g_X = H._baseline_outcome(b['X'])
    Y   = (g_X + H.THETA_TRUE[0] * b['T']
           + (H.THETA_TRUE[1] + xi) * M
           + lambda_k * scaled['K'] + eta)

    tau_true   = scales['tau'] * b['tau_latent']
    bk_bias_cf = float(lambda_k * var_K_scaled / (var_K_scaled + realized_var['noise']))

    return {
        'X': b['X'], 'T': b['T'], 'M': M, 'Y': Y,
        'propensity': b['propensity'],
        'theta_true': H.THETA_TRUE.copy(),
        'tau_true': tau_true,
        'K': scaled['K'],
        'lambda_k': float(lambda_k),
        'bk_bias_closed_form': bk_bias_cf,
        'sigma_eta_sq': float(sigma_eta_sq),
        'scenario': 'C',
        'xi_sd': s_xi,
        'xi_share': float(xi_share),
        'predicted_gest_bias_theta2': 0.0,
        'var_calibration': {
            'scales': scales, 'V_total': V_total,
            'target_shares': var_shares, 'realized_var': realized_var,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario D — essential heterogeneity, NEH violated (shared latent L)
# ─────────────────────────────────────────────────────────────────────────────

def generate_data_scenario_D(
    n_samples: int,
    seed: int,
    var_shares: dict | None = None,
    lambda_k: float = H.LAMBDA_K,
    sigma0_sq: float = H.SIGMA0_SQ,
    gamma: float = 0.4,
    kappa: float = 0.4,
) -> dict:
    """Scenario D: shared latent L modifies M's treatment response (κ) and the
    outcome's M-slope (γ). Predicted asymptotic bias(θ̂₂) → γ·κ for any weight.

    tau_true = E[M|T=1,X] − E[M|T=0,X] = scale_τ·τ_latent (since E[1+κL] = 1),
    so the Stage-1 target is unchanged; the bias is a Stage-2 identification
    failure, not a Stage-1 estimation error.
    """
    if var_shares is None:
        var_shares = H.VAR_SHARES_A.copy()

    b     = _common_draw(n_samples, seed)
    rng_L = np.random.RandomState(seed + 505)
    L     = rng_L.normal(size=n_samples)

    # Treatment channel carries the random-slope modifier (1 + κL); calibrating
    # the whole component to tau_share leaves bias = γκ invariant (scale cancels
    # in the G-moment ratio), while tau_true stays scale_τ·τ_latent.
    comps = {'tau': b['tau_latent'] * (1.0 + kappa * L) * b['T'],
             'D': b['mu0_latent'], 'K': b['K_raw'], 'noise': b['eps_raw']}
    scaled, scales, V_total, realized_var = H._calibrate(comps, var_shares)
    M = scaled['tau'] + scaled['D'] + scaled['K'] + scaled['noise']

    var_K_scaled = realized_var['K']

    # Outcome error gains γ·L·M; approximate Var(γLM) ≈ γ²·E[M²] for the noise
    # budget (exact constancy of Var(ε_Y) is not essential in the failure cell).
    EM2       = float(np.mean(M ** 2))
    extra_var = (gamma ** 2) * EM2
    sigma_eta_sq = max(1e-8,
                       sigma0_sq - (lambda_k ** 2) * var_K_scaled - extra_var)
    eta = b['rng_eps'].randn(n_samples) * np.sqrt(sigma_eta_sq)

    g_X = H._baseline_outcome(b['X'])
    Y   = (g_X + H.THETA_TRUE[0] * b['T'] + H.THETA_TRUE[1] * M
           + gamma * L * M + lambda_k * scaled['K'] + eta)

    tau_true   = scales['tau'] * b['tau_latent']
    # K-confounding part of the BK bias (the essential-heterogeneity part adds on
    # top empirically; reported separately as predicted_gest_bias_theta2).
    bk_bias_cf = float(lambda_k * var_K_scaled / (var_K_scaled + realized_var['noise']))

    return {
        'X': b['X'], 'T': b['T'], 'M': M, 'Y': Y,
        'propensity': b['propensity'],
        'theta_true': H.THETA_TRUE.copy(),
        'tau_true': tau_true,
        'K': scaled['K'],
        'lambda_k': float(lambda_k),
        'bk_bias_closed_form': bk_bias_cf,
        'sigma_eta_sq': float(sigma_eta_sq),
        'scenario': 'D',
        'gamma': float(gamma),
        'kappa': float(kappa),
        'predicted_gest_bias_theta2': float(gamma * kappa),
        'var_calibration': {
            'scales': scales, 'V_total': V_total,
            'target_shares': var_shares, 'realized_var': realized_var,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scenario E — tunable CATE linearity (orthogonalised mix)
# ─────────────────────────────────────────────────────────────────────────────
#
# CATE linearity is a knob: the nonlinear τ surface is mixed with a purely
# linear direction z_lin = 0.5·s₁ + 0.5·s₂.
#
# The nonlinear part is orthogonalised against z_lin using population (reference)
# projection/SD constants, so the two mixing pieces are uncorrelated (weights are
# relative contributions) and the DGP functional form is identical across n. The
# overall scale is left to `_calibrate`, which rescales τ·T to its τ-share. The
# realised full-linear-R² of τ is reported per cell (not exactly w_lin², since
# the orthogonalised residual may retain other linear-in-X directions).

_LINEAR_MIX_CACHE: dict | None = None


def _linear_mix_refs(n_ref: int = 100_000, seed_ref: int = 987_654) -> dict:
    """Population constants for the orthogonalised linear/nonlinear τ mix.

    Returns β = Cov(z_nl, z_lin)/Var(z_lin) (projection of the nonlinear surface
    onto the linear direction), and the reference SDs of the orthogonalised
    nonlinear residual and of z_lin.  Computed once on a large fixed draw and
    cached, so the mixing ratio — and hence the DGP — does not depend on n.
    """
    global _LINEAR_MIX_CACHE
    if _LINEAR_MIX_CACHE is not None:
        return _LINEAR_MIX_CACHE

    X = CovariateGenerator(
        n_features=H.N_FEATURES,
        correlation_rho=H._REF_RHO,
        n_gmm_components_range=H._REF_GMM_RANGE,
        seed=seed_ref,
    ).generate(n_ref, seed=seed_ref)
    scores = H._scores(H._standardize(X))
    z_nl  = H._tau_raw(scores)
    z_nl  = z_nl - z_nl.mean()
    z_lin = 0.5 * scores['s1'] + 0.5 * scores['s2']
    z_lin = z_lin - z_lin.mean()

    beta    = float(np.cov(z_nl, z_lin, ddof=1)[0, 1] / np.var(z_lin, ddof=1))
    z_perp  = z_nl - beta * z_lin
    _LINEAR_MIX_CACHE = {
        'beta': beta,
        'sd_perp': float(np.std(z_perp, ddof=1)),
        'sd_lin': float(np.std(z_lin, ddof=1)),
    }
    return _LINEAR_MIX_CACHE


def _linear_r2(tau: np.ndarray, X: np.ndarray) -> float:
    """R² of the best linear predictor of τ from the raw features X (with intercept)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, tau, rcond=None)
    resid = tau - A @ coef
    ss_tot = float(np.sum((tau - tau.mean()) ** 2))
    return float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')


def generate_data_scenario_E(
    n_samples: int,
    seed: int,
    var_shares: dict | None = None,
    lambda_k: float = H.LAMBDA_K,
    sigma0_sq: float = H.SIGMA0_SQ,
    w_nonlin: float = 0.30,
    w_lin: float = 0.90,
) -> dict:
    """Scenario E: NEH holds (as in A), but τ is made mostly linear.

    τ_mix = w_nonlin·(z_nl ⟂ z_lin)/sd_perp + w_lin·z_lin/sd_lin, with the
    orthogonalisation and SDs taken from population reference constants. Default
    weights (0.30, 0.90) target a linear-dominated surface; the realised
    full-linear-R² is returned as ``realized_linear_r2``. Predicted G-estimation
    bias on θ₂ ≈ 0.
    """
    if var_shares is None:
        var_shares = H.VAR_SHARES_A.copy()

    b   = _common_draw(n_samples, seed)
    ref = _linear_mix_refs()

    scores = H._scores(H._standardize(b['X']))
    z_nl   = H._tau_raw(scores)
    z_nl   = z_nl - z_nl.mean()
    z_lin  = 0.5 * scores['s1'] + 0.5 * scores['s2']
    z_lin  = z_lin - z_lin.mean()
    z_perp = z_nl - ref['beta'] * z_lin

    tau_mix = (w_nonlin * z_perp / ref['sd_perp']
               + w_lin * z_lin / ref['sd_lin'])
    tau_mix = tau_mix - tau_mix.mean()

    comps = {'tau': tau_mix * b['T'], 'D': b['mu0_latent'],
             'K': b['K_raw'], 'noise': b['eps_raw']}
    scaled, scales, V_total, realized_var = H._calibrate(comps, var_shares)
    M = scaled['tau'] + scaled['D'] + scaled['K'] + scaled['noise']

    var_K_scaled = realized_var['K']
    sigma_eta_sq = max(1e-8, sigma0_sq - (lambda_k ** 2) * var_K_scaled)
    eta = b['rng_eps'].randn(n_samples) * np.sqrt(sigma_eta_sq)

    g_X = H._baseline_outcome(b['X'])
    Y   = (g_X + H.THETA_TRUE[0] * b['T'] + H.THETA_TRUE[1] * M
           + lambda_k * scaled['K'] + eta)

    tau_true   = scales['tau'] * tau_mix
    bk_bias_cf = float(lambda_k * var_K_scaled / (var_K_scaled + realized_var['noise']))

    return {
        'X': b['X'], 'T': b['T'], 'M': M, 'Y': Y,
        'propensity': b['propensity'],
        'theta_true': H.THETA_TRUE.copy(),
        'tau_true': tau_true,
        'K': scaled['K'],
        'lambda_k': float(lambda_k),
        'bk_bias_closed_form': bk_bias_cf,
        'sigma_eta_sq': float(sigma_eta_sq),
        'scenario': 'E',
        'w_nonlin': float(w_nonlin),
        'w_lin': float(w_lin),
        'realized_linear_r2': _linear_r2(tau_true, b['X']),
        'predicted_gest_bias_theta2': 0.0,
        'var_calibration': {
            'scales': scales, 'V_total': V_total,
            'target_shares': var_shares, 'realized_var': realized_var,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Numerical self-check (oracle-weight G-estimation)
# ─────────────────────────────────────────────────────────────────────────────

def verify(n_samples: int = 40000, seed: int = 0) -> dict:
    """Confirm predicted biases with oracle Stage-1 weights (τ̂ = tau_true).

    Scenario C: θ̂₂ ≈ 0.5 (no bias). Scenario D: θ̂₂ ≈ 0.5 + γκ.
    Scenario E: θ̂₂ ≈ 0.5 (NEH holds); also reports realised linear-R²(τ).
    Also includes D-small (γ=κ=0.2 → bias ≈ 0.04).
    """
    from rpmdeep.estimators import GEstimator

    cells = [
        ('C', lambda: generate_data_scenario_C(n_samples, seed)),
        ('D', lambda: generate_data_scenario_D(n_samples, seed)),
        ('Dsmall', lambda: generate_data_scenario_D(n_samples, seed, gamma=0.2, kappa=0.2)),
        ('E', lambda: generate_data_scenario_E(n_samples, seed)),
    ]
    out = {}
    for tag, gen in cells:
        d = gen()
        g = GEstimator(g_model_type='ridge', n_folds_g=5,
                       variance_method='sandwich', seed=seed)
        g.fit(d['X'], d['T'], d['M'], d['Y'],
              weights=d['tau_true'], propensity=d['propensity'])
        rec = {
            'theta2_hat': float(g.theta_[1]),
            'theta2_bias': float(g.theta_[1] - 0.5),
            'predicted_bias': d['predicted_gest_bias_theta2'],
            'theta1_hat': float(g.theta_[0]),
        }
        if 'realized_linear_r2' in d:
            rec['realized_linear_r2'] = d['realized_linear_r2']
        out[tag] = rec
    return out


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2, default=float))
