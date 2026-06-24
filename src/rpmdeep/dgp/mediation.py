#!/usr/bin/env python3
"""
Data generating process with a shared latent-score smooth treatment effect.

τ(X) = 0.45·tanh(s₁·s₂) + 0.35·sin(s₁+s₂)·σ(s₁-s₂) + 0.20·sin(2s₁)·cos(2s₂)

Design properties:
- All three τ terms depend only on the shared scores (s₁, s₂).
- μ₀(X) reuses s₁, s₂ via softplus and quadratic terms.
- K is a GMM-structured latent draw, independent of (X, T) by construction.
- Var(ε_Y) = SIGMA0_SQ is held constant across λ_K: σ²_η(λ_K) is adjusted so
  the λ_K sweep changes confounding strength only, not outcome noise.

BK bias closed form (K ⊥ X):
  Bias_BK = λ_K · Var(K_scaled) / (Var(K_scaled) + Var(ε_M))
           = λ_K · s_K / (s_K + s_ε)
"""

import numpy as np

from .base import CovariateGenerator

# ─────────────────────────────────────────────────────────────────────────────
# DGP constants
# ─────────────────────────────────────────────────────────────────────────────

N_FEATURES = 25

# Feature partition (all τ variation enters via the shared scores)
SHARED1_IDX = list(range(0, 5))    # 5 features → s₁
SHARED2_IDX = list(range(5, 10))   # 5 features → s₂
D_IDX       = list(range(15, 18))  # 3 features → rd (μ₀-specific)
# Indices 10–14, 18–24: pure noise (not used in τ or μ₀)

# Unit-norm latent-score weights
def _unit(w: np.ndarray) -> np.ndarray:
    return w / np.linalg.norm(w)

COEF_S1 = _unit(np.array([ 1.00, -0.80,  0.70,  0.90, -0.70]))
COEF_S2 = _unit(np.array([ 0.90,  1.00, -0.90,  0.75, -0.80]))
COEF_RD = _unit(np.array([ 0.90, -0.70,  0.60]))

# Variance shares — Scenario A (τ-share=25%)
VAR_SHARES_A = {'tau': 0.25, 'D': 0.33, 'K': 0.20, 'noise': 0.22}
# Scenario B (τ-share=40%)
VAR_SHARES_B = {'tau': 0.40, 'D': 0.28, 'K': 0.18, 'noise': 0.14}

VAR_SHARES = VAR_SHARES_A  # default

# Outcome noise: Var(ε_Y) = SIGMA0_SQ constant across λ_K
SIGMA0_SQ = 2.5

# K: 3-component GMM (mean-zero by design) + Gaussian noise
_GMM_MEANS   = np.array([-1.2,  0.3,  1.5])
_GMM_STDS    = np.array([ 0.8,  0.6,  0.9])
_GMM_WEIGHTS = np.array([ 0.40, 0.35, 0.25])

# Analytic SD(K_pre) where K_pre = 0.70·K_gmm + 0.30·N(0,1)
# E[K_gmm] = 0 by construction; Var(K_gmm) ≈ 1.7545
_K_GMM_VAR = float(
    np.sum(_GMM_WEIGHTS * (_GMM_STDS ** 2 + _GMM_MEANS ** 2))
    - np.dot(_GMM_WEIGHTS, _GMM_MEANS) ** 2
)
_K_PRE_STD = float(np.sqrt(0.70 ** 2 * _K_GMM_VAR + 0.30 ** 2 * 1.0))

LAMBDA_K   = 1.0
THETA_TRUE = np.array([1.0, 0.5])


# ─────────────────────────────────────────────────────────────────────────────
# Reference standardisation (fixed reference sample, seed=0)
# ─────────────────────────────────────────────────────────────────────────────

_REF_N              = 20000
_REF_SEED           = 0
_REF_RHO            = 0.5
_REF_GMM_RANGE      = (5, 10)


def _compute_reference_scaling() -> tuple:
    X_ref = CovariateGenerator(
        n_features=N_FEATURES,
        correlation_rho=_REF_RHO,
        n_gmm_components_range=_REF_GMM_RANGE,
        seed=_REF_SEED,
    ).generate(_REF_N, seed=_REF_SEED)
    means = X_ref.mean(axis=0)
    stds  = X_ref.std(axis=0, ddof=1)
    stds  = np.where(stds < 1e-8, 1.0, stds)
    return means.astype(float), stds.astype(float)


_REF_MEANS, _REF_STDS = _compute_reference_scaling()


def _standardize(X: np.ndarray) -> np.ndarray:
    return (X - _REF_MEANS) / _REF_STDS


# ─────────────────────────────────────────────────────────────────────────────
# Latent scores and structural functions
# ─────────────────────────────────────────────────────────────────────────────

def _scores(z: np.ndarray) -> dict:
    s1 = z[:, SHARED1_IDX] @ COEF_S1
    s2 = z[:, SHARED2_IDX] @ COEF_S2
    rd = z[:, D_IDX]       @ COEF_RD
    return {'s1': s1, 's2': s2, 'rd': rd}


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def _tau_raw(scores: dict) -> np.ndarray:
    """
    τ(X) = 0.45·tanh(s₁s₂) + 0.35·sin(s₁+s₂)·σ(s₁-s₂) + 0.20·sin(2s₁)·cos(2s₂)

    All terms use the shared scores (s₁, s₂) only.
    """
    s1, s2 = scores['s1'], scores['s2']
    return (
        0.45 * np.tanh(s1 * s2)
        + 0.35 * np.sin(s1 + s2) * _expit(s1 - s2)
        + 0.20 * np.sin(2.0 * s1) * np.cos(2.0 * s2)
    )


def _mu0_raw(scores: dict) -> np.ndarray:
    """μ₀(X) = 0.50·softplus(0.8s₁-0.4s₂) + 0.30·tanh(rd) + 0.20·(s₂²-1)"""
    s1, s2, rd = scores['s1'], scores['s2'], scores['rd']
    return (
        0.50 * _softplus(0.8 * s1 - 0.4 * s2)
        + 0.30 * np.tanh(rd)
        + 0.20 * (s2 ** 2 - 1.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variance calibration
# ─────────────────────────────────────────────────────────────────────────────

def _calibrate(components: dict, var_shares: dict) -> tuple:
    """Scale components to target variance shares; return (scaled, scales, V_total, realized_var)."""
    s = {k: max(0.0, float(v)) for k, v in var_shares.items()}
    S = sum(s.values())
    s = {k: v / S for k, v in s.items()}

    V_total = 1.0 / s['noise'] if s['noise'] > 1e-12 else 10.0
    scaled, scales, realized_var = {}, {}, {}
    for name, arr in components.items():
        v = float(np.var(arr, ddof=1)) + 1e-12
        a = float(np.sqrt(s[name] * V_total / v))
        scaled[name]      = a * arr
        scales[name]      = a
        realized_var[name] = float(np.var(a * arr, ddof=1))
    return scaled, scales, float(V_total), realized_var


def _baseline_outcome(X: np.ndarray) -> np.ndarray:
    n_use = min(7, X.shape[1])
    g = np.zeros(X.shape[0])
    for i in range(n_use):
        g += 0.4 * X[:, i]
        if i % 2 == 0:
            g += 0.2 * X[:, i] ** 2
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Top-level data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_mediation_data(
    n_samples: int,
    seed: int,
    var_shares: dict = None,
    lambda_k: float = LAMBDA_K,
    sigma0_sq: float = SIGMA0_SQ,
) -> dict:
    """
    Generate a mediation dataset for scenario A.

    Parameters
    ----------
    n_samples : int
    seed : int
    var_shares : dict, optional
        Variance shares for {'tau', 'D', 'K', 'noise'}.
        Defaults to VAR_SHARES_A (τ-share=25%, Scenario A).
        Pass VAR_SHARES_B for Scenario B (τ-share=40%).
    lambda_k : float
        Confounding strength. Var(ε_Y) is held constant at sigma0_sq
        by adjusting the idiosyncratic noise variance.
    sigma0_sq : float
        Target Var(ε_Y) = constant across λ_K.

    Returns
    -------
    dict with: X, T, M, Y, propensity, theta_true, tau_true, K,
    lambda_k, bk_bias_closed_form, sigma_eta_sq, var_calibration.
    """
    if var_shares is None:
        var_shares = VAR_SHARES.copy()

    # Separate RNG streams for X, T, K, and noise
    rng_x   = np.random.RandomState(seed)           # for CovariateGenerator (passed as seed)
    rng_t   = np.random.RandomState(seed + 101)     # for T
    rng_k   = np.random.RandomState(seed + 202)     # for K (independent of X,T)
    rng_eps = np.random.RandomState(seed + 303)     # for ε_M and η

    # ── Covariates ──
    X = CovariateGenerator(
        n_features=N_FEATURES,
        correlation_rho=_REF_RHO,
        n_gmm_components_range=_REF_GMM_RANGE,
        seed=seed,
    ).generate(n_samples, seed=seed)

    # ── Treatment (RCT, e(X) = 0.5 known) ──
    T          = rng_t.binomial(1, 0.5, size=n_samples).astype(float)
    propensity = np.full(n_samples, 0.5)

    # ── Latent scores ──
    z      = _standardize(X)
    scores = _scores(z)

    # ── τ(X) and μ₀(X) ──
    tau_latent = _tau_raw(scores)
    tau_latent = tau_latent - tau_latent.mean()
    mu0_latent = _mu0_raw(scores)
    mu0_latent = mu0_latent - mu0_latent.mean()

    # ── K: GMM-structured confounder, independent of (X, T) ──
    # K_pre = 0.70·K_gmm + 0.30·N(0,1); analytic standardisation to unit SD
    comp      = rng_k.choice(3, size=n_samples, p=_GMM_WEIGHTS)
    K_gmm     = rng_k.normal(_GMM_MEANS[comp], _GMM_STDS[comp])
    K_pre     = 0.70 * K_gmm + 0.30 * rng_k.randn(n_samples)
    K_raw     = K_pre / _K_PRE_STD   # analytic SD, not sample SD

    # ── Mediator noise ε_M ──
    eps_raw = rng_eps.randn(n_samples)

    # ── Variance calibration ──
    comps = {'tau': tau_latent * T, 'D': mu0_latent, 'K': K_raw, 'noise': eps_raw}
    scaled, scales, V_total, realized_var = _calibrate(comps, var_shares)

    M = scaled['tau'] + scaled['D'] + scaled['K'] + scaled['noise']

    # ── Outcome with constant Var(ε_Y) across λ_K ──
    var_K_scaled  = realized_var['K']
    sigma_eta_sq  = max(1e-8, sigma0_sq - (lambda_k ** 2) * var_K_scaled)
    eta           = rng_eps.randn(n_samples) * np.sqrt(sigma_eta_sq)
    eps_Y         = lambda_k * scaled['K'] + eta

    g_X = _baseline_outcome(X)
    Y   = g_X + THETA_TRUE[0] * T + THETA_TRUE[1] * M + eps_Y

    # ── True τ_M(X) = E[M|T=1,X] - E[M|T=0,X] ──
    tau_true = scales['tau'] * tau_latent

    # ── BK bias closed form (exact when K ⊥ X) ──
    bk_bias_cf = float(
        lambda_k * var_K_scaled
        / (var_K_scaled + realized_var['noise'])
    )

    return {
        'X':                   X,
        'T':                   T,
        'M':                   M,
        'Y':                   Y,
        'propensity':          propensity,
        'theta_true':          THETA_TRUE.copy(),
        'tau_true':            tau_true,
        'K':                   scaled['K'],
        'lambda_k':            float(lambda_k),
        'bk_bias_closed_form': bk_bias_cf,
        'sigma_eta_sq':        float(sigma_eta_sq),
        'var_calibration': {
            'scales':          scales,
            'V_total':         V_total,
            'target_shares':   var_shares,
            'realized_var':    realized_var,
        },
    }
