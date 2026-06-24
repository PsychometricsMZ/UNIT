"""
Shared Stage-1 components for the scenario pool runners (``rpmdeep.studies``).

This module holds the pieces every pool reuses so the four-learner ladder is
configured identically across scenarios:

  - Stage-1 hyperparameters per learner, scaled to sample size
    (``_tarnet_params``, ``_ridge_params``, ``_rf_params``)
  - K-fold cross-fitting of the Stage-1 CATE (``cross_fit_tau``)
  - Stage-1 and Stage-2 metric helpers (``_tau_metrics``, ``_theta_metrics``)

Each pool generates its own data and drives Stage 2; they import these helpers
so the learner configuration and metric definitions are defined in one place.
"""

import warnings
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from scipy import stats

# Repo root (parents: [0]=rpmdeep, [1]=src, [2]=repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

ALPHA      = 0.05   # significance level → 95% CI
N_FOLDS_CF = 5      # cross-fitting folds for Stage 1


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

def _tarnet_params(n_samples: int) -> dict:
    """
    TARNet (SNet1) hyperparameters scaled to sample size.

    Notes:
    - No seed in params: cross_fit_tau uses the experiment seed as base, so
      each replication gets different fold initializations.
    - n_iter_min: CATENets early stopping fires when (epoch+1)*n_batches > n_iter_min.
      Set n_iter_min ≈ 20 × n_batches to guarantee ≥20 full epochs before any
      early stopping check, where n_batches ≈ 0.8*0.9*n_samples / batch_size
      (cross-fit fold × val_split_prop removed).
    - batch_size scaled with n: larger batches at large n reduce gradient noise,
      keeping early-stopping validation loss smooth.
    - penalty_l2 reduced at large n (data alone regularizes; l2 only hurts capacity).
    """
    def _n_iter_min(n, bs, min_epochs=20):
        n_train = int(n * 0.8 * 0.9)   # cross-fit fold × val_split kept out
        n_batches = max(1, int(n_train / bs))
        return min_epochs * n_batches

    if n_samples <= 500:
        bs = 32
        return dict(n_layers_r=2, n_units_r=100, n_layers_out=1, n_units_out=50,
                    penalty_l2=0.02, n_iter=500, batch_size=bs,
                    patience=20, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)
    elif n_samples <= 2000:
        bs = 64
        return dict(n_layers_r=3, n_units_r=200, n_layers_out=2, n_units_out=100,
                    penalty_l2=0.02, n_iter=700, batch_size=bs,
                    patience=25, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)
    elif n_samples <= 5000:
        bs = 128
        return dict(n_layers_r=3, n_units_r=200, n_layers_out=2, n_units_out=100,
                    penalty_l2=0.01, n_iter=800, batch_size=bs,
                    patience=30, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)
    elif n_samples <= 10000:
        bs = 256
        return dict(n_layers_r=3, n_units_r=200, n_layers_out=2, n_units_out=100,
                    penalty_l2=0.01, n_iter=1000, batch_size=bs,
                    patience=40, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)
    elif n_samples <= 50000:
        bs = 512
        return dict(n_layers_r=4, n_units_r=300, n_layers_out=3, n_units_out=150,
                    penalty_l2=0.005, n_iter=800, batch_size=bs,
                    patience=50, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)
    else:  # n > 50000
        bs = 1024
        return dict(n_layers_r=4, n_units_r=300, n_layers_out=3, n_units_out=150,
                    penalty_l2=0.001, n_iter=600, batch_size=bs,
                    patience=60, n_iter_min=_n_iter_min(n_samples, bs),
                    val_split_prop=0.1)


def _ridge_params(n_samples: int) -> dict:
    return dict(seed=0)


def _rf_params(n_samples: int) -> dict:
    """
    Random Forest hyperparameters for the restricted-depth baseline arm.
    n_jobs=1 required to avoid thread contention in ProcessPoolExecutor workers.
    """
    if n_samples <= 500:
        min_samples_leaf = 5
    elif n_samples <= 1000:
        min_samples_leaf = 8
    elif n_samples <= 5000:
        min_samples_leaf = 15
    else:
        min_samples_leaf = 30
    return dict(
        n_estimators=200,
        max_depth=3,             # depth-3: at most 3-level interactions
        min_samples_leaf=min_samples_leaf,
        max_features=0.35,       # ~35 % of features per split
        n_jobs=1,
        seed=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: cross-fitted τ̂(X)
# ──────────────────────────────────────────────────────────────────────────────

def cross_fit_tau(
    X: np.ndarray,
    T: np.ndarray,
    M: np.ndarray,
    stage1_class,
    stage1_params: dict,
    n_folds: int = N_FOLDS_CF,
    seed: int = 42,
) -> np.ndarray:
    """
    K-fold cross-fitting of Stage 1: estimate τ̂(X) out-of-fold.

    For each fold k: train stage1_class on K-1 folds,
    predict τ̂ on the held-out fold k. This ensures τ̂(X_i)
    is from a model that never observed unit i during training
    (DML / Chernozhukov et al. 2018 cross-fitting principle).
    """
    n       = len(T)
    tau_hat = np.zeros(n)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        params = stage1_params.copy()
        params['seed'] = params.get('seed', seed) + fold_idx  # unique seed per fold
        model  = stage1_class(**params)
        model.fit(X[train_idx], T[train_idx], M[train_idx])
        tau_hat[val_idx] = model.predict_cate(X[val_idx])

    return tau_hat


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def _tau_metrics(tau_hat: np.ndarray, tau_true: np.ndarray) -> dict:
    """Stage 1 quality: RMSE, NRMSE (÷ std), Pearson correlation."""
    rmse     = float(np.sqrt(np.mean((tau_hat - tau_true) ** 2)))
    sd_true  = float(np.std(tau_true))
    nrmse    = rmse / sd_true if sd_true > 1e-10 else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr = float(np.corrcoef(tau_hat, tau_true)[0, 1])
    return {'tau_rmse': rmse, 'tau_nrmse': nrmse, 'tau_corr': corr}


def _theta_metrics(
    theta_hat: np.ndarray,
    theta_se: np.ndarray,
    theta_true: np.ndarray,
) -> dict:
    """
    Stage 2 quality per parameter: estimate, SE, bias, 95% CI coverage.

    Coverage = 1 iff theta_true ∈ [θ̂ ± z_{α/2} · SE].
    """
    z_crit = stats.norm.ppf(1 - ALPHA / 2)   # 1.96 for α=0.05
    out    = {}
    for k in range(len(theta_true)):
        lo      = theta_hat[k] - z_crit * theta_se[k]
        hi      = theta_hat[k] + z_crit * theta_se[k]
        out[f'theta{k+1}_hat']     = float(theta_hat[k])
        out[f'theta{k+1}_se']      = float(theta_se[k])
        out[f'theta{k+1}_bias']    = float(theta_hat[k] - theta_true[k])
        out[f'theta{k+1}_covered'] = float(lo <= theta_true[k] <= hi)
    return out
