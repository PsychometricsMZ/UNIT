#!/usr/bin/env python3
"""
Scenario Dsmall / Wtau pool.

Scenario Dsmall: essential heterogeneity at half amplitude (γ=κ=0.2), NEH
                 violated. Predicted bias(θ̂₂) → γκ = 0.04.
Scenario Wtau  : Scenario-A structure with the τ variance share dropped to 8%.
                 Predicted G-est bias ≈ 0.

Seed-major, scenario-rotated grid; one JSON per experiment in POOL_DIR.
Output: results/pool_scenario_ext/
"""

import os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import json, logging, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy import stats

from rpmdeep.pipeline import REPO_ROOT

SAMPLE_SIZES    = [500, 1000, 2000, 5000, 10000]
SCENARIOS       = ['Dsmall', 'Wtau']
N_SEEDS         = 1000   # seed-major rotation keeps every (scenario × n) cell
#                          balanced at whatever seed count is reached
N_WORKERS       = 10
N_FOLDS_CF      = 5
CHECKPOINT_N    = 10
BUFFER          = 2
RF_TUNED_N_ITER = 6
RF_TUNED_CV     = 3
LAMBDA_K        = 1.0

# Scenario-specific structural parameters
DSMALL_GAMMA    = 0.2    # Scenario Dsmall: half-amplitude essential heterogeneity
DSMALL_KAPPA    = 0.2    # → predicted bias(θ̂₂) = γκ = 0.04
VAR_SHARES_WEAK = {'tau': 0.08, 'D': 0.50, 'K': 0.20, 'noise': 0.22}  # weak instrument

POOL_DIR = REPO_ROOT / 'results' / 'pool_scenario_ext'


def run_experiment(scenario: str, n_samples: int, seed: int) -> dict:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    warnings.filterwarnings('ignore')

    from rpmdeep.dgp.scenarios_cd import generate_data_scenario_D
    from rpmdeep.dgp.mediation import generate_mediation_data
    from rpmdeep.pipeline import (
        cross_fit_tau, _tarnet_params, _ridge_params,
        _tau_metrics, _theta_metrics, ALPHA,
    )
    from rpmdeep.estimators import (
        TARNetMediator, RidgeMediator, GEstimator, BaronKennyEstimator,
    )
    from rpmdeep.estimators.random_forest_tuned_mediator import RandomForestTunedMediator

    t0     = time.time()
    z_crit = stats.norm.ppf(1 - ALPHA / 2)

    record = {
        'n_samples':   n_samples,
        'seed':        seed,
        'scenario':    scenario,
        'lambda_k':    LAMBDA_K,
        'dgp':         f'mediation_{scenario}',
        'theta1_true': 1.0,
        'theta2_true': 0.5,
    }

    try:
        if scenario == 'Dsmall':
            data = generate_data_scenario_D(n_samples, seed, lambda_k=LAMBDA_K,
                                            gamma=DSMALL_GAMMA, kappa=DSMALL_KAPPA)
        elif scenario == 'Wtau':
            data = generate_mediation_data(n_samples, seed, var_shares=VAR_SHARES_WEAK,
                                        lambda_k=LAMBDA_K)
        else:
            raise ValueError(f'unknown scenario {scenario!r}')

        X, T, M, Y = data['X'], data['T'], data['M'], data['Y']
        tau_true   = data['tau_true']
        theta_true = data['theta_true']
        propensity = data['propensity']
        record['tau_sd_true']  = float(np.std(tau_true))
        record['bk_bias_cf']   = data.get('bk_bias_closed_form', float('nan'))
        record['sigma_eta_sq'] = data.get('sigma_eta_sq', float('nan'))
        record['predicted_gest_bias_theta2'] = data.get('predicted_gest_bias_theta2', 0.0)
        if 'realized_linear_r2' in data:
            record['realized_linear_r2'] = data['realized_linear_r2']

        stage1_cfgs = {
            'ridge':    (RidgeMediator,            _ridge_params(n_samples)),
            'rf_tuned': (RandomForestTunedMediator,
                         dict(n_iter=RF_TUNED_N_ITER, cv=RF_TUNED_CV, seed=0)),
            'tarnet':   (TARNetMediator,            _tarnet_params(n_samples)),
        }

        for method, (cls, params) in stage1_cfgs.items():
            t_m = time.time()
            try:
                tau_hat = cross_fit_tau(X, T, M, cls, params,
                                        n_folds=N_FOLDS_CF, seed=seed)
                s1 = _tau_metrics(tau_hat, tau_true)

                if s1['tau_nrmse'] > 1.3 or float(np.std(tau_hat)) < 1e-4:
                    for k, v in s1.items():
                        record[f'{method}_{k}'] = v
                    record[f'{method}_stage2_skipped'] = True
                    record[f'{method}_time'] = time.time() - t_m
                    continue

                g_est = GEstimator(g_model_type='ridge', n_folds_g=5,
                                   variance_method='sandwich', seed=seed)
                g_est.fit(X, T, M, Y, weights=tau_hat, propensity=propensity)

                if g_est.weakly_identified_:
                    for k, v in s1.items():
                        record[f'{method}_{k}'] = v
                    record[f'{method}_weakly_identified'] = True
                    record[f'{method}_stage2_skipped']   = True
                    record[f'{method}_time'] = time.time() - t_m
                    continue

                s2 = _theta_metrics(g_est.theta_, g_est.theta_se_, theta_true)
                s2['converged']    = bool(g_est.converged_)
                s2['n_iterations'] = int(g_est.n_iterations_)

                hat = s2['theta2_hat']; se = s2['theta2_se']
                lo, hi = hat - z_crit * se, hat + z_crit * se
                s2['theta2_reject_null'] = float(not (lo <= 0.0 <= hi))

                for k, v in {**s1, **s2}.items():
                    record[f'{method}_{k}'] = v
                record[f'{method}_time'] = time.time() - t_m

            except Exception as exc:
                record[f'{method}_error'] = str(exc)
                record[f'{method}_time']  = time.time() - t_m

        try:
            bk = BaronKennyEstimator()
            bk.fit(X, T, M, Y)
            s2_bk = _theta_metrics(bk.theta_, bk.theta_se_, theta_true)
            hat = s2_bk['theta2_hat']; se = s2_bk['theta2_se']
            lo, hi = hat - z_crit * se, hat + z_crit * se
            s2_bk['theta2_reject_null'] = float(not (lo <= 0.0 <= hi))
            for k, v in s2_bk.items():
                record[f'baron_kenny_{k}'] = v
        except Exception as exc:
            record['baron_kenny_error'] = str(exc)

    except Exception as exc:
        record['global_error'] = str(exc)

    record['total_time'] = time.time() - t0
    return record


def _experiment_path(pool_dir: Path, scenario: str, n: int, seed: int) -> Path:
    return pool_dir / f'ext_{scenario}_n{n:06d}_s{seed:05d}.json'


def _experiment_grid() -> list:
    """Seed-major, scenario-rotated grid → balanced partial coverage."""
    grid = []
    for sd in range(N_SEEDS):
        for scenario in SCENARIOS:
            for n in SAMPLE_SIZES:
                grid.append((scenario, n, sd))
    return grid


_SCENARIO_HDR = {
    'Dsmall': f'### Scenario Dsmall — essential heterogeneity γ=κ=0.2 '
              f'(predicted G-est bias = γκ = {DSMALL_GAMMA*DSMALL_KAPPA:+.3f})',
    'Wtau':   '### Scenario Wtau — weak instrument, τ-share 8% '
              '(predicted G-est bias ≈ 0)',
}


def _print_summary(df: pd.DataFrame) -> None:
    methods = ['ridge', 'rf_tuned', 'tarnet']
    print('\n' + '=' * 96)
    print(f'Scenario Dsmall / Wtau pool  ({len(df)} experiments)')
    print('=' * 96)
    for scenario in SCENARIOS:
        ds = df[df['scenario'] == scenario] if 'scenario' in df.columns else df.iloc[0:0]
        if len(ds) == 0:
            continue
        print('\n' + _SCENARIO_HDR.get(scenario, f'### Scenario {scenario}'))
        for n in SAMPLE_SIZES:
            sub = ds[ds['n_samples'] == n]
            if len(sub) == 0:
                continue
            print(f'\n— n={n}  ({len(sub)} seeds)')
            print(f'  {"Method":<12} {"NRMSE":>7} {"θ2 bias":>9} '
                  f'{"θ2 SE":>8} {"cover":>7} {"rej%":>6}')
            tarnet_se = (sub['tarnet_theta2_se'].mean()
                         if 'tarnet_theta2_se' in sub.columns else float('nan'))
            for m in methods:
                bias_c = f'{m}_theta2_bias'
                if bias_c not in sub.columns:
                    continue
                nrmse = sub[f'{m}_tau_nrmse'].mean() if f'{m}_tau_nrmse' in sub.columns else float('nan')
                bias  = sub[bias_c].mean()
                se    = sub[f'{m}_theta2_se'].mean() if f'{m}_theta2_se' in sub.columns else float('nan')
                cov   = sub[f'{m}_theta2_covered'].mean() if f'{m}_theta2_covered' in sub.columns else float('nan')
                rej   = sub[f'{m}_theta2_reject_null'].mean() if f'{m}_theta2_reject_null' in sub.columns else float('nan')
                ratio = se / tarnet_se if tarnet_se and np.isfinite(tarnet_se) and tarnet_se > 0 else float('nan')
                print(f'  {m:<12} {nrmse:>7.3f} {bias:>+9.3f} '
                      f'{se:>8.4f} {cov:>7.3f} {rej:>6.2f}   SE/TARNet={ratio:.2f}')
            if 'baron_kenny_theta2_bias' in sub.columns:
                bk_bias = sub['baron_kenny_theta2_bias'].mean()
                bk_cov  = sub['baron_kenny_theta2_covered'].mean() if 'baron_kenny_theta2_covered' in sub.columns else float('nan')
                print(f'  {"baron_kenny":<12} {"—":>7} {bk_bias:>+9.3f} '
                      f'{"—":>8} {bk_cov:>7.3f}')
    print('\n' + '=' * 96)


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS']      = '1'
    os.environ['MKL_NUM_THREADS']      = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def _save_experiment(rec: dict, pool_dir: Path) -> None:
    path = _experiment_path(pool_dir, rec['scenario'], rec['n_samples'], rec['seed'])
    path.write_text(json.dumps(rec, default=float))


def _load_existing(pool_dir: Path) -> list:
    records = []
    for p in sorted(pool_dir.glob('ext_*_n*_s*.json')):
        try:
            records.append(json.loads(p.read_text()))
        except Exception:
            pass
    return records


def main():
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    log_file = POOL_DIR / f'run_{ts}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    full_grid = _experiment_grid()
    missing   = [(sc, n, s) for (sc, n, s) in full_grid
                 if not _experiment_path(POOL_DIR, sc, n, s).exists()]
    done_n    = len(full_grid) - len(missing)

    logging.info(f'[Scenario Dsmall/Wtau pool] {POOL_DIR}')
    logging.info(f'Grid: {len(full_grid)} '
                 f'({len(SCENARIOS)} scenarios × {len(SAMPLE_SIZES)} n × {N_SEEDS} seeds) '
                 f'— done: {done_n}, to run: {len(missing)}')

    if not missing:
        logging.info('Nothing to do.')
        df = pd.DataFrame(_load_existing(POOL_DIR))
        df.to_csv(POOL_DIR / 'pool_checkpoint.csv', index=False)
        _print_summary(df)
        return

    results  = _load_existing(POOL_DIR)
    n_done   = 0
    t_wall   = time.time()
    csv_path = POOL_DIR / 'pool_checkpoint.csv'

    missing_iter = iter(missing)
    in_flight    = N_WORKERS + BUFFER
    pending      = {}

    def _submit_next(executor):
        try:
            sc, n, s = next(missing_iter)
        except StopIteration:
            return False
        fut = executor.submit(run_experiment, sc, n, s)
        pending[fut] = (sc, n, s)
        return True

    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS,
                                 initializer=_worker_init) as executor:
            for _ in range(in_flight):
                if not _submit_next(executor):
                    break

            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    sc, n, s = pending.pop(fut)
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        logging.error(f'FAILED {sc} n={n} seed={s}: {exc}')
                        rec = {'scenario': sc, 'n_samples': n, 'seed': s,
                               'global_error': str(exc)}
                    _save_experiment(rec, POOL_DIR)
                    results.append(rec)
                    n_done += 1
                    _submit_next(executor)

                    elapsed = time.time() - t_wall
                    rate    = n_done / elapsed if elapsed > 0 else 1e-9
                    eta     = (len(missing) - n_done) / rate
                    if n_done % 5 == 0 or n_done == len(missing):
                        counts = {sc_: sum(1 for r in results if r.get('scenario') == sc_)
                                  for sc_ in SCENARIOS}
                        per_sc = '  '.join(f'{k}:{v}' for k, v in counts.items())
                        logging.info(
                            f'[{n_done:4d}/{len(missing)}]  '
                            f'elapsed={elapsed/60:5.1f}m  ETA={eta/60:6.1f}m  | {per_sc}'
                        )
                    if n_done % CHECKPOINT_N == 0 or n_done == len(missing):
                        pd.DataFrame(results).to_csv(csv_path, index=False)

    except KeyboardInterrupt:
        logging.warning('Interrupted. Saving…')
        pd.DataFrame(results).to_csv(csv_path, index=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logging.info(f'Saved {len(df)} rows → {csv_path}')
    _print_summary(df)
    logging.info(f'Total wall time: {(time.time()-t_wall)/60:.1f} min')


if __name__ == '__main__':
    main()
