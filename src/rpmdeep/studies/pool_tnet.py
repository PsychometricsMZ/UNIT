#!/usr/bin/env python3
"""
TNet pool. Neural T-learner (CATENets TNet, separate nets per arm) as a Stage-1
mediator-CATE learner. Uses the same per-arm architecture as TARNet
(pipeline._tarnet_params) with train_separate=True. Data are generated
identically to the main pool at a given seed. Seed-major, scenario-rotated grid;
one JSON per experiment in results/pool_tnet/.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import json, logging, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from scipy import stats

from rpmdeep.pipeline import REPO_ROOT

SAMPLE_SIZES = [500, 1000, 2000, 5000, 10000]
SCENARIOS    = ['A', 'B', 'C', 'D', 'E', 'Dsmall', 'Wtau']
N_SEEDS      = 1000          # seed-major rotation keeps cells balanced
N_WORKERS    = 18
N_FOLDS_CF   = 5
CHECKPOINT_N = 10
BUFFER       = 2
LAMBDA_K     = 1.0

# scenario-specific structural params (identical to the main pool)
XI_SHARE            = 0.15
GAMMA, KAPPA        = 0.4, 0.4
DSMALL_G, DSMALL_K  = 0.2, 0.2
E_W_NONLIN, E_W_LIN = 0.30, 0.90
VAR_SHARES_WEAK     = {'tau': 0.08, 'D': 0.50, 'K': 0.20, 'noise': 0.22}

POOL_DIR = REPO_ROOT / 'results' / 'pool_tnet'


def _generate(scenario, n, seed):
    from rpmdeep.dgp.mediation import generate_mediation_data, VAR_SHARES_A, VAR_SHARES_B
    from rpmdeep.dgp.scenarios_cd import (
        generate_data_scenario_C, generate_data_scenario_D, generate_data_scenario_E,
    )
    if scenario == 'A':
        return generate_mediation_data(n, seed, var_shares=VAR_SHARES_A, lambda_k=LAMBDA_K)
    if scenario == 'B':
        return generate_mediation_data(n, seed, var_shares=VAR_SHARES_B, lambda_k=LAMBDA_K)
    if scenario == 'C':
        return generate_data_scenario_C(n, seed, lambda_k=LAMBDA_K, xi_share=XI_SHARE)
    if scenario == 'D':
        return generate_data_scenario_D(n, seed, lambda_k=LAMBDA_K, gamma=GAMMA, kappa=KAPPA)
    if scenario == 'Dsmall':
        return generate_data_scenario_D(n, seed, lambda_k=LAMBDA_K, gamma=DSMALL_G, kappa=DSMALL_K)
    if scenario == 'E':
        return generate_data_scenario_E(n, seed, lambda_k=LAMBDA_K, w_nonlin=E_W_NONLIN, w_lin=E_W_LIN)
    if scenario == 'Wtau':
        return generate_mediation_data(n, seed, var_shares=VAR_SHARES_WEAK, lambda_k=LAMBDA_K)
    raise ValueError(scenario)


def _tnet_cate(X, T, M, params, seed):
    """5-fold OOF mediator-CATE with CATENets TNet (matched arch to TARNet).
    Mirrors TARNetMediator preprocessing: StandardScaler X, standardized M."""
    cat = REPO_ROOT / 'third_party' / 'catenets'
    if str(cat) not in sys.path:
        sys.path.insert(0, str(cat))
    from catenets.models.jax import TNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    th = np.zeros(len(M))
    for fold, (tr, va) in enumerate(KFold(N_FOLDS_CF, shuffle=True, random_state=seed).split(X)):
        xs = StandardScaler().fit(X[tr]); Xtr = xs.transform(X[tr]); Xva = xs.transform(X[va])
        mu = float(np.mean(M[tr])); msd = float(np.std(M[tr])) or 1.0
        net = TNet(binary_y=False, train_separate=True,
                   n_layers_r=params['n_layers_r'], n_units_r=params['n_units_r'],
                   n_layers_out=params['n_layers_out'], n_units_out=params['n_units_out'],
                   penalty_l2=params['penalty_l2'], n_iter=params['n_iter'],
                   batch_size=params['batch_size'], val_split_prop=params['val_split_prop'],
                   early_stopping=True, patience=params['patience'],
                   n_iter_min=params['n_iter_min'], nonlin='elu', seed=seed + fold)
        net.fit(X=Xtr, y=(M[tr] - mu) / msd, w=T[tr])
        th[va] = np.array(net.predict(Xva)).flatten() * msd
    return th


def run_experiment(scenario, n_samples, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    warnings.filterwarnings('ignore')
    from rpmdeep.pipeline import _tarnet_params, _tau_metrics, _theta_metrics, ALPHA
    from rpmdeep.estimators import GEstimator, BaronKennyEstimator

    t0 = time.time(); z_crit = stats.norm.ppf(1 - ALPHA / 2)
    record = {'n_samples': n_samples, 'seed': seed, 'scenario': scenario,
              'lambda_k': LAMBDA_K, 'dgp': f'tnet_{scenario}',
              'theta1_true': 1.0, 'theta2_true': 0.5}
    try:
        data = _generate(scenario, n_samples, seed)
        X, T, M, Y = data['X'], data['T'], data['M'], data['Y']
        tau_true = data['tau_true']; theta_true = data['theta_true']; prop = data['propensity']
        record['tau_sd_true'] = float(np.std(tau_true))
        record['bk_bias_cf'] = data.get('bk_bias_closed_form', float('nan'))
        record['predicted_gest_bias_theta2'] = data.get('predicted_gest_bias_theta2', 0.0)
        if 'realized_linear_r2' in data:
            record['realized_linear_r2'] = data['realized_linear_r2']

        t_m = time.time()
        try:
            tau_hat = _tnet_cate(X, T, M, _tarnet_params(n_samples), seed)
            s1 = _tau_metrics(tau_hat, tau_true)
            if s1['tau_nrmse'] > 1.3 or float(np.std(tau_hat)) < 1e-4:
                for k, v in s1.items():
                    record[f'tnet_{k}'] = v
                record['tnet_stage2_skipped'] = True
            else:
                g = GEstimator(g_model_type='ridge', n_folds_g=5,
                               variance_method='sandwich', seed=seed)
                g.fit(X, T, M, Y, weights=tau_hat, propensity=prop)
                if g.weakly_identified_:
                    for k, v in s1.items():
                        record[f'tnet_{k}'] = v
                    record['tnet_weakly_identified'] = True
                    record['tnet_stage2_skipped'] = True
                else:
                    s2 = _theta_metrics(g.theta_, g.theta_se_, theta_true)
                    s2['converged'] = bool(g.converged_); s2['n_iterations'] = int(g.n_iterations_)
                    hat = s2['theta2_hat']; se = s2['theta2_se']
                    lo, hi = hat - z_crit * se, hat + z_crit * se
                    s2['theta2_reject_null'] = float(not (lo <= 0.0 <= hi))
                    for k, v in {**s1, **s2}.items():
                        record[f'tnet_{k}'] = v
            record['tnet_time'] = time.time() - t_m
        except Exception as exc:
            record['tnet_error'] = str(exc); record['tnet_time'] = time.time() - t_m

        try:
            bk = BaronKennyEstimator(); bk.fit(X, T, M, Y)
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


def _experiment_path(pool_dir, scenario, n, seed):
    return pool_dir / f'tnet_{scenario}_n{n:06d}_s{seed:05d}.json'


def _experiment_grid():
    return [(sc, n, sd) for sd in range(N_SEEDS) for sc in SCENARIOS for n in SAMPLE_SIZES]


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[v] = '1'


def _save(rec, pool_dir):
    _experiment_path(pool_dir, rec['scenario'], rec['n_samples'], rec['seed']).write_text(
        json.dumps(rec, default=float))


def _print_summary(df):
    print('\n' + '=' * 80)
    print(f'TNet pool ({len(df)} experiments)')
    for sc in SCENARIOS:
        ds = df[df['scenario'] == sc] if 'scenario' in df.columns else df.iloc[0:0]
        if not len(ds):
            continue
        print(f'\n### {sc}')
        for n in SAMPLE_SIZES:
            sub = ds[ds['n_samples'] == n]
            if not len(sub) or 'tnet_theta2_bias' not in sub.columns:
                continue
            nr = sub['tnet_tau_nrmse'].mean() if 'tnet_tau_nrmse' in sub.columns else float('nan')
            co = sub['tnet_tau_corr'].mean() if 'tnet_tau_corr' in sub.columns else float('nan')
            b = sub['tnet_theta2_bias'].median(); se = sub['tnet_theta2_se'].median()
            print(f'  n={n:<6} seeds={len(sub):<4} NRMSE={nr:.3f} corr={co:.3f} '
                  f'theta2_bias={b:+.3f} SE={se:.4f}')


def main():
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-7s  %(message)s',
                        handlers=[logging.FileHandler(POOL_DIR / f'run_{ts}.log'),
                                  logging.StreamHandler(sys.stdout)])
    grid = _experiment_grid()
    missing = [(sc, n, s) for (sc, n, s) in grid if not _experiment_path(POOL_DIR, sc, n, s).exists()]
    logging.info(f'[TNet pool] {POOL_DIR}')
    logging.info(f'Grid {len(grid)} ({len(SCENARIOS)}x{len(SAMPLE_SIZES)}x{N_SEEDS}) '
                 f'— done {len(grid)-len(missing)}, to run {len(missing)}')
    if not missing:
        logging.info('Nothing to do.'); return

    results = [json.loads(p.read_text()) for p in sorted(POOL_DIR.glob('tnet_*_n*_s*.json'))]
    csv_path = POOL_DIR / 'pool_checkpoint.csv'
    it = iter(missing); pending = {}; n_done = 0; t_wall = time.time()

    def _submit(ex):
        try:
            sc, n, s = next(it)
        except StopIteration:
            return False
        pending[ex.submit(run_experiment, sc, n, s)] = (sc, n, s); return True

    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
            for _ in range(N_WORKERS + BUFFER):
                if not _submit(ex):
                    break
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    sc, n, s = pending.pop(fut)
                    try:
                        rec = fut.result()
                    except Exception as exc:
                        rec = {'scenario': sc, 'n_samples': n, 'seed': s, 'global_error': str(exc)}
                    _save(rec, POOL_DIR); results.append(rec); n_done += 1; _submit(ex)
                    if n_done % 5 == 0 or n_done == len(missing):
                        el = time.time() - t_wall; rate = n_done / el if el else 1e-9
                        cnt = {c: sum(1 for r in results if r.get('scenario') == c) for c in SCENARIOS}
                        logging.info(f'[{n_done}/{len(missing)}] {el/60:.1f}m '
                                     f'ETA={ (len(missing)-n_done)/rate/60:.0f}m | '
                                     + '  '.join(f'{k}:{v}' for k, v in cnt.items()))
                    if n_done % CHECKPOINT_N == 0 or n_done == len(missing):
                        pd.DataFrame(results).to_csv(csv_path, index=False)
    except KeyboardInterrupt:
        logging.warning('Interrupted; saving.'); pd.DataFrame(results).to_csv(csv_path, index=False)

    df = pd.DataFrame(results); df.to_csv(csv_path, index=False)
    logging.info(f'Saved {len(df)} rows -> {csv_path}'); _print_summary(df)


if __name__ == '__main__':
    main()
