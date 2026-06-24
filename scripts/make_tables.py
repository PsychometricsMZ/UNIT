#!/usr/bin/env python
"""Canonical results aggregator.

Reads the per-experiment JSON checkpoints from the scenario pools and
deterministically rebuilds the headline aggregate and the table/CSV artifacts.

Reproducibility contract
------------------------
1.  Deterministic seed freeze.  Each cell uses the first ``N_FREEZE`` seeds by
    seed id (sorted ascending) from the committed files.  Extra seeds beyond the
    freeze are ignored so the number of seeds behind every table is fixed.

2.  Guard / filter at aggregation.  A seed contributes to a cell x method
    statistic ONLY if its Stage-2 fit is valid, i.e. it is NOT any of:
      * errored        ({m}_error present)
      * stage2-skipped ({m}_stage2_skipped, NRMSE>1.3 or degenerate tau)
      * weakly identified ({m}_weakly_identified)
      * non-converged  ({m}_converged is False)
      * non-finite theta-hat / SE
      * divergent      max(|theta1_hat-theta1|, |theta2_hat-theta2|) > THETA_GUARD
      * blown SE       theta2_se > SE_GUARD (degenerate / near-singular G-moment:
                       for an O(0.5) estimand a well-identified SE is <1; >2 means
                       an uninformative CI that can only spuriously "cover")
    Excluded seeds are counted in ``skip``, never in ``used``.  Coverage is a
    mean over valid rows only, so a divergent seed can never inflate a coverage
    cell.  Point summaries use the MEDIAN (bias, SE) -- robust to the heavy tails
    of G-estimation at small n -- with the std of theta2-hat reported as ``sd``.

3.  One builder, one aggregate.  Output is written from raw, so every printed
    number round-trips from committed artifacts.

Outputs
-------
  results/tables/summary_canonical.json   canonical aggregate
  results/tables/cells.csv                tidy long form, one row per cell x method
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_FREEZE = 200          # seeds per cell (deterministic: first N_FREEZE by seed id)
THETA_GUARD = 50.0      # |theta_hat - theta_true| divergence guard (applied here,
                        # at aggregation: this script is the single place a seed is
                        # judged valid for a cell)
SE_GUARD = 2.0          # theta2_se plausibility guard; >2 => degenerate G-moment
                        # (estimand ~0.5, median valid SE 0.07, 99% of fits <1)

SCENARIOS = ["A", "B", "C", "D", "E", "Dsmall", "Wtau"]
SAMPLE_SIZES = [500, 1000, 2000, 5000, 10000]
# Scenario-pool methods (ridge/rf_tuned/tarnet all share one record per seed);
# TNet is sourced separately from pool_tnet (one record per seed).  Output order
# follows the four-learner ladder Ridge -> RF -> TNet -> TARNet.
POOL_METHODS = ["ridge", "rf_tuned", "tarnet"]
LADDER = ["ridge", "rf_tuned", "tnet", "tarnet"]

# scenario -> (pool dir, file prefix); files are "{prefix}_n{n:06d}_s{seed}.json"
POOLS = {
    "A":      (RESULTS / "pool_scenarioa",    "scena"),
    "B":      (RESULTS / "pool_scenariob",    "scenb"),
    "C":      (RESULTS / "pool_scenario_cd",  "cd_C"),
    "D":      (RESULTS / "pool_scenario_cd",  "cd_D"),
    "E":      (RESULTS / "pool_scenario_cd",  "cd_E"),
    "Dsmall": (RESULTS / "pool_scenario_ext", "ext_Dsmall"),
    "Wtau":   (RESULTS / "pool_scenario_ext", "ext_Wtau"),
}
# TNet (separate per-arm neural learner) lives in its own pool, one method
# ("tnet") per record, prefix "tnet_{scenario}".
TNET_DIR = RESULTS / "pool_tnet"

_SEED_RE = re.compile(r"_s(\d+)\.json$")


# --------------------------------------------------------------------------- #
# loading + validity filter
# --------------------------------------------------------------------------- #
def load_cell(scenario: str, n: int) -> list[dict]:
    """First N_FREEZE seed records for a cell, sorted by seed id (deterministic)."""
    pool_dir, prefix = POOLS[scenario]
    pattern = str(pool_dir / f"{prefix}_n{n:06d}_s*.json")
    by_seed: dict[int, str] = {}
    for f in glob.glob(pattern):
        m = _SEED_RE.search(f)
        if m:
            by_seed.setdefault(int(m.group(1)), f)
    recs = []
    for seed in sorted(by_seed)[:N_FREEZE]:
        with open(by_seed[seed]) as fh:
            recs.append(json.load(fh))
    return recs


def load_cell_tnet(scenario: str, n: int) -> list[dict]:
    """First N_FREEZE TNet seed records for a cell, from the separate tnet pool."""
    pattern = str(TNET_DIR / f"tnet_{scenario}_n{n:06d}_s*.json")
    by_seed: dict[int, str] = {}
    for f in glob.glob(pattern):
        m = _SEED_RE.search(f)
        if m:
            by_seed.setdefault(int(m.group(1)), f)
    recs = []
    for seed in sorted(by_seed)[:N_FREEZE]:
        with open(by_seed[seed]) as fh:
            recs.append(json.load(fh))
    return recs


def is_valid(rec: dict, m: str) -> bool:
    """True iff this seed's Stage-2 fit for method ``m`` may enter a statistic."""
    if f"{m}_error" in rec:
        return False
    if rec.get(f"{m}_stage2_skipped"):
        return False
    if rec.get(f"{m}_weakly_identified"):
        return False
    if rec.get(f"{m}_converged") is False:
        return False
    hat2 = rec.get(f"{m}_theta2_hat")
    se2 = rec.get(f"{m}_theta2_se")
    if hat2 is None or se2 is None:
        return False
    if not (np.isfinite(hat2) and np.isfinite(se2)):
        return False
    if se2 > SE_GUARD:          # blown SE -> degenerate fit, CI uninformative
        return False
    # divergence guard on the full theta vector
    hat1 = rec.get(f"{m}_theta1_hat", rec.get("theta1_true"))
    t1 = rec.get("theta1_true")
    t2 = rec.get("theta2_true")
    if t1 is not None and t2 is not None:
        if max(abs(hat1 - t1), abs(hat2 - t2)) > THETA_GUARD:
            return False
    return True


def _median(xs):
    return float(np.median(xs)) if xs else float("nan")


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _stage1_metric(recs: list[dict], key: str) -> float:
    """Mean of a Stage-1 metric over every seed that recorded it (independent of
    Stage-2 validity), so a skipped Stage-2 still counts its learner error."""
    vals = [r[key] for r in recs
            if isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
    return _mean(vals)


def aggregate_method(recs: list[dict], m: str) -> dict:
    """Aggregate one learner ``m`` over the seed records that contain its fields."""
    valid = [r for r in recs if is_valid(r, m)]
    used = len(valid)
    return {
        "bias": _median([r[f"{m}_theta2_bias"] for r in valid]),
        "se":   _median([r[f"{m}_theta2_se"] for r in valid]),
        "sd":   float(np.std([r[f"{m}_theta2_hat"] for r in valid])) if valid else float("nan"),
        "cov":  _mean([r[f"{m}_theta2_covered"] for r in valid]),
        "corr": _stage1_metric(recs, f"{m}_tau_corr"),
        "nrmse": _stage1_metric(recs, f"{m}_tau_nrmse"),
        "skip": len(recs) - used,   # everything excluded by the validity filter
        "used": used,
    }


def se_ratio(recs: list[dict], num: str, den: str = "tarnet") -> float:
    """Median per-seed SE ratio num/den over seeds where BOTH fits are valid.
    (Median of ratios, not ratio of medians -- matches the paper's definition.)"""
    rs = []
    for r in recs:
        if is_valid(r, num) and is_valid(r, den):
            d = r[f"{den}_theta2_se"]
            if d > 0:
                rs.append(r[f"{num}_theta2_se"] / d)
    return _median(rs)


def aggregate_cell(recs: list[dict], tnet_recs: list[dict]) -> dict:
    out: dict = {"n_seeds": len(recs)}
    for m in POOL_METHODS:
        out[m] = aggregate_method(recs, m)
    if tnet_recs:
        out["tnet"] = aggregate_method(tnet_recs, "tnet")
        out["n_seeds_tnet"] = len(tnet_recs)
    out["rt_ratio"]  = se_ratio(recs, "ridge")      # Ridge / TARNet
    out["rft_ratio"] = se_ratio(recs, "rf_tuned")   # RF / TARNet
    bk = [r for r in recs if isinstance(r.get("baron_kenny_theta2_bias"), (int, float))]
    out["bk"] = {
        "bias": _median([r["baron_kenny_theta2_bias"] for r in bk]),
        "cov":  _mean([r["baron_kenny_theta2_covered"] for r in bk]),
    }
    return out


def build() -> dict:
    agg: dict = {}
    for sc in SCENARIOS:
        agg[sc] = {}
        for n in SAMPLE_SIZES:
            recs = load_cell(sc, n)
            if recs:
                agg[sc][str(n)] = aggregate_cell(recs, load_cell_tnet(sc, n))
    return agg


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
def write_json(agg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agg, indent=2))


def write_csv(agg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scenario", "n", "method", "bias", "se", "sd", "cov",
            "corr", "nrmse", "skip", "used", "n_seeds"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for sc in SCENARIOS:
            for n in SAMPLE_SIZES:
                cell = agg.get(sc, {}).get(str(n))
                if not cell:
                    continue
                for m in LADDER:
                    d = cell.get(m)
                    if not d:
                        continue   # tnet may be absent for a cell with no pool data
                    n_seeds = cell["n_seeds_tnet"] if m == "tnet" else cell["n_seeds"]
                    w.writerow([sc, n, m, d["bias"], d["se"], d["sd"], d["cov"],
                                d["corr"], d["nrmse"], d["skip"], d["used"], n_seeds])
                bk = cell["bk"]
                w.writerow([sc, n, "bk", bk["bias"], "", "", bk["cov"],
                            "", "", "", "", cell["n_seeds"]])


# --------------------------------------------------------------------------- #
# LaTeX table emit
# --------------------------------------------------------------------------- #
DISPLAY = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"),
           ("Dsmall", "Dsmall"), ("E", "E"), ("Wtau", "Wtau")]


def _d2(x):
    """2 decimals, no leading zero (e.g. 0.96 -> .96, 1.18 -> 1.18)."""
    if x is None or not np.isfinite(x):
        return "---"
    s = f"{abs(x):.2f}"
    s = s[1:] if s.startswith("0.") else s
    return ("$-$" if x < 0 else "") + s


def _b3(x):
    """signed 3-decimal in math mode, no leading zero (0.029 -> $+.029$)."""
    if x is None or not np.isfinite(x):
        return "---"
    s = f"{abs(x):.3f}"
    s = s[1:] if s.startswith("0.") else s
    return f"${'-' if x < 0 else '+'}{s}$"


def _r2(x):
    return "---" if (x is None or not np.isfinite(x)) else f"{x:.2f}"


def _tab_skip(agg: dict) -> str:
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"Scenario & Ridge & RF & TNet & TARNet\\", r"\midrule"]
    for key, disp in DISPLAY:
        c = agg.get(key, {}).get("500")
        if not c:
            L.append(f"{disp} & --- & --- & --- & ---\\\\")
            continue
        sk = lambda m: str(c[m]["skip"]) if c.get(m) else "---"
        L.append(f"{disp} & {sk('ridge')} & {sk('rf_tuned')} & {sk('tnet')} & {sk('tarnet')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def _tab_stage1(agg: dict, field: str) -> str:
    """Wide ladder table (4 numbers/cell: Ridge/RF/TNet/TARNet, see caption).
    Wrapped in \\resizebox + tightened \\tabcolsep so it fits the text width; the
    learner key lives in the caption, keeping the header short. Needs graphicx."""
    rows = []
    for key, disp in DISPLAY:
        cells = []
        for n in SAMPLE_SIZES:
            c = agg.get(key, {}).get(str(n))
            if not c:
                cells.append("/".join("--" for _ in LADDER))
                continue
            cells.append("/".join(_d2(c[m][field]) if c.get(m) else "--" for m in LADDER))
        rows.append(f"{disp} & " + " & ".join(cells) + r"\\")
    L = [r"\setlength{\tabcolsep}{4pt}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
         r"Scenario & $n{=}500$ & $1000$ & $2000$ & $5000$ & $10000$\\",
         r"\midrule",
         *rows,
         r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(L)


def _tab_results(agg: dict) -> str:
    """Full four-learner ladder. TT/T = TNet (not-shared) / TARNet (shared) SE,
    the parameter-sharing efficiency cost; ratio of median SEs (TNet is a separate
    pool, so it cannot be paired per seed like R/T and RF/T)."""
    L = [r"\begin{tabular}{@{}cl cccc c ccc@{}}", r"\toprule",
         r"& $n$ & Ridge $b$(cov) & RF $b$(cov) & TNet $b$(cov) & TARNet $b$(cov) "
         r"& BK & R/T & RF/T & TT/T\\",
         r"\midrule"]
    for i, (key, disp) in enumerate(DISPLAY):
        for j, n in enumerate(SAMPLE_SIZES):
            c = agg.get(key, {}).get(str(n))
            mr = f"\\multirow{{5}}{{*}}{{{disp}}}" if j == 0 else ""
            if not c:
                L.append(f"{mr} & {n} & --- & --- & --- & --- & --- & --- & --- & ---\\\\")
                continue
            def bc(m):
                d = c.get(m)
                return f"{_b3(d['bias'])} ({_d2(d['cov'])})" if d else "---"
            t = c["tarnet"]["se"]
            tn = c.get("tnet", {}).get("se")
            tt = (tn / t) if (tn is not None and np.isfinite(tn) and np.isfinite(t) and t > 0) else float("nan")
            L.append(f"{mr} & {n} & {bc('ridge')} & {bc('rf_tuned')} & {bc('tnet')} & {bc('tarnet')} "
                     f"& {_b3(c['bk']['bias'])} & {_r2(c['rt_ratio'])} & {_r2(c['rft_ratio'])} & {_r2(tt)}\\\\")
        if i < len(DISPLAY) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def _pooled_rt(key: str, sizes=(2000, 5000, 10000)) -> float:
    rs = []
    for n in sizes:
        for r in load_cell(key, n):
            if is_valid(r, "ridge") and is_valid(r, "tarnet"):
                d = r["tarnet_theta2_se"]
                if d > 0:
                    rs.append(r["ridge_theta2_se"] / d)
    return _median(rs)


def _tab_efficiency(agg: dict) -> str:
    L = [r"\begin{tabular}{@{}l" + "c" * len(DISPLAY) + r"@{}}", r"\toprule",
         "Scenario & " + " & ".join(d for _, d in DISPLAY) + r"\\", r"\midrule"]
    L.append("Median R/T SE ratio ($n\\ge2000$) & " +
             " & ".join(_r2(_pooled_rt(k)) for k, _ in DISPLAY) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def write_latex(agg: dict, path: Path) -> str:
    blocks = {
        "skip":  _tab_skip(agg),
        "nrmse": _tab_stage1(agg, "nrmse"),
        "corr":  _tab_stage1(agg, "corr"),
        "results": _tab_results(agg),
        "efficiency": _tab_efficiency(agg),
    }
    doc = ["% Auto-generated by scripts/make_tables.py -- do not edit by hand.\n"]
    for name, body in blocks.items():
        doc.append(f"% ===== table: {name} =====\n{body}\n")
    path.write_text("\n".join(doc))
    return blocks["results"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonical UNIT/RPMDEEP table builder.")
    ap.add_argument("--out-dir", default=str(RESULTS / "tables"),
                    help="directory for canonical artifacts")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    agg = build()
    write_json(agg, out_dir / "summary_canonical.json")
    write_csv(agg, out_dir / "cells.csv")
    write_latex(agg, out_dir / "tables.tex")

    # console summary: the four-learner corr(tau_hat, tau) ladder per cell
    print(f"freeze={N_FREEZE} seeds/cell  guard=|theta-theta|>{THETA_GUARD}, SE>{SE_GUARD}")
    print(f"  corr(tau_hat,tau) ladder      {'ridge':>7}{'rf':>7}{'tnet':>7}{'tarnet':>8}")
    for sc in SCENARIOS:
        for n in SAMPLE_SIZES:
            cell = agg.get(sc, {}).get(str(n))
            if not cell:
                continue
            def c(m):
                d = cell.get(m)
                return f"{d['corr']:.3f}" if d and np.isfinite(d['corr']) else "  -  "
            print(f"  {sc:>7} n={n:<6}        "
                  f"{c('ridge'):>7}{c('rf_tuned'):>7}{c('tnet'):>7}{c('tarnet'):>8}")
    print(f"\nwrote {out_dir/'summary_canonical.json'}")
    print(f"wrote {out_dir/'cells.csv'}")
    print(f"wrote {out_dir/'tables.tex'}")


if __name__ == "__main__":
    main()
