#!/usr/bin/env python
"""Plot Stage-1 CATE quality vs sample size, per scenario, per learner.

Reproducible companion to scripts/make_tables.py: reads the canonical aggregate
(results/tables/summary_canonical.json) and renders the visual counterparts of
Tables 2 (NRMSE) and 3 (corr) to figures/stage1_{nrmse,corr}.pdf.

    .venv/bin/python scripts/plot_stage1_quality.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "results" / "tables" / "summary_canonical.json"
FIGDIR = ROOT / "figures"
SIZES = [500, 1000, 2000, 5000, 10000]

# scenario data-key -> paper label (F is displayed as E)
DISPLAY = [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"),
           ("Dsmall", "Dsmall"), ("E", "E"), ("Wtau", "Wtau")]

# learner key, label, colour, marker, linestyle (ladder order)
LEARNERS = [
    ("ridge",    "Ridge",  "#777777", "s", "--"),
    ("rf_tuned", "RF",     "#2ca02c", "^", "--"),
    ("tnet",     "TNet",   "#1f77b4", "o", "-"),
    ("tarnet",   "TARNet", "#d62728", "D", "-"),
]


def _series(agg, key, m, field):
    ys = []
    for n in SIZES:
        c = agg.get(key, {}).get(str(n))
        cell = c.get(m) if c else None
        v = cell[field] if cell else None
        ys.append(v if (v is not None and v == v) else float("nan"))
    return ys


def plot_metric(agg, field, ylabel, title, outfile, ylim=None, hlines=()):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (key, disp) in zip(axes, DISPLAY):
        for hy in hlines:
            ax.axhline(hy, color="k", lw=0.6, ls=":")
        for m, label, color, marker, ls in LEARNERS:
            ax.plot(SIZES, _series(agg, key, m, field), marker=marker, color=color,
                    ls=ls, label=label, ms=5, lw=1.6)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels(["500", "1k", "2k", "5k", "10k"], fontsize=8)
        ax.set_title(f"Scenario {disp}", fontsize=10)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)
    # spare panel -> legend
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=11,
                    title="Stage-1 learner")
    for i in (0, 4):
        axes[i].set_ylabel(ylabel)
    for i in (3, 4, 5, 6):
        axes[i].set_xlabel("$n$")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print("wrote", outfile)


def main():
    agg = json.loads(AGG.read_text())
    plot_metric(agg, "nrmse", "NRMSE",
                r"Stage-1 CATE quality: NRMSE vs $n$ (lower is better)",
                FIGDIR / "stage1_nrmse.pdf", ylim=(0.2, 2.4), hlines=(1.0, 1.3))
    plot_metric(agg, "corr", r"$\mathrm{Corr}(\hat\tau,\tau)$",
                r"Stage-1 CATE quality: correlation vs $n$ (higher is better)",
                FIGDIR / "stage1_corr.pdf", ylim=(0.2, 1.0))


if __name__ == "__main__":
    main()
