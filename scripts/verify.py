#!/usr/bin/env python
"""Verify the replication package: data integrity + table reproducibility.

Two independent checks:

1. **Integrity** -- every committed per-seed record under ``results/pool_*/``
   matches the sha256 recorded in ``results/MANIFEST.json`` (nothing has been
   edited or corrupted), and the count is exactly 200 seeds x (scenario x n).

2. **Reproducibility** -- re-running ``scripts/make_tables.py`` against the
   committed data reproduces ``results/tables/{summary_canonical.json,
   cells.csv, tables.tex}`` with sha256 matching the values the MANIFEST
   certifies. (Only numpy is required for this; jax/CATENets are not.)

Usage:
    python scripts/verify.py            # run both checks (exit 0 iff all pass)
    python scripts/verify.py --write    # (re)generate results/MANIFEST.json
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
POOL_DIRS = ["pool_scenarioa", "pool_scenariob", "pool_scenario_cd",
             "pool_scenario_ext", "pool_tnet"]
TABLE_FILES = ["summary_canonical.json", "cells.csv", "tables.tex"]
N_FREEZE = 200
SAMPLE_SIZES = [500, 1000, 2000, 5000, 10000]
# Number of distinct scenarios materialised in each pool dir. The frozen freeze
# is exactly N_FREEZE seeds x len(SAMPLE_SIZES) cells per scenario, so the
# expected file count per dir is fully determined (and independent of MANIFEST).
POOL_SCENARIOS = {
    "pool_scenarioa":    1,   # A
    "pool_scenariob":    1,   # B
    "pool_scenario_cd":  3,   # C, D, E
    "pool_scenario_ext": 2,   # Dsmall, Wtau
    "pool_tnet":         7,   # all scenarios, neural T-learner arm
}
MANIFEST = RESULTS / "MANIFEST.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _pool_files() -> list[Path]:
    files: list[Path] = []
    for d in POOL_DIRS:
        files.extend(sorted((RESULTS / d).glob("*.json")))
    return files


def write_manifest() -> None:
    inputs = [{"path": str(p.relative_to(RESULTS)), "sha256": _sha256(p)}
              for p in _pool_files()]
    outputs = {f: _sha256(RESULTS / "tables" / f) for f in TABLE_FILES}
    manifest = {
        "n_freeze": N_FREEZE,
        "n_input_files": len(inputs),
        "table_outputs_sha256": outputs,
        "inputs": sorted(inputs, key=lambda d: d["path"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {MANIFEST} ({len(inputs)} inputs)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="(re)generate results/MANIFEST.json and exit")
    args = ap.parse_args()
    if args.write:
        write_manifest()
        return 0

    manifest = json.loads(MANIFEST.read_text())
    ok = True

    # 1. integrity ---------------------------------------------------------
    # 1a. the 200-seeds x (scenario x n) contract, checked against the files
    #     actually on disk -- independent of anything the manifest records, so a
    #     short or partially-completed pool fails here.
    disk_files = _pool_files()
    for d, n_scen in POOL_SCENARIOS.items():
        have = len(sorted((RESULTS / d).glob("*.json")))
        want = n_scen * len(SAMPLE_SIZES) * N_FREEZE
        print(f"[count] {d:20s} {have:5d} files (expected {want}): "
              f"{'OK' if have == want else 'MISMATCH'}")
        ok &= (have == want)
    expected_total = sum(n_scen for n_scen in POOL_SCENARIOS.values()) \
        * len(SAMPLE_SIZES) * N_FREEZE
    print(f"[count] total {len(disk_files)} files (expected {expected_total}): "
          f"{'OK' if len(disk_files) == expected_total else 'MISMATCH'}")
    ok &= (len(disk_files) == expected_total)

    # 1b. the manifest must enumerate exactly the files on disk.
    n = len(manifest["inputs"])
    print(f"[integrity] manifest lists {n} input files "
          f"(on disk {len(disk_files)}, self-reported {manifest['n_input_files']}): "
          f"{'OK' if n == len(disk_files) == manifest['n_input_files'] else 'MISMATCH'}")
    ok &= (n == len(disk_files) == manifest["n_input_files"])

    bad = 0
    for rec in manifest["inputs"]:
        p = RESULTS / rec["path"]
        if not p.exists() or _sha256(p) != rec["sha256"]:
            bad += 1
            if bad <= 5:
                print(f"    CORRUPT/MISSING: {rec['path']}")
    print(f"[integrity] {n - bad}/{n} inputs match their sha256: "
          f"{'OK' if bad == 0 else f'{bad} BAD'}")
    ok &= (bad == 0)

    # 2. reproducibility ---------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_tables.py"),
                        "--out-dir", td], check=True, cwd=ROOT,
                       stdout=subprocess.DEVNULL)
        for f in TABLE_FILES:
            got = _sha256(Path(td) / f)
            want = manifest["table_outputs_sha256"][f]
            live = _sha256(RESULTS / "tables" / f)
            match = (got == want == live)
            print(f"[reproduce] {f:28s} rebuilt->{got[:12]} "
                  f"manifest->{want[:12]} committed->{live[:12]} "
                  f"{'OK' if match else 'MISMATCH'}")
            ok &= match

    print("\nRESULT:",
          "ALL CHECKS PASS" if ok else "FAILED -- see mismatches above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
