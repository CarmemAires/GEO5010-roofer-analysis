#!/usr/bin/env python3
"""
Bayesian optimization of Roofer reconstruction parameters using Optuna (TPE).

This script performs tile-wise Bayesian hyperparameter optimization for Roofer
using a quality-based objective function derived from CityJSON outputs.
Each trial runs Roofer with a sampled parameter set, evaluates reconstruction
quality based on 3D validity (val3dity) and extrusion success, and maximizes
the average quality ratio across tiles.

Key features:
- Tree-structured Parzen Estimator (TPE) sampler via Optuna
- Parallel evaluation of trials
- Early stopping for poor parameter sets after the first tile
- Automated conversion from CityJSONL to CityJSON using cjio

⚠️  WARNING:
This script executes Roofer and may overwrite outputs in the working directory.
Ensure input data are backed up and paths are correctly configured before running.

Requirements:
- Python ≥ 3.9
- Roofer executable available on the system
- cjio (CityJSON tools) available on PATH

Configuration:
- Set WORK_DIR, ROOFER_EXE.

Author: Carmem E. F. Aires
Date: 2026-01-12
"""

from __future__ import annotations
import subprocess
import json
import logging
from pathlib import Path
import time
import re
from typing import Any, List
import optuna
from concurrent.futures import ProcessPoolExecutor

# ===================== CONFIG =====================

WORK_DIR = Path(r"data/output") # Set the path to the working directory containing .laz and .gpkg files

ROOFER_EXE = Path(r"bin/roofer.exe") # Set the path to the Roofer executable

CJIO_CMD = ["cjio", "stdin", "save", "stdout"]

# ---- Bayesian optimization ----
N_TRIALS = 12
N_JOBS = 5        # parallel workers

# ---- early stopping threshold ----
EARLY_STOP_RATIO = 0.96   # after first tile

# =================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def find_matching_gpkg(laz: Path, gpkg_files: list[Path]) -> Path | None:
    for g in gpkg_files:
        if laz.stem in g.stem:
            return g
    return None


def run_roofer(
    laz: Path,
    gpkg: Path,
    out_dir: Path,
    params: list[str]
) -> Path | None:

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(ROOFER_EXE)] + params + [str(laz), str(gpkg), str(out_dir)]

    logging.info("RUNNING ROOFER:")
    logging.info(" ".join(cmd))

    start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        logging.info("[roofer] " + line.rstrip())

    ret = proc.wait()
    elapsed = time.time() - start

    logging.info(f"Roofer finished in {elapsed:.1f}s (exit={ret})")

    if ret != 0:
        return None

    outputs = sorted(
        out_dir.glob("*.city.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return outputs[0] if outputs else None


def convert_cityjson(jsonl: Path, out_json: Path) -> None:
    with open(jsonl, "rb") as fin, open(out_json, "wb") as fout:
        proc = subprocess.run(
            CJIO_CMD,
            stdin=fin,
            stdout=fout,
            stderr=subprocess.PIPE
        )

    if proc.returncode != 0:
        raise RuntimeError("cjio failed")


def evaluate_quality(cityjson_path: Path) -> float:
    """
    Calculate quality ratio based on:
    1. Valid 3D geometry (rf_val3dity_lod22 has no error codes)
    2. Successful extrusion (rf_extrusion_mode != "skip")
    
    Matches the quality indicator logic from analyze_cityjson_quality.py
    but without the footprint overlap check.
    """
    with open(cityjson_path, "r", encoding="utf-8") as f:
        cj = json.load(f)

    buildings = [
        obj for obj in cj["CityObjects"].values()
        if obj.get("type") == "Building"
    ]

    if not buildings:
        raise RuntimeError("No buildings reconstructed")

    total = 0
    valid = 0

    for obj in buildings:
        attrs = obj.get("attributes", {})
        total += 1

        # Parse val3dity codes (could be string like "[102]", list, or None)
        val3dity_raw = attrs.get("rf_val3dity_lod22")
        val3dity_codes = parse_val3dity_codes(val3dity_raw)
        
        # Check validation: valid if no error codes
        val3dity_valid = len(val3dity_codes) == 0
        
        # Check extrusion: successful if not "skip"
        extrusion = attrs.get("rf_extrusion_mode")
        extrusion_ok = extrusion != "skip"

        # Quality passes if both conditions are met
        if val3dity_valid and extrusion_ok:
            valid += 1

    logging.info(f"Quality evaluation: {valid}/{total} buildings passed ({valid/total:.1%})")
    return valid / total


def parse_val3dity_codes(val3dity_data: Any) -> List[int]:
    """
    Parse val3dity error codes from various possible formats.
    Handles: "[102]", "[102, 103]", "102", 102, [102], None, etc.
    """
    if val3dity_data is None:
        return []
    
    # If it's already a list of integers, return it
    if isinstance(val3dity_data, list):
        return [int(x) for x in val3dity_data if str(x).strip().isdigit()]
    
    # If it's a string, try to extract numbers
    if isinstance(val3dity_data, str):
        # Remove brackets and split by commas
        cleaned = val3dity_data.strip('[]')
        if not cleaned:
            return []
        
        # Extract all numbers from the string
        import re
        numbers = re.findall(r'\d+', cleaned)
        return [int(num) for num in numbers if num.strip().isdigit()]
    
    # If it's a single integer
    if isinstance(val3dity_data, int):
        return [val3dity_data]
    
    # If it's a single float
    if isinstance(val3dity_data, float):
        return [int(val3dity_data)]
    
    logging.warning(f"Unexpected val3dity data type: {type(val3dity_data)} = {val3dity_data}")
    return []


# -------------------------------------------------
# Objective function for Optuna
# -------------------------------------------------

def objective(trial: optuna.Trial) -> float:

    # ---- Bayesian parameter sampling ----
    complexity = trial.suggest_float("complexity", 0.6, 1.0)
    min_points = trial.suggest_int("min_points", 10, 30)
    epsilon = trial.suggest_float("epsilon", 0.1, 0.3)
    k = trial.suggest_categorical("k", [15, 20])

    params = [
        "--complexity-factor", f"{complexity:.3f}",
        "--plane-detect-min-points", str(min_points),
        "--plane-detect-epsilon", f"{epsilon:.3f}",
        "--plane-detect-k", str(k)
    ]

    logging.info(f"TRIAL {trial.number} params = {params}")

    # ---- Find all .laz files in the directory ----
    laz_files = sorted(WORK_DIR.glob("*.laz"))
    
    if not laz_files:
        raise RuntimeError(f"No .laz files found in {WORK_DIR}")
    
    logging.info(f"Found {len(laz_files)} tiles to process")
    
    gpkg_files = list(WORK_DIR.glob("*.gpkg"))

    run_dir = WORK_DIR / "bayes_runs" / f"trial_{trial.number}"
    run_dir.mkdir(parents=True, exist_ok=True)

    ratios = []

    for i, laz in enumerate(laz_files):

        gpkg = find_matching_gpkg(laz, gpkg_files)
        if gpkg is None:
            logging.warning(f"No GPKG for {laz.name}, skipping")
            continue

        tile_out = run_dir / laz.stem
        tile_out.mkdir(exist_ok=True)

        jsonl = run_roofer(laz, gpkg, tile_out, params)
        if not jsonl:
            logging.warning(f"Roofer failed for {laz.name}")
            continue

        cityjson = tile_out / (jsonl.stem + ".city.json")
        convert_cityjson(jsonl, cityjson)

        ratio = evaluate_quality(cityjson)
        ratios.append(ratio)

        # ---- EARLY STOP after first tile ----
        if i == 0 and ratio < EARLY_STOP_RATIO:
            logging.info("Early stopping this trial (bad params)")
            return ratio

    if not ratios:
        return 0.0

    return sum(ratios) / len(ratios)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=4,  # random warm-up
        multivariate=True
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=N_JOBS
    )

    print("\n" + "=" * 80)
    print("BEST PARAMETERS (BAYESIAN)")
    print("=" * 80)
    print(study.best_trial)

    print("\nRecommended ROOFER_PARAMS =")
    print([
        "--complexity-factor", f"{study.best_params['complexity']:.3f}",
        "--plane-detect-min-points", str(study.best_params['min_points']),
        "--plane-detect-epsilon", f"{study.best_params['epsilon']:.3f}",
        "--plane-detect-k", str(study.best_params['k']),
    ])


if __name__ == "__main__":
    main()