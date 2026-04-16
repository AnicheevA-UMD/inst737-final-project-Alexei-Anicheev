"""
main.py - Project Pipeline Entry Point
========================================
Runs the full data science pipeline in sequence:
    1. Extract  — pulls datasets from Maryland Open Data API
    2. Transform — cleans, standardizes, runs EDA
    3. Load     — merges datasets into analysis-ready file
    4. Model 1  — K-Means clustering
    5. Model 2  — DBSCAN clustering
    6. Model 3  — Random Forest regression (scaffolding, Parts 3-4)
    7. Visualize — produces analytical output charts

Each stage can also be run independently via its own .py file.

Usage:
    python main.py                # full pipeline
    python main.py --skip-extract # skip API calls, use cached data
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Project root is wherever this file lives
PROJECT_ROOT = Path(__file__).resolve().parent


def run_stage(name, script_path, allow_failure=False):
    """
    Run a single pipeline stage as a subprocess.

    Each stage is run as its own process so that import paths
    resolve correctly relative to each script's location.

    Parameters
    ----------
    name : str
        Human-readable name of the stage (for console output).
    script_path : str
        Path to the .py file to execute, relative to project root.
    allow_failure : bool
        If True, continue the pipeline even if this stage fails.
        Used for scaffolding stages that aren't fully implemented.

    Returns
    -------
    bool
        True if the stage succeeded, False otherwise.
    """
    full_path = PROJECT_ROOT / script_path

    print(f"\n{'#' * 60}")
    print(f"# STAGE: {name}")
    print(f"# Script: {script_path}")
    print(f"{'#' * 60}\n")

    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode != 0:
        if allow_failure:
            print(f"\n  ⚠ {name} exited with code {result.returncode}")
            print(f"  Continuing pipeline (allow_failure=True)")
            return False
        else:
            print(f"\n  ❌ {name} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    return True


def main():
    """
    Entry point for the full project pipeline.

    Parses command-line arguments and runs each pipeline stage
    in sequence. The --skip-extract flag allows skipping the
    API extraction stage when working with cached data.

    Parameters
    ----------
    None (reads from command-line arguments)

    Returns
    -------
    None
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip API extraction (use existing data in data/extracted/)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Maryland Foreclosure Indicator Analysis")
    print("  Full Pipeline Execution")
    print("=" * 60)

    # ── ETL Pipeline ──

    if not args.skip_extract:
        run_stage("EXTRACTION", "etl/extract.py")
    else:
        print("\n  Skipping extraction (--skip-extract flag set)")

    run_stage("TRANSFORMATION", "etl/transform.py")

    run_stage("LOAD", "etl/load.py")

    # ── Analysis ──

    run_stage("K-MEANS CLUSTERING", "analysis/model_1.py")

    run_stage("DBSCAN CLUSTERING", "analysis/model_2.py")

    run_stage("RANDOM FOREST REGRESSION", "analysis/model_3.py")

    # ── Visualization ──

    run_stage("VISUALIZATIONS", "vis/visualizations.py")

    # ── Done ──

    print(f"\n{'#' * 60}")
    print("# ✅ PIPELINE COMPLETE")
    print(f"{'#' * 60}")
    print("\nOutputs:")
    print("  data/extracted/      — raw API pulls")
    print("  data/transformed/    — cleaned datasets")
    print("  data/load/           — merged analysis-ready data")
    print("  data/model_outputs/  — cluster labels, profiles, RF metrics")
    print("  data/visualizations/ — all charts and plots")


# Standard Python entry point guard
if __name__ == "__main__":
    main()