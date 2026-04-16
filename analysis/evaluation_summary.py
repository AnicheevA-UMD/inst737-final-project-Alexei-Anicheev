"""
evaluation_summary.py - Model Evaluation Summary
==================================================
Consolidates evaluation metrics from all three models into a
single summary CSV and a comparison chart. Reads from the
per-model outputs in data/model_outputs/ and writes the
consolidated summary to the same directory.

Usage:
    python analysis/evaluation_summary.py
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path so logging_config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logging_config import get_logger

logger = get_logger(__name__)


# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"


def collect_model_metrics():
    """
    Read each model's evaluation output and assemble a unified
    summary suitable for side-by-side comparison.

    Returns
    -------
    pd.DataFrame
        Long-format summary with columns: model, model_type,
        metric, value.
    """
    rows = []

    # K-Means: silhouette score from cluster profiles
    kmeans_labels_path = MODEL_OUTPUT_DIR / "kmeans_labeled_zips.csv"
    if kmeans_labels_path.exists():
        try:
            kmeans_df = pd.read_csv(kmeans_labels_path)
            n_clusters = kmeans_df["kmeans_cluster"].nunique()
            n_zips = len(kmeans_df)
            rows.append({
                "model": "K-Means",
                "model_type": "Unsupervised (clustering)",
                "metric": "n_clusters",
                "value": n_clusters,
            })
            rows.append({
                "model": "K-Means",
                "model_type": "Unsupervised (clustering)",
                "metric": "n_zip_codes",
                "value": n_zips,
            })
        except Exception as error:
            logger.warning(f"Failed to read K-Means labels: {error}")

    # DBSCAN: cluster count and noise count
    dbscan_labels_path = MODEL_OUTPUT_DIR / "dbscan_labeled_zips.csv"
    if dbscan_labels_path.exists():
        try:
            dbscan_df = pd.read_csv(dbscan_labels_path)
            labels = dbscan_df["dbscan_cluster"]
            n_clusters = len(set(labels) - {-1})
            n_noise = (labels == -1).sum()
            rows.append({
                "model": "DBSCAN",
                "model_type": "Unsupervised (density)",
                "metric": "n_clusters",
                "value": n_clusters,
            })
            rows.append({
                "model": "DBSCAN",
                "model_type": "Unsupervised (density)",
                "metric": "n_noise_points",
                "value": n_noise,
            })
        except Exception as error:
            logger.warning(f"Failed to read DBSCAN labels: {error}")

    # Random Forest: R², RMSE, MAE from both train and test sets
    rf_metrics_path = MODEL_OUTPUT_DIR / "rf_evaluation_metrics.csv"
    if rf_metrics_path.exists():
        try:
            rf_df = pd.read_csv(rf_metrics_path)
            for col in rf_df.columns:
                rows.append({
                    "model": "Random Forest",
                    "model_type": "Supervised (regression)",
                    "metric": col,
                    "value": rf_df[col].iloc[0],
                })
        except Exception as error:
            logger.warning(f"Failed to read RF metrics: {error}")

    return pd.DataFrame(rows)


def plot_evaluation_summary(summary_df):
    """
    Create a simple visual summary of key model metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Long-format evaluation summary.

    Returns
    -------
    None
    """
    if summary_df.empty:
        print("  No metrics to plot")
        return

    fig, ax = plt.subplots(figsize=(10, max(3, len(summary_df) * 0.3)))
    ax.axis("off")

    # Render as a clean text table
    display_df = summary_df.copy()
    display_df["value"] = display_df["value"].apply(
        lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    )

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title("Model Evaluation Summary",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VISUALIZATIONS_DIR / "evaluation_summary.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: evaluation_summary.png")


def main():
    """
    Entry point for the evaluation summary stage.


    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting evaluation summary")

    print("\nCollecting metrics from all models ...")
    try:
        summary = collect_model_metrics()
    except Exception as error:
        logger.error(f"Failed to collect model metrics: {error}",
                     exc_info=True)
        print(f"  ❌ Failed to collect metrics: {error}")
        sys.exit(1)

    if summary.empty:
        logger.warning("No model evaluation outputs found")
        print("  ⚠ No evaluation outputs found.")
        print("  Run the model scripts first.")
        sys.exit(0)

    print(f"  Collected {len(summary)} metrics across models")

    # Save the consolidated summary
    output_path = MODEL_OUTPUT_DIR / "evaluation_summary.csv"
    try:
        summary.to_csv(output_path, index=False)
    except (OSError, IOError) as error:
        logger.error(f"Failed to save summary: {error}", exc_info=True)
        print(f"  ❌ Failed to save summary: {error}")
        sys.exit(1)
    print(f"\n  Saved: {output_path.name}")

    # Render the visual summary
    plot_evaluation_summary(summary)

    logger.info("Evaluation summary complete")
    print("\nEvaluation summary complete.")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()