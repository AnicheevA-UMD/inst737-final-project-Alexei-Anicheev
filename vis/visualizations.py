"""
visualizations.py - Analytical Output Visualizations
=====================================================
Reads model outputs from data/model_outputs/ and the merged
analysis dataset from data/load/, then produces presentation-
quality visualizations of the project's findings.

Includes:
    - Lagged cross-correlation analysis (tests whether secondary
      indicators lead foreclosure changes over time)
    - Time series trends by K-Means cluster
    - Model comparison (K-Means vs DBSCAN classification)
    - Correlation heatmap between all indicators

Usage:
    python vis/visualizations.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


# Path setup - this file is in vis/, so .parent.parent reaches
# the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_DIR = PROJECT_ROOT / "data" / "load"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"


# ── Lagged Cross-Correlation ───────────────────────────────────


def plot_lagged_correlation(dataframe, max_lag=4):
    """
    Test whether secondary indicators lead or lag foreclosure
    changes by computing cross-correlations at different time
    offsets.

    This directly addresses the business problem: if sewer
    overflows in quarter t correlate with foreclosures in
    quarter t+2, that is a genuine early-warning signal.
    Negative lags mean the indicator leads foreclosures
    (the useful direction for prediction).

    Parameters
    ----------
    dataframe : pd.DataFrame
        Merged analysis dataset from data/load/.
    max_lag : int
        Maximum number of quarters to test in each direction.

    Returns
    -------
    None
    """
    # Identify secondary indicator columns (everything except
    # foreclosure_notices and non-indicator columns)
    indicator_columns = [col for col in dataframe.columns
                         if col not in ("record_id", "zip_code", "year",
                                        "quarter", "foreclosure_notices")]

    if not indicator_columns:
        print("  No secondary indicator columns found.")
        return

    # Aggregate to statewide quarterly totals for a cleaner signal
    # Individual zip codes are too noisy for lag analysis
    quarterly = (
        dataframe.groupby(["year", "quarter"])
        .agg({col: "sum" for col in ["foreclosure_notices"] + indicator_columns})
        .sort_index()
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(indicator_columns),
                              figsize=(6 * len(indicator_columns), 5))
    if len(indicator_columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, indicator_columns):
        lags = range(-max_lag, max_lag + 1)
        correlations = []

        for lag in lags:
            # Positive lag: foreclosures shifted forward (indicator leads)
            # Negative lag: foreclosures shifted backward (indicator lags)
            if lag >= 0:
                x = quarterly[col].iloc[:len(quarterly) - lag].values
                y = quarterly["foreclosure_notices"].iloc[lag:].values
            else:
                x = quarterly[col].iloc[-lag:].values
                y = quarterly["foreclosure_notices"].iloc[:len(quarterly) + lag].values

            # Compute correlation if enough valid data points
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() > 2:
                r, p_value = stats.pearsonr(x[mask], y[mask])
            else:
                r = np.nan
            correlations.append(r)

        # Color bars: blue when indicator leads, orange when it lags
        colors = ["steelblue" if lag <= 0 else "coral" for lag in lags]
        ax.bar(list(lags), correlations, color=colors, edgecolor="white")
        ax.set_xlabel("Lag (quarters)")
        ax.set_ylabel("Pearson r")
        ax.set_title(col.replace("_", " ").title(), fontsize=11)
        ax.axhline(0, color="black", linewidth=0.5)

        # Annotate the strongest leading correlation
        leading_lags = [i for i, lag in enumerate(lags) if lag < 0]
        if leading_lags:
            leading_corrs = [correlations[i] for i in leading_lags]
            best_idx = leading_lags[np.argmax(np.abs(leading_corrs))]
            best_lag = list(lags)[best_idx]
            best_r = correlations[best_idx]
            ax.annotate(f"r={best_r:.2f} at lag {best_lag}",
                        xy=(best_lag, best_r),
                        xytext=(best_lag, best_r + 0.1),
                        fontsize=9, fontweight="bold", color="red")

    fig.suptitle(
        "Lagged Cross-Correlation with Foreclosure Notices\n"
        "(Blue = indicator leads foreclosures, Orange = indicator lags)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "lagged_correlation.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: lagged_correlation.png")


# ── Time Series by Cluster ─────────────────────────────────────


def plot_cluster_time_series(dataframe, labeled_zips):
    """
    Show how each K-Means cluster's indicators evolve over time.

    Aggregates each indicator by cluster and quarter, revealing
    whether clusters diverge, converge, or follow parallel trends.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Merged analysis dataset from data/load/.
    labeled_zips : pd.DataFrame
        K-Means labeled zip codes from data/model_outputs/.

    Returns
    -------
    None
    """
    # Join cluster labels onto the time-series data
    cluster_map = labeled_zips[["zip_code", "kmeans_cluster"]].copy()
    cluster_map["zip_code"] = cluster_map["zip_code"].astype(str)
    dataframe["zip_code"] = dataframe["zip_code"].astype(str)
    merged = dataframe.merge(cluster_map, on="zip_code", how="inner")

    # Create a period label for the x-axis
    merged["period"] = (merged["year"].astype(str) + "-Q"
                        + merged["quarter"].astype(str))

    # Sort periods chronologically
    period_order = sorted(
        merged["period"].unique(),
        key=lambda p: (int(p.split("-Q")[0]), int(p.split("-Q")[1]))
    )

    # Identify indicator columns
    indicator_columns = [col for col in dataframe.columns
                         if col not in ("record_id", "zip_code", "year",
                                        "quarter")]

    fig, axes = plt.subplots(len(indicator_columns), 1,
                              figsize=(12, 4 * len(indicator_columns)),
                              sharex=True)
    if len(indicator_columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, indicator_columns):
        # Aggregate by cluster and period
        cluster_trends = (
            merged.groupby(["kmeans_cluster", "period"])[col]
            .mean()
            .reset_index()
        )

        # Plot each cluster as a separate line
        for cluster_id in sorted(merged["kmeans_cluster"].unique()):
            cluster_data = cluster_trends[
                cluster_trends["kmeans_cluster"] == cluster_id
            ]
            # Reindex to period order
            cluster_data = cluster_data.set_index("period").reindex(period_order)
            ax.plot(period_order, cluster_data[col].values,
                    marker="o", linewidth=2,
                    label=f"Cluster {cluster_id}")

        ax.set_ylabel(col.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Indicator Trends by K-Means Cluster",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "cluster_time_series.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_time_series.png")


# ── Correlation Heatmap ─────────────────────────────────────────


def plot_correlation_heatmap(dataframe):
    """
    Visualize Pearson and Spearman correlations between all
    numeric indicators.

    Spearman is particularly important because the relationships
    between indicators like sewer overflows and foreclosures are
    unlikely to be strictly linear.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Merged analysis dataset from data/load/.

    Returns
    -------
    None
    """
    # Select only indicator columns
    indicator_columns = [col for col in dataframe.columns
                         if col not in ("record_id", "zip_code", "year",
                                        "quarter")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, method in zip(axes, ["pearson", "spearman"]):
        corr = dataframe[indicator_columns].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    ax=ax, square=True, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{method.title()} Correlation", fontweight="bold")

    fig.suptitle("Correlations Between Indicators",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "correlation_heatmap.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: correlation_heatmap.png")


# ── Model Comparison ────────────────────────────────────────────


def plot_model_comparison(kmeans_zips, dbscan_zips):
    """
    Compare how K-Means and DBSCAN classified the same zip codes.

    Shows a cross-tabulation of cluster assignments and highlights
    which K-Means clusters contain DBSCAN's noise points.

    Parameters
    ----------
    kmeans_zips : pd.DataFrame
        K-Means labeled zip codes.
    dbscan_zips : pd.DataFrame
        DBSCAN labeled zip codes.

    Returns
    -------
    None
    """
    # Merge the two label sets on zip_code
    kmeans_zips["zip_code"] = kmeans_zips["zip_code"].astype(str)
    dbscan_zips["zip_code"] = dbscan_zips["zip_code"].astype(str)
    comparison = kmeans_zips[["zip_code", "kmeans_cluster"]].merge(
        dbscan_zips[["zip_code", "dbscan_cluster"]],
        on="zip_code", how="inner"
    )

    # Label DBSCAN noise as "Noise" for readability
    comparison["dbscan_label"] = comparison["dbscan_cluster"].apply(
        lambda x: "Noise" if x == -1 else f"Cluster {x}"
    )

    # Cross-tabulation
    crosstab = pd.crosstab(
        comparison["kmeans_cluster"],
        comparison["dbscan_label"],
        margins=True
    )

    print("\n  K-Means vs DBSCAN cross-tabulation:")
    print(crosstab.to_string())

    # Visualize: for each K-Means cluster, what fraction was
    # flagged as noise by DBSCAN?
    comparison["is_noise"] = (comparison["dbscan_cluster"] == -1).astype(int)
    noise_by_kmeans = comparison.groupby("kmeans_cluster")["is_noise"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    noise_by_kmeans.plot.bar(ax=ax, color="coral", edgecolor="white")
    ax.set_xlabel("K-Means Cluster")
    ax.set_ylabel("Fraction Flagged as Noise by DBSCAN")
    ax.set_title("DBSCAN Noise Points by K-Means Cluster",
                 fontweight="bold")
    ax.set_ylim(0, 1)

    # Add count labels on each bar
    for i, (cluster, fraction) in enumerate(noise_by_kmeans.items()):
        count = comparison[
            (comparison["kmeans_cluster"] == cluster) &
            (comparison["is_noise"] == 1)
        ].shape[0]
        total = comparison[comparison["kmeans_cluster"] == cluster].shape[0]
        ax.text(i, fraction + 0.02, f"{count}/{total}",
                ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "model_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: model_comparison.png")


# ── Main Entry Point ────────────────────────────────────────────


def main():
    """
    Entry point for the visualization pipeline.

    Loads model outputs and the merged dataset, then generates
    all analytical visualizations.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the merged analysis dataset
    print("\nLoading data ...")
    dataframe = pd.read_csv(LOAD_DIR / "merged_analysis.csv")
    print(f"  Merged dataset: {len(dataframe)} records")

    # Load model outputs
    kmeans_zips = pd.read_csv(MODEL_OUTPUT_DIR / "kmeans_labeled_zips.csv")
    dbscan_zips = pd.read_csv(MODEL_OUTPUT_DIR / "dbscan_labeled_zips.csv")
    print(f"  K-Means labels: {len(kmeans_zips)} zip codes")
    print(f"  DBSCAN labels:  {len(dbscan_zips)} zip codes")

    # Generate visualizations
    print("\nGenerating correlation heatmap ...")
    plot_correlation_heatmap(dataframe)

    print("\nGenerating lagged cross-correlation ...")
    plot_lagged_correlation(dataframe)

    print("\nGenerating cluster time series ...")
    plot_cluster_time_series(dataframe, kmeans_zips)

    print("\nGenerating model comparison ...")
    plot_model_comparison(kmeans_zips, dbscan_zips)

    print(f"\nVisualization complete. Charts in data/visualizations/")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()