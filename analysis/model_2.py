"""
model_2.py - DBSCAN Clustering Analysis
=========================================
Reads the merged analysis-ready dataset from data/load/, engineers
per-zip-code features, and applies DBSCAN (Density-Based Spatial
Clustering of Applications with Noise) to discover density-based
groupings and identify outlier zip codes that don't fit any pattern.

DBSCAN complements K-Means (model_1.py) by not requiring a
pre-specified number of clusters and by explicitly flagging noise
points — zip codes with unusual indicator combinations that may
warrant individual investigation.

Includes pre-processing, modeling, and evaluation as required.

Usage:
    python analysis/model_2.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# Import shared pre-processing functions from model_1
# This avoids duplicating the feature engineering code and keeps
# both models working from the same feature definitions
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_1 import build_zip_features, preprocess_features

# Add project root to path so logging_config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logging_config import get_logger

logger = get_logger(__name__)


# Path setup - this file is in analysis/, so .parent.parent
# reaches the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_DIR = PROJECT_ROOT / "data" / "load"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"


# ── Modeling ────────────────────────────────────────────────────


def find_optimal_eps(X, min_samples=5):
    """
    Use the k-distance plot to find a good eps value for DBSCAN.

    Computes the distance from each point to its k-th nearest
    neighbor (where k = min_samples), sorts these distances, and
    plots them. The "elbow" in this curve suggests a natural eps
    threshold: distances below the elbow are within-cluster
    proximity, distances above are between-cluster gaps.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    min_samples : int
        The min_samples parameter that will be used for DBSCAN.
        The k-distance plot uses this same value for k.

    Returns
    -------
    float
        Suggested eps value based on the elbow location.
    """
    # Compute distance to the k-th nearest neighbor for each point
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Sort the k-th neighbor distances in ascending order
    k_distances = np.sort(distances[:, -1])

    # Plot the k-distance curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(k_distances)), k_distances, linewidth=2)
    ax.set_xlabel("Points (sorted by distance)")
    ax.set_ylabel(f"Distance to {min_samples}th Nearest Neighbor")
    ax.set_title("DBSCAN: k-Distance Plot for eps Selection",
                 fontweight="bold")

    # Estimate the elbow using the maximum distance from a
    # straight line connecting the first and last points of the
    # k-distance curve. This is more robust than simple max-diff,
    # which can be fooled by extreme outliers at the tail.
    n_points = len(k_distances)
    line_start = np.array([0, k_distances[0]])
    line_end = np.array([n_points - 1, k_distances[-1]])
    line_vector = line_end - line_start

    # For each point on the curve, compute perpendicular distance
    # to the straight line between first and last point
    distances_to_line = np.zeros(n_points)
    line_length = np.linalg.norm(line_vector)
    for i in range(n_points):
        point = np.array([i, k_distances[i]])
        # Perpendicular distance from point to line using 2D cross product
        diff = line_start - point
        cross = line_vector[0] * diff[1] - line_vector[1] * diff[0]
        distances_to_line[i] = abs(cross) / line_length

    elbow_index = np.argmax(distances_to_line)
    suggested_eps = k_distances[elbow_index]

    ax.axhline(suggested_eps, color="red", linestyle="--", alpha=0.7,
               label=f"Suggested eps = {suggested_eps:.2f}")
    ax.legend()
    plt.tight_layout()

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VISUALIZATIONS_DIR / "dbscan_kdistance.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Suggested eps: {suggested_eps:.2f}")
    print("  Saved: dbscan_kdistance.png")

    return suggested_eps


def run_dbscan(X, eps, min_samples=5):
    """
    Fit a DBSCAN model and return cluster labels.

    Points labeled -1 are noise (outliers that don't belong to
    any cluster). Unlike K-Means, the number of clusters is
    determined automatically by the algorithm.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    eps : float
        Maximum distance between two points to be considered
        neighbors.
    min_samples : int
        Minimum number of points required to form a dense region
        (i.e., a cluster core).

    Returns
    -------
    np.ndarray
        Cluster label for each zip code. -1 indicates noise.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    # Count clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\n  DBSCAN results:")
    print(f"    Clusters found: {n_clusters}")
    print(f"    Noise points:   {n_noise}")
    print(f"    Clustered:      {(labels != -1).sum()}")

    # Compute silhouette score (excluding noise points since
    # they don't belong to any cluster)
    if n_clusters > 1:
        clustered_mask = labels != -1
        score = silhouette_score(X[clustered_mask], labels[clustered_mask])
        print(f"    Silhouette (excl. noise): {score:.3f}")

    return labels


# ── Evaluation ──────────────────────────────────────────────────


def visualize_dbscan(X, labels, feature_names):
    """
    Create a 2D PCA projection of DBSCAN clusters with noise
    points highlighted.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    labels : np.ndarray
        Cluster labels from DBSCAN (-1 = noise).
    feature_names : list of str
        Names of the features for reference.

    Returns
    -------
    None
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    print(f"\n  PCA: PC1 explains {explained_variance[0]:.1%}, "
          f"PC2 explains {explained_variance[1]:.1%}")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot noise points first (in gray) so clusters draw on top
    noise_mask = labels == -1
    ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
               c="lightgray", marker="x", s=30, alpha=0.6,
               label=f"Noise ({noise_mask.sum()})")

    # Plot each cluster in a distinct color
    cluster_ids = sorted(set(labels) - {-1})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_ids), 1)))
    for cluster_id, color in zip(cluster_ids, colors):
        mask = labels == cluster_id
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[color], s=40, alpha=0.7, edgecolor="white",
                   label=f"Cluster {cluster_id} ({mask.sum()})")

    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
    ax.set_title("DBSCAN Clusters (PCA Projection)", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "dbscan_pca_clusters.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: dbscan_pca_clusters.png")


def profile_dbscan(features_clean, labels, feature_names):
    """
    Characterize each DBSCAN cluster and analyze noise points.

    For clusters, computes mean feature profiles (same as K-Means).
    For noise points, lists them individually since they are the
    outlier zip codes that may warrant special attention.

    Parameters
    ----------
    features_clean : pd.DataFrame
        Clean feature matrix with zip_code column.
    labels : np.ndarray
        Cluster labels from DBSCAN (-1 = noise).
    feature_names : list of str
        Names of the feature columns.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        - profiles: cluster profile means
        - noise_zips: DataFrame of noise zip codes and their features
    """
    dataframe = features_clean.copy()
    dataframe["cluster"] = labels

    # Profile each cluster (excluding noise)
    clustered = dataframe[dataframe["cluster"] != -1]
    cluster_ids = sorted(clustered["cluster"].unique())

    if len(cluster_ids) > 0:
        profiles = clustered.groupby("cluster")[feature_names].mean()

        # Heatmap of cluster profiles (only useful with 2+ clusters)
        if len(cluster_ids) > 1:
            z_profiles = profiles.apply(stats.zscore)
            fig, ax = plt.subplots(
                figsize=(max(10, len(feature_names) * 0.8),
                         max(4, len(profiles) * 1.2))
            )
            sns.heatmap(z_profiles, annot=True, fmt=".2f", cmap="RdYlGn_r",
                        center=0, ax=ax, linewidths=0.5)
            ax.set_title("DBSCAN Cluster Profiles (Z-Scored Feature Means)",
                         fontweight="bold")
            ax.set_ylabel("Cluster")
            plt.tight_layout()
            fig.savefig(VISUALIZATIONS_DIR / "dbscan_cluster_profiles.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("  Saved: dbscan_cluster_profiles.png")
        else:
            print("  Single cluster found — skipping profile heatmap")
            print("  (DBSCAN's value here is in identifying noise/outliers)")

        # Print cluster summaries
        print("\n  " + "=" * 50)
        print("  DBSCAN CLUSTER PROFILES")
        print("  " + "=" * 50)
        for cluster_id in cluster_ids:
            row = profiles.loc[cluster_id]
            zip_count = (labels == cluster_id).sum()
            print(f"\n  Cluster {cluster_id} ({zip_count} zip codes):")
            for feature in feature_names:
                print(f"    {feature}: {row[feature]:.2f}")
    else:
        profiles = pd.DataFrame()
        print("  No clusters found (all points classified as noise)")

    # Analyze noise points - these are the outlier zip codes
    noise_zips = dataframe[dataframe["cluster"] == -1].copy()
    if len(noise_zips) > 0:
        print(f"\n  NOISE POINTS ({len(noise_zips)} zip codes):")
        print("  These zip codes have unusual indicator combinations")
        print("  that don't fit any cluster pattern:")
        # Show first 10 noise zip codes with their key features
        for _, row in noise_zips.head(10).iterrows():
            print(f"\n    Zip {row['zip_code']}:")
            for feature in feature_names[:4]:
                print(f"      {feature}: {row[feature]:.2f}")

        # Create a comparison chart: noise points vs cluster mean
        if len(cluster_ids) > 0:
            cluster_mean = clustered[feature_names].mean()
            noise_mean = noise_zips[feature_names].mean()

            comparison = pd.DataFrame({
                "Main Cluster": cluster_mean,
                "Noise (Outliers)": noise_mean,
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            comparison.plot.barh(ax=ax)
            ax.set_title("DBSCAN: Noise Points vs Main Cluster",
                         fontweight="bold")
            ax.set_xlabel("Mean Feature Value")
            plt.tight_layout()
            fig.savefig(VISUALIZATIONS_DIR / "dbscan_noise_comparison.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            print("\n  Saved: dbscan_noise_comparison.png")

    return profiles, noise_zips


# ── Main Entry Point ────────────────────────────────────────────


def main():
    """
    Entry point for DBSCAN clustering analysis.

    Runs the full pipeline: load data, engineer features,
    preprocess, find optimal eps, fit DBSCAN, visualize,
    and profile clusters and noise points.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting DBSCAN analysis")

    # Load the merged analysis-ready dataset
    print("\nLoading merged data from data/load/ ...")
    try:
        dataframe = pd.read_csv(LOAD_DIR / "merged_analysis.csv")
    except FileNotFoundError as error:
        logger.error(f"Merged dataset not found: {error}", exc_info=True)
        print(f"\n  ❌ Merged dataset not found. Run etl/load.py first.")
        sys.exit(1)
    print(f"  Loaded {len(dataframe)} records")
    logger.info(f"Loaded {len(dataframe)} records for DBSCAN")

    try:
        # Pre-processing: build per-zip feature vectors
        print("\nBuilding per-zip-code features ...")
        features = build_zip_features(dataframe)

        print("\nStandardizing features ...")
        features_clean, X, feature_names = preprocess_features(features)

        # Modeling: find optimal eps and run DBSCAN
        print("\nFinding optimal eps ...")
        eps = find_optimal_eps(X, min_samples=5)
        logger.info(f"DBSCAN eps selected: {eps:.2f}")

        print("\nRunning DBSCAN ...")
        labels = run_dbscan(X, eps, min_samples=5)

        # Evaluation: visualize and profile clusters
        print("\nVisualizing clusters ...")
        visualize_dbscan(X, labels, feature_names)

        print("\nProfiling clusters ...")
        profiles, noise_zips = profile_dbscan(features_clean, labels,
                                               feature_names)
    except Exception as error:
        logger.error(f"DBSCAN analysis failed: {error}", exc_info=True)
        print(f"\n  ❌ DBSCAN analysis failed: {error}")
        sys.exit(1)

    # Save outputs to data/model_outputs/
    try:
        features_clean["dbscan_cluster"] = labels
        features_clean.to_csv(MODEL_OUTPUT_DIR / "dbscan_labeled_zips.csv",
                              index=False)
        if len(profiles) > 0:
            profiles.to_csv(MODEL_OUTPUT_DIR / "dbscan_cluster_profiles.csv")
        noise_zips.to_csv(MODEL_OUTPUT_DIR / "dbscan_noise_zips.csv",
                          index=False)
    except (OSError, IOError) as error:
        logger.error(f"Failed to save DBSCAN outputs: {error}",
                     exc_info=True)
        print(f"\n  ⚠ Failed to save outputs: {error}")

    print("\n  Saved: dbscan_labeled_zips.csv")
    print("  Saved: dbscan_cluster_profiles.csv")
    print("  Saved: dbscan_noise_zips.csv")

    logger.info("DBSCAN analysis complete")
    print("\nDBSCAN analysis complete.")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()