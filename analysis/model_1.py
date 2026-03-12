"""
model_1.py - K-Means Clustering Analysis
==========================================
Reads the merged analysis-ready dataset from data/load/, engineers
per-zip-code features, and applies K-Means clustering to discover
groupings of zip codes with similar indicator profiles. The goal
is to find non-obvious patterns linking secondary indicators
(EV registrations, sewer overflows) to foreclosure activity.

Includes pre-processing, modeling, and evaluation as required.

Usage:
    python analysis/model_1.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy import stats


# Path setup - this file is in analysis/, so .parent.parent
# reaches the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_DIR = PROJECT_ROOT / "data" / "load"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"


# ── Pre-Processing ──────────────────────────────────────────────


def build_zip_features(dataframe):
    """
    Aggregate time-series data into per-zip-code feature vectors.

    For each indicator, computes four summary statistics that
    capture different aspects of a zip code's profile:
        - mean:  average level of activity
        - std:   volatility (how much it fluctuates)
        - trend: direction of change over time (slope)
        - max:   peak value (captures stress events)

    Parameters
    ----------
    dataframe : pd.DataFrame
        Merged dataset from data/load/ with columns: zip_code,
        year, quarter, and indicator columns.

    Returns
    -------
    pd.DataFrame
        One row per zip code with summary features for each
        indicator.
    """
    # Identify the indicator columns (everything numeric except
    # zip_code, year, and quarter which are identifiers/time keys)
    indicator_columns = [col for col in dataframe.select_dtypes(
        include=[np.number]).columns
        if col not in ("zip_code", "year", "quarter")]

    def compute_trend(series):
        """
        Compute the linear trend (slope) of a series over time.

        A positive slope means the indicator is increasing over
        the observed quarters; negative means declining.

        Parameters
        ----------
        series : pd.Series
            Numeric values ordered chronologically.

        Returns
        -------
        float or NaN
            Slope of the linear fit. NaN if fewer than 3 points.
        """
        values = series.dropna().values
        if len(values) < 3:
            return np.nan
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, values
        )
        return slope

    # Build aggregation dictionary: four features per indicator
    aggregations = {}
    for col in indicator_columns:
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, "std")
        aggregations[f"{col}_max"] = (col, "max")
        aggregations[f"{col}_trend"] = (col, compute_trend)

    # Count how many quarters each zip code has data for
    aggregations["n_quarters"] = ("year", "count")

    features = dataframe.groupby("zip_code").agg(**aggregations).reset_index()

    # Fill NaN in std columns (zip codes with only 1 quarter
    # produce NaN for standard deviation; 0 is appropriate)
    std_columns = [col for col in features.columns if col.endswith("_std")]
    features[std_columns] = features[std_columns].fillna(0)

    print(f"  Feature matrix: {features.shape[0]} zip codes x "
          f"{features.shape[1] - 1} features")
    return features


def preprocess_features(features):
    """
    Clean and standardize the feature matrix for clustering.

    Drops zip codes with excessive missing data, imputes remaining
    NaN values with column medians, and applies z-score
    standardization so that all features contribute equally to
    distance calculations.

    Parameters
    ----------
    features : pd.DataFrame
        Raw feature matrix from build_zip_features().

    Returns
    -------
    tuple of (pd.DataFrame, np.ndarray, list of str)
        - features_clean: DataFrame with zip codes and clean values
        - X: standardized numpy array for clustering
        - feature_names: list of feature column names
    """
    feature_names = [col for col in features.columns if col != "zip_code"]

    # Drop zip codes missing more than half their features
    threshold = len(feature_names) * 0.5
    valid_mask = features[feature_names].notna().sum(axis=1) >= threshold
    features_clean = features[valid_mask].copy()
    dropped_count = len(features) - len(features_clean)
    if dropped_count > 0:
        print(f"  Dropped {dropped_count} zip codes with >50% missing features")

    # Impute remaining NaN with column medians
    features_clean[feature_names] = features_clean[feature_names].fillna(
        features_clean[feature_names].median()
    )

    # Standardize to z-scores so all features are on the same scale
    scaler = StandardScaler()
    X = scaler.fit_transform(features_clean[feature_names])

    print(f"  After preprocessing: {X.shape[0]} zip codes x "
          f"{X.shape[1]} features")
    return features_clean, X, feature_names


# ── Modeling ────────────────────────────────────────────────────


def find_optimal_k(X, k_range=range(2, 11)):
    """
    Determine the best number of clusters using the elbow method
    and silhouette analysis.

    The elbow method looks for the k where adding more clusters
    stops producing meaningful reductions in within-cluster
    variance. The silhouette score measures how well-separated
    clusters are (higher is better, range -1 to 1).

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    k_range : range
        Range of k values to evaluate (default: 2 to 10).

    Returns
    -------
    int
        The k with the highest silhouette score.
    """
    inertias = []
    silhouette_scores = []

    # Test each k and record both metrics
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    best_k = list(k_range)[np.argmax(silhouette_scores)]
    print(f"\n  Best k by silhouette: {best_k} "
          f"(score: {max(silhouette_scores):.3f})")

    # Plot both metrics side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, "bo-", linewidth=2)
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster SS)")
    ax1.set_title("Elbow Method")

    ax2.plot(list(k_range), silhouette_scores, "ro-", linewidth=2)
    ax2.axvline(best_k, color="green", linestyle="--", alpha=0.7,
                label=f"Best k = {best_k}")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Analysis")
    ax2.legend()

    fig.suptitle("K-Means: Optimal Cluster Selection",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VISUALIZATIONS_DIR / "kmeans_cluster_selection.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kmeans_cluster_selection.png")

    return best_k


def run_kmeans(X, k):
    """
    Fit a K-Means model and return cluster labels.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Cluster label for each zip code (0 to k-1).
    """
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"\n  K-Means (k={k}): silhouette score = {score:.3f}")
    return labels


# ── Evaluation ──────────────────────────────────────────────────


def visualize_clusters(X, labels, feature_names):
    """
    Create a 2D PCA projection of the clusters and a feature
    loadings chart showing which features drive the groupings.

    Parameters
    ----------
    X : np.ndarray
        Standardized feature matrix.
    labels : np.ndarray
        Cluster labels from K-Means.
    feature_names : list of str
        Names of the features for the loadings chart.

    Returns
    -------
    PCA
        Fitted PCA object (useful for interpreting components).
    """
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    print(f"\n  PCA: PC1 explains {explained_variance[0]:.1%}, "
          f"PC2 explains {explained_variance[1]:.1%}")

    # Scatter plot of clusters in PCA space
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels,
                         cmap="tab10", alpha=0.6, s=40, edgecolor="white")
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} variance)")
    ax.set_title("K-Means Clusters (PCA Projection)", fontweight="bold")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "kmeans_pca_clusters.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kmeans_pca_clusters.png")

    # Feature loadings - shows what each principal component
    # represents in terms of the original features
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=feature_names,
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    loadings.plot.barh(ax=ax)
    ax.set_title("PCA Feature Loadings", fontweight="bold")
    ax.set_xlabel("Loading")
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "kmeans_pca_loadings.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kmeans_pca_loadings.png")

    return pca


def profile_clusters(features_clean, labels, feature_names):
    """
    Characterize each cluster by its average feature values.

    This is the primary analytical output: it reveals which
    combinations of indicators define each cluster, and whether
    clusters with high foreclosure activity also show distinctive
    patterns in secondary indicators.

    Parameters
    ----------
    features_clean : pd.DataFrame
        Clean feature matrix with zip_code column.
    labels : np.ndarray
        Cluster labels from K-Means.
    feature_names : list of str
        Names of the feature columns.

    Returns
    -------
    pd.DataFrame
        Cluster profile means for each feature.
    """
    dataframe = features_clean.copy()
    dataframe["cluster"] = labels

    # Compute mean feature values per cluster
    profiles = dataframe.groupby("cluster")[feature_names].mean()

    # Z-score the profiles for visual comparison across features
    # that have very different scales
    z_profiles = profiles.apply(stats.zscore)

    # Heatmap showing which features are elevated or depressed
    # in each cluster relative to the overall mean
    fig, ax = plt.subplots(
        figsize=(max(10, len(feature_names) * 0.8),
                 max(4, len(profiles) * 1.2))
    )
    sns.heatmap(z_profiles, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("K-Means Cluster Profiles (Z-Scored Feature Means)",
                 fontweight="bold")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "kmeans_cluster_profiles.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: kmeans_cluster_profiles.png")

    # Print a readable summary of each cluster
    print("\n  " + "=" * 50)
    print("  CLUSTER PROFILES")
    print("  " + "=" * 50)
    for cluster_id in sorted(profiles.index):
        row = profiles.loc[cluster_id]
        zip_count = (labels == cluster_id).sum()
        print(f"\n  Cluster {cluster_id} ({zip_count} zip codes):")
        for feature in feature_names:
            print(f"    {feature}: {row[feature]:.2f}")

    return profiles


# ── Main Entry Point ────────────────────────────────────────────


def main():
    """
    Entry point for K-Means clustering analysis.

    Runs the full pipeline: load data, engineer features,
    preprocess, find optimal k, fit K-Means, visualize,
    and profile clusters.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the merged analysis-ready dataset
    print("\nLoading merged data from data/load/ ...")
    dataframe = pd.read_csv(LOAD_DIR / "merged_analysis.csv")
    print(f"  Loaded {len(dataframe)} records")

    # Pre-processing: build per-zip feature vectors
    print("\nBuilding per-zip-code features ...")
    features = build_zip_features(dataframe)

    print("\nStandardizing features ...")
    features_clean, X, feature_names = preprocess_features(features)

    # Modeling: find optimal k and run K-Means
    print("\nFinding optimal cluster count ...")
    best_k = find_optimal_k(X)

    print("\nRunning K-Means ...")
    labels = run_kmeans(X, best_k)

    # Evaluation: visualize and profile clusters
    print("\nVisualizing clusters ...")
    visualize_clusters(X, labels, feature_names)

    print("\nProfiling clusters ...")
    profiles = profile_clusters(features_clean, labels, feature_names)

    # Save outputs to data/model_outputs/
    features_clean["kmeans_cluster"] = labels
    features_clean.to_csv(MODEL_OUTPUT_DIR / "kmeans_labeled_zips.csv",
                          index=False)
    profiles.to_csv(MODEL_OUTPUT_DIR / "kmeans_cluster_profiles.csv")
    print("\n  Saved: kmeans_labeled_zips.csv")
    print("  Saved: kmeans_cluster_profiles.csv")

    print("\nK-Means analysis complete.")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()