"""
model_3.py - Random Forest Regression
=======================================
Reads the merged analysis-ready dataset from data/load/ and uses
supervised learning to predict foreclosure notice counts from
secondary indicators (EV registrations, sewer overflows, waste
violations).

Unlike models 1 and 2 (unsupervised clustering), this model
directly answers: "can we predict foreclosures from these
indicators?" The feature importance ranking reveals which
indicators matter most for prediction.

Uses a temporal train/test split (train on 2023-2024, test on
2025) to simulate real-world prediction where you train on
historical data and predict the future.

Includes pre-processing, modeling, and evaluation as required.

Usage:
    python analysis/model_3.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Path setup - this file is in analysis/, so .parent.parent
# reaches the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOAD_DIR = PROJECT_ROOT / "data" / "load"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "model_outputs"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"


# ── Pre-Processing ──────────────────────────────────────────────


def prepare_supervised_data(dataframe):
    """
    Prepare feature matrix (X) and target vector (y) for
    supervised learning with a temporal train/test split.

    Uses a temporal split rather than random split because
    random splitting would leak future data into training,
    inflating performance metrics. Training on 2023-2024 and
    testing on 2025 simulates real-world prediction.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Merged analysis dataset from data/load/.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
              list of str)
        X_train, X_test, y_train, y_test, feature_names
    """
    # Define features and target
    target_column = "foreclosure_notices"
    feature_columns = [col for col in dataframe.columns
                       if col not in ("record_id", "zip_code", "year",
                                      "quarter", target_column)]

    print(f"  Target: {target_column}")
    print(f"  Features: {feature_columns}")

    X = dataframe[feature_columns]
    y = dataframe[target_column]

    # Temporal split: train on 2023-2024, test on 2025
    train_mask = dataframe["year"] <= 2024
    X_train = X[train_mask]
    X_test = X[~train_mask]
    y_train = y[train_mask]
    y_test = y[~train_mask]

    print(f"\n  Temporal split:")
    print(f"    Train: {len(X_train)} records (2023-2024)")
    print(f"    Test:  {len(X_test)} records (2025)")

    return X_train, X_test, y_train, y_test, feature_columns


# ── Modeling ────────────────────────────────────────────────────


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest regressor with cross-validation.

    Random Forest is chosen because it handles non-linear
    relationships well, is robust to feature scaling, and
    provides built-in feature importance rankings.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target (foreclosure notices).

    Returns
    -------
    RandomForestRegressor
        Fitted model.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,  # use all CPU cores for training
    )

    # Cross-validation on training data to assess stability
    cv_scores = cross_val_score(model, X_train, y_train,
                                 cv=5, scoring="neg_mean_squared_error")
    cv_rmse = np.sqrt(-cv_scores)
    print(f"\n  5-Fold Cross-Validation RMSE:")
    print(f"    Mean: {cv_rmse.mean():.2f}")
    print(f"    Std:  {cv_rmse.std():.2f}")
    print(f"    Range: {cv_rmse.min():.2f} - {cv_rmse.max():.2f}")

    # Fit the final model on all training data
    model.fit(X_train, y_train)
    print(f"\n  Model trained on {len(X_train)} records")

    return model


# ── Evaluation ──────────────────────────────────────────────────


def evaluate_model(model, X_train, y_train, X_test, y_test,
                   feature_names):
    """
    Evaluate Random Forest performance on both train and test sets,
    then visualize feature importances and prediction quality.

    Parameters
    ----------
    model : RandomForestRegressor
        Fitted model.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        True training values.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True test values.
    feature_names : list of str
        Names of the feature columns.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    # Generate predictions for both sets
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Compute metrics for both train and test
    metrics = {
        "train_r2": r2_score(y_train, train_predictions),
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_predictions)),
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "test_r2": r2_score(y_test, test_predictions),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_predictions)),
        "test_mae": mean_absolute_error(y_test, test_predictions),
    }

    print("\n  " + "=" * 50)
    print("  MODEL EVALUATION")
    print("  " + "=" * 50)
    print(f"\n  Training Set (2023-2024):")
    print(f"    R²:   {metrics['train_r2']:.4f}")
    print(f"    RMSE: {metrics['train_rmse']:.2f}")
    print(f"    MAE:  {metrics['train_mae']:.2f}")
    print(f"\n  Test Set (2025):")
    print(f"    R²:   {metrics['test_r2']:.4f}")
    print(f"    RMSE: {metrics['test_rmse']:.2f}")
    print(f"    MAE:  {metrics['test_mae']:.2f}")

    # Check for overfitting: large gap between train and test R²
    r2_gap = metrics["train_r2"] - metrics["test_r2"]
    if r2_gap > 0.15:
        print(f"\n  ⚠ Possible overfitting: R² gap = {r2_gap:.3f}")
        print(f"    Consider reducing max_depth or increasing min_samples_leaf")

    # ── Feature Importance Plot ──
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.5)))
    importances.plot.barh(ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Random Forest: Feature Importance",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance (Mean Decrease in Impurity)")
    plt.tight_layout()
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VISUALIZATIONS_DIR / "rf_feature_importance.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: rf_feature_importance.png")

    # ── Actual vs Predicted Scatter Plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training set
    ax1.scatter(y_train, train_predictions, alpha=0.3, s=20,
                edgecolor="white")
    max_val = max(y_train.max(), train_predictions.max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2,
             label="Perfect prediction")
    ax1.set_xlabel("Actual Foreclosure Notices")
    ax1.set_ylabel("Predicted Foreclosure Notices")
    ax1.set_title(f"Training Set (R² = {metrics['train_r2']:.3f})")
    ax1.legend()

    # Test set
    ax2.scatter(y_test, test_predictions, alpha=0.3, s=20,
                edgecolor="white", color="coral")
    max_val = max(y_test.max(), test_predictions.max())
    ax2.plot([0, max_val], [0, max_val], "r--", linewidth=2,
             label="Perfect prediction")
    ax2.set_xlabel("Actual Foreclosure Notices")
    ax2.set_ylabel("Predicted Foreclosure Notices")
    ax2.set_title(f"Test Set (R² = {metrics['test_r2']:.3f})")
    ax2.legend()

    fig.suptitle("Random Forest: Actual vs Predicted",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "rf_actual_vs_predicted.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: rf_actual_vs_predicted.png")

    # ── Residuals Distribution ──
    test_residuals = y_test - test_predictions

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(test_residuals, bins=40, edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Random Forest: Test Set Residuals",
                 fontsize=13, fontweight="bold")
    median_residual = test_residuals.median()
    ax.axvline(median_residual, color="orange", linestyle="--",
               label=f"Median: {median_residual:.1f}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(VISUALIZATIONS_DIR / "rf_residuals.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: rf_residuals.png")

    return metrics


# ── Main Entry Point ────────────────────────────────────────────


def main():
    """
    Entry point for Random Forest regression analysis.

    Runs the full supervised learning pipeline: load data,
    prepare features, train model with cross-validation,
    evaluate on held-out test set, and visualize results.

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

    # Pre-processing: prepare features and temporal split
    print("\nPreparing supervised data ...")
    X_train, X_test, y_train, y_test, feature_names = \
        prepare_supervised_data(dataframe)

    # Modeling: train Random Forest with cross-validation
    print("\nTraining Random Forest ...")
    model = train_random_forest(X_train, y_train)

    # Evaluation: metrics, feature importance, visualizations
    print("\nEvaluating model ...")
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test,
                             feature_names)

    # Save outputs to data/model_outputs/
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(MODEL_OUTPUT_DIR / "rf_evaluation_metrics.csv",
                      index=False)
    print("\n  Saved: rf_evaluation_metrics.csv")

    # Save feature importances
    importances_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances_df.to_csv(MODEL_OUTPUT_DIR / "rf_feature_importances.csv",
                          index=False)
    print("  Saved: rf_feature_importances.csv")

    print("\nRandom Forest analysis complete.")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()