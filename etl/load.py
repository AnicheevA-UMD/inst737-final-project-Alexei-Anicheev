"""
load.py - Data Loading and Merging
===================================
Reads cleaned datasets from data/transformed/, merges them into
a single analysis-ready dataset keyed on (zip_code, year, quarter),
and saves the result to data/load/.

This is the final stage of the ETL pipeline. The output in
data/load/ is what the analysis and visualization stages consume.

Usage:
    python etl/load.py
"""

import pandas as pd
from pathlib import Path


# Path setup - this file is in etl/, so .parent.parent reaches
# the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSFORMED_DIR = PROJECT_ROOT / "data" / "transformed"
LOAD_DIR = PROJECT_ROOT / "data" / "load"


def merge_datasets(foreclosures, ev_registrations, sewer_overflows):
    """
    Merge all cleaned datasets into one analysis-ready DataFrame.

    Uses an outer join on (zip_code, year, quarter) so that all
    zip-quarter combinations are preserved even if only one dataset
    has data for that period. After merging, NaN values in count
    columns are filled with 0 since no record means zero activity.

    Parameters
    ----------
    foreclosures : pd.DataFrame
        Cleaned foreclosure notices with columns: zip_code, year,
        quarter, foreclosure_notices.
    ev_registrations : pd.DataFrame
        Cleaned EV registrations with columns: zip_code, year,
        quarter, ev_registrations.
    sewer_overflows : pd.DataFrame
        Cleaned sewer overflows with columns: zip_code, year,
        quarter, sewer_overflow_incidents.

    Returns
    -------
    pd.DataFrame
        Merged dataset containing all indicator columns. Count
        columns are filled with 0 where no data existed.
    """
    # Define the columns that all three datasets share
    join_keys = ["zip_code", "year", "quarter"]

    # Outer join: keeps all zip-quarter combinations from every
    # dataset, filling NaN where one dataset has no matching row
    merged = foreclosures.merge(ev_registrations, on=join_keys, how="outer")
    merged = merged.merge(sewer_overflows, on=join_keys, how="outer")

    # Fill NaN with 0 for count-based indicator columns
    # This is appropriate because all indicators are counts:
    # no record means zero activity, not unknown activity
    count_columns = ["foreclosure_notices", "ev_registrations",
                     "sewer_overflow_incidents"]
    for col in count_columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    # Sort for readability: by zip code, then chronologically
    merged.sort_values(join_keys, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged


def print_merge_summary(merged):
    """
    Print a summary of the merged dataset for verification.

    Shows shape, date range, unique zip codes, and missing
    value counts to help confirm the merge was successful.

    Parameters
    ----------
    merged : pd.DataFrame
        The merged analysis-ready dataset.

    Returns
    -------
    None
    """
    print(f"\n  Merged dataset shape: {merged.shape}")
    print(f"  Unique zip codes:    {merged['zip_code'].nunique()}")
    print(f"  Year range:          {merged['year'].min()} - {merged['year'].max()}")

    # Show how many NaN values each indicator column has
    # NaN is expected since datasets cover different time ranges
    print(f"\n  Missing values per column:")
    for col in merged.columns:
        null_count = merged[col].isnull().sum()
        if null_count > 0:
            null_pct = (null_count / len(merged) * 100)
            print(f"    {col}: {null_count} ({null_pct:.1f}%)")


def main():
    """
    Entry point for the load stage.

    Reads each cleaned dataset from data/transformed/, merges
    them into a single analysis-ready file, and saves it to
    data/load/.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    LOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Read each cleaned dataset from data/transformed/
    print("\nLoading cleaned datasets ...")
    foreclosures = pd.read_csv(TRANSFORMED_DIR / "foreclosures_clean.csv")
    print(f"  Foreclosures:     {len(foreclosures)} records")

    ev_registrations = pd.read_csv(TRANSFORMED_DIR / "ev_registrations_clean.csv")
    print(f"  EV registrations: {len(ev_registrations)} records")

    sewer_overflows = pd.read_csv(TRANSFORMED_DIR / "sewer_overflows_clean.csv")
    print(f"  Sewer overflows:  {len(sewer_overflows)} records")

    # Merge all datasets on (zip_code, year, quarter)
    print("\nMerging datasets ...")
    merged = merge_datasets(foreclosures, ev_registrations, sewer_overflows)

    # Save the analysis-ready dataset to data/load/
    output_path = LOAD_DIR / "merged_analysis.csv"
    merged.to_csv(output_path, index=False)
    print(f"\n  Saved merged dataset to {output_path}")

    # Print summary for verification
    print_merge_summary(merged)

    print("\nLoad complete.")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()