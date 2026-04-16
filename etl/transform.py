"""
transform.py - Data Transformation and Cleaning
================================================
Reads raw extracted datasets from data/extracted/, cleans and
standardizes them, performs exploratory data analysis to inform
cleaning decisions, and saves the transformed datasets to
data/transformed/.

EDA is included in this stage per course requirements, as it helps
guide transformation decisions and produces reusable assets for
the testing phase.

Usage:
    python etl/transform.py
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path so logging_config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from logging_config import get_logger

logger = get_logger(__name__)


# Path setup - this file is in etl/, so .parent.parent reaches
# the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"
TRANSFORMED_DIR = PROJECT_ROOT / "data" / "transformed"
VISUALIZATIONS_DIR = PROJECT_ROOT / "data" / "visualizations"

# Month name to number mapping for parsing the foreclosure
# dataset's column names (e.g., 'july_2022' -> month 7)
MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ── Shared Utility Functions ────────────────────────────────────


def standardize_zip(series):
    """
    Coerce a column of zip codes to clean 5-digit strings.

    Handles common issues: zip codes stored as floats (e.g.,
    20774.0), as integers (dropping leading zeros), ZIP+4
    formats (e.g., '21201-1234'), and extra whitespace.

    Parameters
    ----------
    series : pd.Series
        Raw zip code column from any dataset.

    Returns
    -------
    pd.Series
        Cleaned zip codes as 5-digit strings. Invalid entries
        become NaN.
    """
    # Convert to string, strip whitespace, extract first 5 digits
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)  # handle float-style like '20774.0'
        .str.extract(r"(\d{5})", expand=False)
    )
    return cleaned


def month_year_to_quarter(month):
    """
    Convert a month number (1-12) to its calendar quarter (1-4).

    Parameters
    ----------
    month : int
        Month number where January=1, December=12.

    Returns
    -------
    int
        Quarter number (1-4).
    """
    return (month - 1) // 3 + 1


def find_column(dataframe, candidates):
    """
    Find the first matching column name from a list of candidates.

    Performs case-insensitive matching so that column names like
    'Zip_Code', 'zip_code', and 'ZIP_CODE' all match.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to search.
    candidates : list of str
        Column names to look for, in priority order.

    Returns
    -------
    str or None
        The actual column name found, or None if no match.
    """
    # Build a lookup of lowercase -> actual column name
    lowercase_columns = {col.lower(): col for col in dataframe.columns}
    for candidate in candidates:
        if candidate.lower() in lowercase_columns:
            return lowercase_columns[candidate.lower()]
    return None


# ── Exploratory Data Analysis ───────────────────────────────────


def run_eda(dataframe, dataset_name):
    """
    Run exploratory data analysis on a single dataset.

    Prints summary statistics and missing data report to the
    console, and saves distribution plots to data/visualizations/.
    This EDA informs cleaning decisions and produces assets
    reusable in the testing phase.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The cleaned dataset to analyze.
    dataset_name : str
        Human-readable name used for chart titles and filenames.

    Returns
    -------
    None
    """
    print(f"\n  --- EDA: {dataset_name} ---")

    # Summary statistics for all numeric columns
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        print(f"\n  Descriptive statistics:")
        print(dataframe[numeric_columns].describe().round(2).to_string())

    # Missing data report - shows how much cleaning is still needed
    print(f"\n  Missing values:")
    missing_count = dataframe.isnull().sum()
    missing_percent = (dataframe.isnull().mean() * 100).round(1)
    has_missing = False
    for col in dataframe.columns:
        if missing_count[col] > 0:
            print(f"    {col}: {missing_count[col]} ({missing_percent[col]}%)")
            has_missing = True
    if not has_missing:
        print("    No missing values found.")

    # Save distribution plots for numeric columns
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if len(numeric_columns) > 0:
        fig, axes = plt.subplots(1, len(numeric_columns),
                                  figsize=(5 * len(numeric_columns), 4))
        if len(numeric_columns) == 1:
            axes = [axes]

        for ax, col in zip(axes, numeric_columns):
            dataframe[col].dropna().hist(ax=ax, bins=30, edgecolor="white")
            ax.set_title(col.replace("_", " ").title())
            ax.set_ylabel("Count")

        fig.suptitle(f"Distributions: {dataset_name}", fontweight="bold")
        plt.tight_layout()

        # Save chart to data/visualizations/ for reuse
        chart_filename = f"eda_{dataset_name.lower().replace(' ', '_')}.png"
        fig.savefig(VISUALIZATIONS_DIR / chart_filename, dpi=150,
                    bbox_inches="tight")
        plt.close()
        print(f"  Saved distribution chart: {chart_filename}")


# ── Dataset-Specific Transformers ───────────────────────────────


def transform_foreclosures(filepath):
    """
    Clean and standardize the foreclosure notices dataset.

    The Socrata API returns this dataset in a wide format where
    each row is a zip code and each column is a month (e.g.,
    'july_2022') with the foreclosure notice count as the value.
    Because apparently giving us tidy data would have been too
    generous. This function melts the wide structure into a long
    format, parses month-year column names into proper dates, and
    aggregates to the zip-quarter level.

    Parameters
    ----------
    filepath : Path
        Path to the raw foreclosures CSV in data/extracted/.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns: zip_code, year, quarter,
        foreclosure_notices.
    """
    dataframe = pd.read_csv(filepath, dtype=str)
    dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(" ", "_")

    # Identify the zip code column
    zip_col = find_column(dataframe, ["zip_code", "zipcode", "zip"])

    # Standardize zip codes to 5-digit strings
    dataframe["zip_code"] = standardize_zip(dataframe[zip_col])

    # Identify which columns represent monthly foreclosure counts
    # The API returns two sets of monthly columns:
    #   - Timestamp-style: '_2022_07_21t00_00_00_000' (duplicates)
    #   - Human-readable:  'july_2022', 'august_2022', etc.
    # We use only the human-readable ones since they parse cleanly
    month_columns = []
    for col in dataframe.columns:
        # Match pattern like 'july_2022' or 'january_2023'
        match = re.match(r"^([a-z]+)_(\d{4})$", col)
        if match and match.group(1) in MONTH_MAP:
            month_columns.append(col)

    # Melt from wide to long: one row per zip code per month
    melted = dataframe[["zip_code"] + month_columns].melt(
        id_vars=["zip_code"],
        var_name="month_year",
        value_name="foreclosure_notices",
    )

    # Parse the month_year column name into actual year and month
    # e.g., 'july_2022' -> month=7, year=2022
    parsed = melted["month_year"].str.extract(r"^([a-z]+)_(\d{4})$")
    melted["month"] = parsed[0].map(MONTH_MAP)
    melted["year"] = pd.to_numeric(parsed[1])
    melted["quarter"] = melted["month"].apply(month_year_to_quarter)

    # Convert foreclosure counts to numeric (they come as strings)
    melted["foreclosure_notices"] = pd.to_numeric(
        melted["foreclosure_notices"], errors="coerce"
    )

    # Aggregate to zip code + quarter level
    aggregated = (
        melted.dropna(subset=["zip_code", "year", "quarter"])
        .groupby(["zip_code", "year", "quarter"], as_index=False)
        .agg(foreclosure_notices=("foreclosure_notices", "sum"))
    )
    aggregated["year"] = aggregated["year"].astype(int)
    aggregated["quarter"] = aggregated["quarter"].astype(int)

    print(f"  Foreclosures: {len(aggregated)} zip-quarter records")
    return aggregated


def transform_ev_registrations(filepath):
    """
    Clean and standardize the EV registration dataset.

    The API returns this dataset in a clean long format with
    columns: year_month (e.g., '2020/07'), fuel_category,
    zip_code, and count. This function parses the year_month
    into year and quarter, then aggregates counts.

    Parameters
    ----------
    filepath : Path
        Path to the raw EV registrations CSV in data/extracted/.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns: zip_code, year, quarter,
        ev_registrations.
    """
    dataframe = pd.read_csv(filepath, dtype=str)
    dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(" ", "_")

    # Standardize zip codes
    zip_col = find_column(dataframe, ["zip_code", "zipcode", "zip"])
    dataframe["zip_code"] = standardize_zip(dataframe[zip_col])

    # Parse year_month column (format: '2020/07') into year and quarter
    year_month_col = find_column(dataframe, ["year_month", "yearmonth"])
    if year_month_col:
        parsed = dataframe[year_month_col].str.extract(r"(\d{4})/(\d{2})")
        dataframe["year"] = pd.to_numeric(parsed[0], errors="coerce")
        month = pd.to_numeric(parsed[1], errors="coerce")
        dataframe["quarter"] = month.apply(
            lambda m: month_year_to_quarter(int(m)) if pd.notna(m) else np.nan
        )

    # Convert count to numeric
    count_col = find_column(dataframe, ["count", "registrations", "total",
                                         "ev_registrations", "vehicle_count"])
    if count_col:
        dataframe["ev_registrations"] = pd.to_numeric(
            dataframe[count_col], errors="coerce"
        )
    else:
        dataframe["ev_registrations"] = 1

    # Aggregate to zip code + quarter level, combining all
    # fuel categories (Electric and Plug-in Hybrid)
    aggregated = (
        dataframe.dropna(subset=["zip_code", "year", "quarter"])
        .groupby(["zip_code", "year", "quarter"], as_index=False)
        .agg(ev_registrations=("ev_registrations", "sum"))
    )
    aggregated["year"] = aggregated["year"].astype(int)
    aggregated["quarter"] = aggregated["quarter"].astype(int)

    print(f"  EV registrations: {len(aggregated)} zip-quarter records")
    return aggregated


def transform_sewer_overflows(filepath):
    """
    Clean and standardize the sewer overflow dataset.

    The API returns one row per overflow incident with a
    start_date timestamp, zip code (as a float), and various
    incident details. This function standardizes the zip codes,
    parses dates, and aggregates to the zip-quarter level.

    Parameters
    ----------
    filepath : Path
        Path to the raw sewer overflows CSV in data/extracted/.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns: zip_code, year, quarter,
        sewer_overflow_incidents.
    """
    dataframe = pd.read_csv(filepath, dtype=str)
    dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(" ", "_")

    # Standardize zip codes (stored as floats like '20774.0')
    zip_col = find_column(dataframe, ["zip_code", "zipcode", "zip"])
    dataframe["zip_code"] = standardize_zip(dataframe[zip_col])

    # Parse start_date into year and quarter
    # Format from API: '2026-03-11T00:00:00.000'
    date_col = find_column(dataframe, ["start_date", "discovery_date",
                                        "date", "report_date",
                                        "incident_date", "overflow_date"])
    if date_col:
        parsed_dates = pd.to_datetime(dataframe[date_col], errors="coerce")
        dataframe["year"] = parsed_dates.dt.year
        dataframe["quarter"] = parsed_dates.dt.quarter

    # Each row is one overflow incident
    dataframe["sewer_overflow_incidents"] = 1

    # Aggregate to zip code + quarter level
    aggregated = (
        dataframe.dropna(subset=["zip_code", "year", "quarter"])
        .groupby(["zip_code", "year", "quarter"], as_index=False)
        .agg(sewer_overflow_incidents=("sewer_overflow_incidents", "sum"))
    )
    aggregated["year"] = aggregated["year"].astype(int)
    aggregated["quarter"] = aggregated["quarter"].astype(int)

    print(f"  Sewer overflows: {len(aggregated)} zip-quarter records")
    return aggregated



def transform_waste_violations(filepath):
    """
    Clean and standardize the solid waste violations dataset.

    The API returns one row per violation, with zip codes embedded
    in a combined 'city, state zip' field (format: 'Curtis Bay,MD,21226')
    and dates as ISO timestamps from the API (e.g.,
    '2021-10-14T09:36:32.204') or DD/MM/YYYY from CSV downloads. A single site inspection can
    generate multiple violation rows, so we count unique violation
    dates per zip-quarter rather than raw rows to avoid inflating
    counts from multi-violation inspections.

    Parameters
    ----------
    filepath : Path
        Path to the raw waste violations CSV in data/extracted/.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with columns: zip_code, year, quarter,
        waste_violation_events.
    """
    dataframe = pd.read_csv(filepath, dtype=str)
    dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace(" ", "_")

    # Extract zip codes from the combined city/state/zip field
    # Format: 'Curtis Bay,MD,21226' or similar
    city_state_zip_col = find_column(dataframe, ["city,_state_zip",
                                                  "city_state_zip"])
    if city_state_zip_col:
        dataframe["zip_code"] = standardize_zip(dataframe[city_state_zip_col])
    else:
        # Fall back to a dedicated zip column if format changed
        zip_col = find_column(dataframe, ["zip_code", "zipcode", "zip"])
        if zip_col:
            dataframe["zip_code"] = standardize_zip(dataframe[zip_col])

    # Parse violation dates - the API returns ISO timestamps
    # (e.g., '2021-10-14T09:36:32.204') while the CSV download
    # uses DD/MM/YYYY format. We try general parsing first which
    # handles both, then fall back to specific formats.
    date_col = find_column(dataframe, ["violation_date", "date",
                                        "violation_dt"])
    if date_col:
        # General parsing handles ISO, YYYY-MM-DD, and most formats
        parsed_dates = pd.to_datetime(dataframe[date_col], errors="coerce")
        # If general parsing failed for most rows, try DD/MM/YYYY
        if parsed_dates.isna().sum() > len(dataframe) * 0.5:
            parsed_dates = pd.to_datetime(dataframe[date_col],
                                           format="%d/%m/%Y", errors="coerce")
        # If that also failed, try MM/DD/YYYY
        if parsed_dates.isna().sum() > len(dataframe) * 0.5:
            parsed_dates = pd.to_datetime(dataframe[date_col],
                                           format="%m/%d/%Y", errors="coerce")
        dataframe["year"] = parsed_dates.dt.year
        dataframe["quarter"] = parsed_dates.dt.quarter
        dataframe["violation_date_clean"] = parsed_dates.dt.date

    # Count unique violation dates per zip-quarter to control for
    # multi-violation inspections (one inspection of a single site
    # can generate 10+ violation rows)
    aggregated = (
        dataframe.dropna(subset=["zip_code", "year", "quarter"])
        .groupby(["zip_code", "year", "quarter"], as_index=False)
        .agg(waste_violation_events=("violation_date_clean", "nunique"))
    )
    aggregated["year"] = aggregated["year"].astype(int)
    aggregated["quarter"] = aggregated["quarter"].astype(int)

    print(f"  Waste violations: {len(aggregated)} zip-quarter records")
    return aggregated


# ── Main Entry Point ────────────────────────────────────────────


def transform_and_save(transformer_fn, input_name, output_name,
                        label):
    """
    Run one dataset's transform, EDA, and save with error handling.

    Wraps each individual transformation in a try/except so that
    one bad dataset does not block the others from processing.
    Errors are logged with full traceback for later debugging.

    Parameters
    ----------
    transformer_fn : callable
        The transform_* function to call for this dataset.
    input_name : str
        Filename of the raw CSV in data/extracted/.
    output_name : str
        Filename for the cleaned CSV in data/transformed/.
    label : str
        Human-readable name for console output and logs.

    Returns
    -------
    bool
        True if the transformation succeeded, False otherwise.
    """
    print(f"\nTransforming {label.lower()} data ...")
    logger.info(f"Starting transformation: {label}")

    try:
        dataframe = transformer_fn(EXTRACTED_DIR / input_name)
    except FileNotFoundError as error:
        logger.error(f"Raw file not found for {label}: {error}",
                     exc_info=True)
        print(f"  ❌ {label}: raw file not found")
        return False
    except Exception as error:
        logger.error(f"Transformation failed for {label}: {error}",
                     exc_info=True)
        print(f"  ❌ {label}: {error}")
        return False

    try:
        dataframe.to_csv(TRANSFORMED_DIR / output_name, index=False)
    except (OSError, IOError) as error:
        logger.error(f"Failed to save cleaned {label}: {error}",
                     exc_info=True)
        print(f"  ❌ {label}: save failed")
        return False
    
    try:
        run_eda(dataframe, label)
    except Exception as error:
        logger.warning(f"EDA failed for {label}: {error}", exc_info=True)
        print(f"  ⚠ EDA failed for {label}, continuing")

    logger.info(f"Completed transformation: {label} "
                f"({len(dataframe)} records)")
    return True


def main():
    """
    Entry point for the transformation pipeline.

    Reads each raw dataset from data/extracted/, transforms it,
    runs EDA, and saves the cleaned result to data/transformed/.
    Each dataset is processed independently so a failure in one
    does not block the others.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Ensure the output directory exists
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting transformation stage")

    # Each dataset is transformed independently so that a failure
    # in one does not prevent the others from being processed
    results = {
        "Foreclosures": transform_and_save(
            transform_foreclosures,
            "foreclosures_raw.csv",
            "foreclosures_clean.csv",
            "Foreclosures",
        ),
        "EV Registrations": transform_and_save(
            transform_ev_registrations,
            "ev_registrations_raw.csv",
            "ev_registrations_clean.csv",
            "EV Registrations",
        ),
        "Sewer Overflows": transform_and_save(
            transform_sewer_overflows,
            "sewer_overflows_raw.csv",
            "sewer_overflows_clean.csv",
            "Sewer Overflows",
        ),
        "Waste Violations": transform_and_save(
            transform_waste_violations,
            "waste_violations_raw.csv",
            "waste_violations_clean.csv",
            "Waste Violations",
        ),
    }

    failed = [name for name, success in results.items() if not success]
    if failed:
        logger.warning(f"Transformation completed with failures: {failed}")
        print(f"\n⚠ Failed to transform: {failed}")
        # Non-zero exit if everything failed, zero if partial success
        sys.exit(1 if len(failed) == len(results) else 0)

    print("\nTransformation complete. Cleaned files in data/transformed/")


# Ensures main() only runs when executed directly, not when imported
if __name__ == "__main__":
    main()