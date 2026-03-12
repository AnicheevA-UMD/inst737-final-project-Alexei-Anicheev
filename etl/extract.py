"""
extract.py - Data Extraction from Maryland Open Data (Socrata API)
=================================================================
Connects to opendata.maryland.gov and downloads datasets using the
Socrata Open Data API (SODA). Each dataset is saved as a raw CSV
file in the data/extracted/ directory for downstream processing.

No API key is required for public datasets. An optional app token
can be provided to increase the rate limit.

Usage:
    python etl/extract.py                  # pulls all datasets
    python etl/extract.py --dataset foreclosures
    python etl/extract.py --token YOUR_APP_TOKEN
"""

import argparse
import os
import time
import pandas as pd
import requests
from pathlib import Path


# Dataset registry - central configuration for all datasets
# To add a new dataset, add an entry here and it will automatically
# flow through the extraction pipeline
DATASETS = {
    "foreclosures": {
        "name": "Maryland Notices of Intent to Foreclose by Zip Code",
        "endpoint": "ftsr-vapt",
        "filename": "foreclosures_raw.csv",
        "description": "Foreclosure notices aggregated by zip code and date",
    },
    "ev_registrations": {
        "name": "MDOT MVA Electric and Plug-in Hybrid Vehicle Registrations",
        "endpoint": "tugr-unu9",
        "filename": "ev_registrations_raw.csv",
        "description": "EV/PHEV registrations by zip code (2020+)",
    },
    "sewer_overflows": {
        "name": "Reported Sewer Overflows (2023+)",
        "endpoint": "stgj-u72u",
        "filename": "sewer_overflows_raw.csv",
        "description": "Reported sewer overflow incidents with location data",
    },
}

# Base URL template for Socrata API requests
# The {endpoint} placeholder gets replaced with each dataset's identifier
BASE_URL = "https://opendata.maryland.gov/resource/{endpoint}.json"

# Output directory for raw extracted data
# Path: this file is in etl/, so .parent goes to etl/ and
# .parent.parent goes to the project root, then into data/extracted/
EXTRACTED_DIR = Path(__file__).resolve().parent.parent / "data" / "extracted"


def fetch_dataset(endpoint, app_token=None, limit=50000, max_records=None):
    """
    Fetch a complete dataset from the Socrata API using pagination.

    Downloads data in pages of up to 50,000 records each, continuing
    until all records are retrieved or the max_records cap is reached.

    Parameters
    ----------
    endpoint : str
        Socrata 4x4 dataset identifier (e.g., 'ftsr-vapt').
    app_token : str or None
        Optional Socrata app token for higher rate limits.
    limit : int
        Maximum records per page (Socrata cap is 50,000).
    max_records : int or None
        Optional cap on total records fetched. Useful for testing.

    Returns
    -------
    pd.DataFrame
        Raw dataset as a pandas DataFrame.
    """
    url = BASE_URL.format(endpoint=endpoint)

    # Set up authentication header if a token was provided
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    all_rows = []
    total_fetched = 0
    offset = 0

    # Pagination loop - keep requesting pages until the API returns
    # an empty page (meaning we've downloaded everything)
    while True:
        # Socrata uses $offset and $limit for pagination, and $order
        # ensures consistent ordering across pages
        params = {"$limit": limit, "$offset": offset, "$order": ":id"}
        print(f"  Fetching offset={offset}, limit={limit} ...")

        # Attempt the API request with error handling for network issues
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            print(f"  WARNING: Request failed: {error}")
            if total_fetched > 0:
                print(f"  Returning {total_fetched} records fetched so far.")
                break
            raise

        batch = response.json()

        # Empty batch means we've reached the end of the dataset
        if not batch:
            break

        all_rows.extend(batch)
        total_fetched += len(batch)
        print(f"  Fetched {len(batch)} records (total: {total_fetched})")

        # If a record cap was set (for testing), stop once we hit it
        if max_records and total_fetched >= max_records:
            all_rows = all_rows[:max_records]
            break

        # Fewer records than the limit means this was the last page
        if len(batch) < limit:
            break

        offset += limit

        # Brief pause between requests to avoid overwhelming the API
        time.sleep(0.5)

    dataset = pd.DataFrame(all_rows)
    print(f"  DONE: {len(dataset)} total records retrieved")
    return dataset

def save_extracted(dataframe, filename):
    """
    Save a raw DataFrame to CSV in the data/extracted/ directory.

    Creates the output directory if it does not already exist.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The raw extracted data to save.
    filename : str
        Name for the output CSV file (e.g., 'foreclosures_raw.csv').

    Returns
    -------
    Path
        Full file path where the CSV was saved.
    """
    # Ensure the output directory exists before writing
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = EXTRACTED_DIR / filename
    dataframe.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")
    return output_path


def print_data_profile(dataframe):
    """
    Print a quick profile of a freshly extracted dataset.

    Shows shape, column names, and data types so the user can
    verify the extraction looks correct before moving on.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataset to profile.

    Returns
    -------
    None
    """
    print(f"\n  Shape: {dataframe.shape}")
    print(f"  Columns: {list(dataframe.columns)}")
    print(f"  Data types:\n{dataframe.dtypes.to_string()}\n")

def main():
    """
    Entry point for the extraction pipeline.

    Parses command-line arguments, then extracts and saves each
    requested dataset from the Maryland Open Data API.

    Parameters
    ----------
    None (reads from command-line arguments)

    Returns
    -------
    None
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Extract datasets from Maryland Open Data (Socrata API)"
    )
    parser.add_argument(
        "--dataset", choices=list(DATASETS.keys()),
        help="Pull a single dataset instead of all (default: all)"
    )
    parser.add_argument(
        "--token", default=os.environ.get("SOCRATA_APP_TOKEN"),
        help="Socrata app token (or set SOCRATA_APP_TOKEN env var)"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Cap records per dataset (useful for quick testing)"
    )
    args = parser.parse_args()

    # Determine which datasets to extract - either one specific
    # dataset or all of them
    targets = [args.dataset] if args.dataset else list(DATASETS.keys())

    # Loop through each target dataset and extract it
    for key in targets:
        dataset_info = DATASETS[key]
        print(f"\n{'=' * 60}")
        print(f"Extracting: {dataset_info['name']}")
        print(f"Endpoint:   {dataset_info['endpoint']}")
        print(f"{'=' * 60}")

        # Fetch the dataset from the API
        dataframe = fetch_dataset(
            endpoint=dataset_info["endpoint"],
            app_token=args.token,
            max_records=args.max_records,
        )

        # Save the raw data to CSV in data/extracted/
        save_extracted(dataframe, dataset_info["filename"])

        # Print a quick profile for verification
        print_data_profile(dataframe)

    print("\nExtraction complete.")


# Standard Python entry point guard - ensures main() only runs
# when this script is executed directly, not when imported as a module
if __name__ == "__main__":
    main()
