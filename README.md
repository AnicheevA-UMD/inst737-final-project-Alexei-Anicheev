# Maryland Foreclosure Indicator Analysis

INST737 Data Science Techniques — Final Project  
Alexei Anicheev

## Project Overview

Business problem: 
The core question: can indirect non-financial indicators predict foreclosure waves?

Who would use this: real estate agencies, non-profits, home sellers, low-income outreach organizations, government agencies attempting to inform people of resources they might not be aware of, specific agencies and entities involved in the services comprising the indicators.

Which secondary indicators might work where conventional ones are already well-trodden and which won't.

Can these be used to make useful predictions?

Datasets: 

foreclosures dataset - number of foreclosures in specific zipcodes by year and month. Wide structure, requires melt to long structure to function.

EV_registrations dataset - number of EVs registered by month in specific zipcodes

sewer_overflows dataset - number of sewer overflow incidents by month and year in specific zip codes.

Final merged dataset - keyed on (zip_code, year, quarter) with a composite unique ID, containing 6,780 records across 565 Maryland zip codes.

All initial datasets obtained from Open Data Maryland website using Socrata API to extract them. They have different year coverage, so the merged dataset that is used for analysis is narrowed down to the years where there is overlap - 2023 to 2025. 

Additionally, the EV dataset has an issue where there are non-Maryland zip codes included in the set and those are being filtered out. I need to do further investigation for why this is happening. 

Additional datasets will be added in part 3 as the analysis so far is preliminary and additional datasets along with additional analytical models would improve things.

Techniques:

ETL pipeline with automated API extraction - no Socrata token is needed, relatively clean and not many issues.

EDA during transformation stage - visualizations of distributions to confirm that transformation is successful

Two unsupervised models: 
K-Means (pattern grouping) - intended to discover zip code groupings
DBSCAN (outlier detection) - intended to discover the "main body" of zip codes and identify the outliers where data is outside the norm (will likely end up being zip codes with bigger populations)

Additional visualizations and models: Lagged cross-correlation to test predictive timing and Random Forest planned for Parts 3–4
Additional analysis based on further datasets that will be uncovered.


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/inst737-final-project-alexei-anicheev.git
cd inst737-final-project-alexei-anicheev
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

Run the full pipeline:
```bash
python main.py
```

Skip the API extraction stage (use cached data):
```bash
python main.py --skip-extract
```

Individual stages can also be run standalone:
```bash
python etl/extract.py
python etl/transform.py
python etl/load.py
python analysis/model_1.py
python analysis/model_2.py
python vis/visualizations.py
```

## Code Package Structure

```
inst737-final-project-alexei-anicheev/
├── data/
│   ├── extracted/          — Raw API pulls (CSVs)
│   ├── transformed/        — Cleaned individual datasets
│   ├── load/               — Merged analysis-ready dataset
│   ├── model_outputs/      — Cluster labels, profiles, noise zips
│   ├── visualizations/     — All charts and plots
│   └── reference-tables/   — Data dictionaries, reference CSVs
├── etl/
│   ├── extract.py          — Socrata API data extraction
│   ├── transform.py        — Cleaning, standardization, EDA
│   └── load.py             — Dataset merging, unique IDs
├── analysis/
│   ├── model_1.py          — K-Means clustering
│   ├── model_2.py          — DBSCAN clustering
│   └── model_3.py          — Random Forest (scaffolding, Parts 3-4)
├── vis/
│   └── visualizations.py   — Analytical output charts
├── .vscode/
│   └── settings.json       — VS Code Python interpreter config
├── main.py                 — Pipeline entry point
├── README.md
└── requirements.txt
```