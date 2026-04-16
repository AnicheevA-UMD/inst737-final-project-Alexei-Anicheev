# Maryland Foreclosure Indicator Analysis

INST737 Data Science Techniques — Final Project  
Alexei Anicheev

## Project Overview

<!-- TODO: Write this section — see notes in Part 2 document -->

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/AnicheevA-UMD/inst737-final-project-Alexei-Anicheev.git
cd inst737-final-project-Alexei-Anicheev
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

Run the full pipeline (extracts live data from Maryland Open Data API):
```bash
python main.py
```

Skip the API extraction stage (use cached data in data/extracted/):
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
python analysis/model_3.py
python analysis/evaluation_summary.py
python vis/visualizations.py
```

Pipeline logs are written to `logs/pipeline.log`.

## Code Package Structure

```
inst737-final-project-alexei-anicheev/
├── data/
│   ├── extracted/          — Raw API pulls (CSVs)
│   ├── transformed/        — Cleaned individual datasets + EDA charts
│   ├── load/               — Merged analysis-ready dataset
│   ├── model_outputs/      — Cluster labels, profiles, RF metrics
│   ├── visualizations/     — All analytical charts and plots
│   └── reference-tables/   — Data dictionaries, reference CSVs
├── etl/
│   ├── extract.py          — Socrata API data extraction
│   ├── transform.py        — Cleaning, standardization, EDA
│   └── load.py             — Dataset merging, zip filtering, unique IDs
├── analysis/
│   ├── model_1.py          — K-Means clustering
│   ├── model_2.py          — DBSCAN clustering
│   ├── model_3.py          — Random Forest regression
│   └── evaluation_summary.py — Consolidated model evaluation metrics
├── vis/
│   └── visualizations.py   — All analytical output charts (clustering + RF)
├── logs/
│   └── pipeline.log        — Runtime log (auto-generated)
├── .vscode/
│   └── settings.json       — VS Code Python interpreter config
├── logging_config.py       — Shared logging configuration
├── main.py                 — Pipeline entry point
├── .gitignore
├── README.md
└── requirements.txt
```

## Datasets

All sourced from [opendata.maryland.gov](https://opendata.maryland.gov) via Socrata API (no token required):

Dataset | Endpoint | Role |

| Foreclosure Notices by Zip Code | ftsr-vapt | Primary target |
| EV/Plug-in Hybrid Registrations | tugr-unu9 | Secondary indicator |
| Reported Sewer Overflows | stgj-u72u | Secondary indicator |
| Solid Waste Program Violations | tzjz-wfys | Secondary indicator (Part 3) |

Analysis window: 2023–2025 (overlapping coverage across all datasets).
Merged dataset: 6,780 records across 565 Maryland zip codes, keyed on (zip_code, year, quarter) with composite unique IDs.

## Models

| Model | Type | Purpose |

| K-Means (model_1.py) | Unsupervised clustering | Groups zip codes by indicator profiles |
| DBSCAN (model_2.py) | Unsupervised density-based | Identifies outlier zip codes |
| Random Forest (model_3.py) | Supervised regression | Predicts foreclosures, ranks feature importance |

## Logging and Error Handling

All pipeline stages log to `logs/pipeline.log` via a shared logger configured in `logging_config.py`.