# European Energy Market Analytics & Price Forecasting Engine

## Overview

This repository is a work-in-progress, production-grade time-series forecasting pipeline that predicts hourly day-ahead electricity prices (EUR/MWh) across European energy markets. Built for quantitative analysts, energy traders, and grid operators, the engine ingests real-time power grid metrics — such as total load demand and renewable generation by source — to forecast market-clearing spot prices. The system processes multi-year historical time-series data into an optimized local analytical cache, trains explainable gradient boosting models on engineered features, and serves predictions through an interactive scenario-simulation dashboard.

## Data Sources

Data is pulled directly from the **ENTSO-E Transparency Platform** REST API using the official Python client wrapper, [`entsoe-py`](https://github.com/EnergieID/entsoe-py).

**Why not OPSD?** While Open Power System Data (OPSD) was historically a popular aggregated research repository, it is no longer actively updated. Querying ENTSO-E directly ensures access to authoritative, live, hourly-updated transmission system data straight from European TSOs — including data recent enough to support a realistic train/test split.

**Zones covered (12 bidding zones):**

| Zone | Area | Country |
|------|------|---------|
| SE1 | Luleå | Sweden |
| SE2 | Sundsvall | Sweden |
| SE3 | Stockholm | Sweden |
| SE4 | Malmö | Sweden |
| NO1 | Oslo | Norway |
| NO2 | Kristiansand | Norway |
| NO3 | Trondheim | Norway |
| NO4 | Tromsø | Norway |
| NO5 | Bergen | Norway |
| DK1 | West | Denmark |
| DK2 | East | Denmark |
| FI | — | Finland |

**Historical range:** 2021-01-01 to present.

## Key Design Decisions

- **Python 3.12, after starting on 3.14** → Development began on the newest available Python release, but several core dependencies (`polars`, `duckdb`, `xgboost`, `shap`) didn't yet have stable prebuilt wheels for it, risking slow or broken source-compiled installs. Standardizing on 3.12, mature, stable, universally supported by the scientific Python ecosystem, traded a few months of "newest version" for a reliable, reproducible environment.
- **Hourly resolution as the canonical grain** → On October 1, 2025, Europe's entire day-ahead electricity market (Nord Pool included) switched from hourly to 15-minute pricing intervals across every bidding zone. Left as-is, this would silently change the data's granularity partway through the historical range, corrupting lag features and rolling averages that assume a consistent frequency. Every price series is resampled to hourly at the ingestion layer, averaging each quarter-hour group where applicable, trading a small amount of intra-hour detail in the recent period for one clean, consistent multi-year series.
- **Multi-country Nordic scope (12 zones)** → Bidding zones don't exist in isolation; cross-border transmission flows and regional weather patterns heavily influence localized prices. Modeling interconnected zones together supports capturing spatial feature interactions (e.g., a wind generation surplus in DK1 depressing prices in SE3).
- **`src/` layout package structure** → Enforces a clean separation between runnable scripts (`scripts/`) and reusable core library logic (`src/energy_forecaster/`), avoiding import-path ambiguity and enabling a proper editable install (`pip install -e .`) for development.
- **DuckDB + Parquet as the storage and query layer** → Analytical tables are stored as columnar `.parquet` files and queried via DuckDB, which runs in-process (no separate database server to install, configure, or manage, unlike Postgres) while still supporting multi-gigabyte aggregations, CTEs, and window functions directly over the files on disk.

## Data Quality Issues Found & Resolved

Working against a live, external data source surfaced several real data-quality issues, each tracked down from first symptom to root cause before being fixed.

### 1. Day-ahead price timestamp duplication at year boundaries

**Symptom:** Per-zone row counts in the `prices` table came in slightly higher than expected. A `GROUP BY zone, timestamp HAVING COUNT(*) > 1` check confirmed exactly two rows sharing the same zone and timestamp at `January 1st, 00:00:00` for every year boundary from 2022 onward (2021 has no earlier year to collide with).

**Investigation:** Comparing the two duplicate rows directly showed identical values at boundaries before the October 2025 resolution change, but genuinely different values afterward, the clue linking this back to the hourly → 15-minute transition above. Each year's pull requested data through `{year+1}-01-01 00:00`, inclusive.

**Root Cause:** The ENTSO-E API treats the query's `start`/`end` window as inclusive at both ends, so consecutive years' pulls both legitimately returned the shared boundary hour, once each.

**Fix:** Rather than deduplicate after the fact (which fails silently once both sides of a 15-minute-resolution boundary hold genuinely different values), each year's fetch window was kept intentionally generous, but the result is explicitly filtered to `timestamp.dt.year == year` immediately before writing to Parquet, removing the possibility of a boundary leak at the source.

### 2. Daylight Saving Time (DST) timestamp dtype collapse

**Symptom:** Applying the year-boundary fix above raised `AttributeError: Can only use .dt accessor with datetimelike values` when pulling a full calendar year, despite working correctly on every earlier single-day and single-week test.

**Investigation:** Inspecting `df["timestamp"].dtype` showed `object` instead of a proper timestamp type. `entsoe-py` returns timestamps as fixed UTC offsets (`+01:00` in winter, `+02:00` in summer for Stockholm) rather than a named timezone. A short test window never happened to cross a DST transition; a full year always does.

**Root Cause:** Pandas can only use its fast, vectorized `datetime64[ns, tz]` dtype when every value in a column shares one consistent UTC offset. A column spanning both `+01:00` and `+02:00` silently falls back to a generic `object` dtype, which lacks the `.dt` accessor.

**Fix:** Every timestamp column is explicitly reparsed with `pd.to_datetime(..., utc=True)` to force one consistent internal representation, then converted back with `.dt.tz_convert("Europe/Stockholm")`, necessary so calendar-based filtering (including the year-boundary fix above) reflects correct local-time semantics rather than UTC-shifted ones.

## Reproducing This Project

### 1. Clone the repository

```bash
git clone https://github.com/RoneiCO/european-energy-forecaster.git
cd european-energy-forecaster
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
```

### 4. Configure your ENTSO-E API token

Register for API access at the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) (approval typically takes a few business days), then create a `.env` file in the project root:

```
ENTSOE_API_KEY=your_token_here
```

### 5. Pull and build the dataset

```bash
python scripts/pull_historical_data.py
python scripts/build_dataset.py
```

This populates `data/raw/` (per-zone, per-year Parquet cache) and produces `data/processed/hourly_dataset.parquet`, the joined, feature-ready modeling table.

## Project Status

- [x] Phase 1: Data ingestion, DuckDB warehouse layer, hourly modeling dataset
- [ ] Phase 2: Feature engineering
- [ ] Phase 3: Model training & evaluation
- [ ] Phase 4: Testing, CI/CD, Docker
- [ ] Phase 5: Streamlit dashboard & deployment