# European Energy Market Analytics & Price Forecasting Engine

![CI](https://github.com/RoneiCO/european-energy-forecaster/actions/workflows/ci.yml/badge.svg)

## Overview

This repository is a work-in-progress, production-grade time-series forecasting pipeline that predicts hourly day-ahead electricity prices (EUR/MWh) across European energy markets. Built for quantitative analysts, energy traders, and grid operators, the engine ingests real grid data (day-ahead prices, load, generation by source, and the grid operators' own day-ahead forecasts of load and wind/solar output) to forecast market-clearing spot prices. The system processes multi-year historical time-series data into an optimized local analytical cache, builds forecast-safe features, trains explainable gradient boosting models, and serves predictions through an interactive scenario-simulation dashboard.

## Data Sources

Data is pulled directly from the **ENTSO-E Transparency Platform** REST API using the official Python client wrapper, [`entsoe-py`](https://github.com/EnergieID/entsoe-py).

**Why not OPSD?** While Open Power System Data (OPSD) was historically a popular aggregated research repository, it is no longer actively updated. Querying ENTSO-E directly ensures access to authoritative, live, hourly-updated transmission system data straight from European TSOs — including data recent enough to support a realistic train/test split.

**Datasets per zone and hour:** day-ahead price (EUR/MWh), actual load, actual generation by source, and the TSOs' day-ahead forecasts of load and of wind/solar generation.

**Zones covered (12 bidding zones):**

| Zone | Area | Country | Time zone |
|------|------|---------|-----------|
| SE1 | Luleå | Sweden | Europe/Stockholm |
| SE2 | Sundsvall | Sweden | Europe/Stockholm |
| SE3 | Stockholm | Sweden | Europe/Stockholm |
| SE4 | Malmö | Sweden | Europe/Stockholm |
| NO1 | Oslo | Norway | Europe/Oslo |
| NO2 | Kristiansand | Norway | Europe/Oslo |
| NO3 | Trondheim | Norway | Europe/Oslo |
| NO4 | Tromsø | Norway | Europe/Oslo |
| NO5 | Bergen | Norway | Europe/Oslo |
| DK1 | West | Denmark | Europe/Copenhagen |
| DK2 | East | Denmark | Europe/Copenhagen |
| FI | — | Finland | Europe/Helsinki |

Zones, their countries and their time zones live in one place, `src/energy_forecaster/config.py`, so the library can be pointed at other regions by editing a single dictionary.

**Historical range:** 2021-01-01 to present.

## Key Design Decisions

- **Python 3.12, after starting on 3.14** → Development began on the newest available Python release, but several core dependencies (`polars`, `duckdb`, `xgboost`, `shap`) didn't yet have stable prebuilt wheels for it, risking slow or broken source-compiled installs. Standardizing on 3.12 traded a few months of "newest version" for a reliable, reproducible environment.
- **Hourly resolution as the canonical grain** → On October 1, 2025, Europe's entire day-ahead electricity market (Nord Pool included) switched from hourly to 15-minute pricing intervals across every bidding zone. Left as-is, this would silently change the data's granularity partway through the historical range, corrupting lag features and rolling averages that assume a consistent frequency. Every series is resampled to hourly at the ingestion layer, averaging each quarter-hour group where applicable, trading a small amount of intra-hour detail in the recent period for one clean, consistent multi-year series.
- **Multi-country Nordic scope (12 zones)** → Bidding zones don't exist in isolation; cross-border transmission flows and regional weather patterns heavily influence localized prices. Modeling interconnected zones together supports capturing spatial feature interactions (e.g., a wind generation surplus in DK1 depressing prices in SE3).
- **`src/` layout package structure** → Enforces a clean separation between runnable scripts (`scripts/`) and reusable core library logic (`src/energy_forecaster/`), avoiding import-path ambiguity and enabling a proper editable install (`pip install -e .`) for development.
- **DuckDB + Parquet as the storage and query layer** → Analytical tables are stored as columnar `.parquet` files and queried via DuckDB, which runs in-process while still supporting multi-gigabyte aggregations, CTEs, and window functions directly over the files on disk.
- **Forecast-safe features only** → The day-ahead auction for day D closes at 12:00 on day D-1, so a feature is usable only if it would have existed at that moment. Price lags are at least 24 hours; lags of actual load and generation are at least 48 hours; and the wind, solar and load picture for the target day comes from the TSOs' own day-ahead forecasts, never from same-hour actuals. The joined `hourly_dataset.parquet` contains same-hour actuals (they are needed to build lags), so it is *not* safe to feed to a model directly, `build_feature_table` produces the safe table, and drops the raw generation columns once the lagged features are computed.
- **Calendar features on each zone's own clock** → Hour of day, weekday and public holidays are computed from each row's local time in its zone's time zone. Using a single clock would, for example, mark the first hour of Finland's Independence Day as an ordinary day.
- **Polars for feature engineering, pandas only at the scikit-learn boundary** → Per-zone lag and rolling features are expressed with Polars' `.over("zone")` (the equivalent of SQL's `PARTITION BY`); the data is converted to pandas only where scikit-learn requires it.
- **Preprocessing is defined now and fit later** → The `ColumnTransformer` (scaling, one-hot encoding) lives in `preprocessing.py`, but it is only ever fit on the training split, because fitting a scaler on data that includes the test period leaks the test period's distribution into the model.

- **Expanding-year cross-validation, with 2026 held out and scored once** → Folds train on all years before the evaluated year (2021–22 → 2023, 2021–23 → 2024, 2021–24 → 2025), mirroring how the model would be used. Model and feature decisions were made on those folds only; the 2026 test period was scored a single time at the end and nothing was changed afterwards.
- **Relative MAE against a "yesterday" baseline as the headline metric** → The baseline is the price of the same hour 24 hours earlier, the strongest of several naive baselines tried. 1.0 means "no better than yesterday". RMSE is reported alongside, because a few spikes dominate it.
- **Pre-registered predictions and adoption rules** → Each experiment had its expected result and its adoption rule written down before it was run, so the outcome could not bend the decision. Misses are reported with the hits.
- **Day-level bootstrap intervals** → Hourly errors within a day are strongly correlated, so uncertainty is estimated by resampling whole days.

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

### 3. Silently lost points in long price requests

**Symptom:** After joining all sources, the price column had 48 nulls (12 zones × 4 timestamps, all at `Dec 31 00:00` of 2021–2024) even though the market had priced those hours. Each null also blanked a week of the rolling price features built on top.

**Investigation:** Re-requesting the missing hour in a narrow window returned it. Padding each yearly request with extra days did not fix it; it moved the null from Dec 31 to Dec 16, exactly one calendar year (minus a day) after the padded request's start. Scanning request lengths from 30 to 400 days showed no loss up to 364 days, then exactly one missing point for every request spanning a full year, at the day before its one-year mark. In the 15-minute era the lost point is a single quarter-hour, so the hourly average silently uses 3 values instead of 4 and no null gives it away. Two earlier hypotheses were disproved by direct tests before this one survived.

**Root Cause:** A prices request that spans a full calendar year drops one data point at its one-year boundary. (Observed behavior of the request path; the exact layer, client or API, was not isolated.)

**Fix:** Prices are requested in 150-day windows, each padded by two days, then merged, de-duplicated and trimmed; and the fetch raises if any point is missing between two present points, at either cadence. After a full re-pull the null counts matched the predicted warm-up values exactly.

### 4. Public-holiday calendars that count Sundays

**Symptom:** The holiday flag marked about 61–63 days per year in Sweden against 10–12 in Norway and Denmark.

**Root Cause:** The `holidays` library's Swedish calendar treats every Sunday as a public holiday by default, which would have made the feature mean "special day" in some countries and "Sunday" in others.

**Fix:** The Swedish calendar is built with `include_sundays=False`, isolated in one commented function. Sweden then falls in line with the other countries.

### 5. Zero-filled gaps in renewable forecasts

**Symptom:** Runs of exactly 0 MW of forecast wind and solar, 12 hours or longer, in zones that otherwise have both, starting at midnight, lasting whole days, and shared by all zones of a country.

**Investigation:** Querying the API directly for sample days showed two origins: days the API had no data for (which ingestion had filled with zeros) and days where the grid operator itself published zeros. The two are indistinguishable in the stored data.

**Fix:** At feature time, such runs are treated as missing: 1,577 zone-hours (about 0.26% of rows) are set to null, while zones that are entirely zero (NO5 has no wind or solar in the data) are left alone. The count matched the sum of the listed runs exactly.

## Modeling & Results

The model forecasts the hourly price of the next day in all 12 zones using only information available before the 12:00 auction close, including the grid operators' own day-ahead forecasts for the target day.

### Model ladder (cross-validation, mean relative MAE over 2023, 2024 and 2025)

| Model | Relative MAE |
|---|---|
| "Yesterday" baseline | 1.000 |
| Ridge regression | 0.977 |
| XGBoost, base features | 0.801 |
| + cross-zone forecast aggregates | 0.756 |
| + German day-ahead market features | **0.706** |

German features improved every fold (0.048, 0.054 and 0.048), most in western Denmark (0.119). A random search over 13 XGBoost configurations found a best mean of 0.698, a gain of 0.008. That is below the pre-registered threshold of 0.01, so the default settings were kept.

### Held-out test, 1 January to 7 October 2026 (scored once)

| Model | MAE | RMSE | Relative MAE | 95% interval |
|---|---|---|---|---|
| "Yesterday" baseline | about 24.6 | | 1.000 | |
| Full model | 19.1 | 28.6 | **0.776** | [0.737, 0.815] |
| Without wind/solar forecasts | 21.6 | 32.4 | **0.877** | [0.846, 0.908] |

Intervals are 95% bootstrap intervals over whole days. The model beats the baseline clearly, but **less than cross-validation suggested** (0.706). 2026 was a harder year: the mean price was 80 EUR/MWh against 45 in 2025, and the baseline's own error rose 22%. The model's error rose more (about 36%), and the loss is concentrated in the zones where prices moved most. In NO3 and NO5 the model was worse than the baseline (relative MAE 1.09). Across the 12 zones, the rank correlation between the price rise and the deterioration was 0.83. Two explanations fit this and the data cannot separate them: price levels the model rarely saw, and hydrology (reservoir levels) that no feature describes. Neither has been tested.

### What the model uses

SHAP values (model fitted on 2021–2024, explained on 2025) show that price history carries about 46% of the attribution (the previous day's price alone, 28%), the wind, solar and load forecasts about 40%, and the German features 13% combined. Calendar features are used little, because the daily shape is already in the previous day's price.

### Limitations

- **Forecast vintage.** The stored wind/solar forecasts carry no publication time, and the regulation guarantees them only by 18:00 on D-1, after the auction closes. Removing them gives a lower bracket (0.929 in cross-validation, 0.877 on the test year). Independent noise at the size of the German forecast error costs only about 0.01, so the model relies on the forecast's level more than its precision. Errors correlated across zones were not tested.
- **One test year**, and an unusual one. The interval reflects which days happened to be sampled, not a change of regime.
- **Reservoir levels are not features.** The hydro-dominated Norwegian zones are the weakest.
- **Selection effects.** Features were chosen on the same three folds used to report cross-validation scores, so 0.706 is somewhat optimistic by construction.

## Known Data Gaps

- **Swedish load forecast, 2025-12-04:** the API returns no day-ahead load forecast for SE1–SE4 that day (96 zone-hours). Left as null.
- **Actual generation and load reporting gaps:** about 12 zone-hours where a zone's generation feed reported nothing (all four Swedish zones at 2022-03-27 23:00; Danish zones on 2025-04-08, 2025-05-08 and 2026-05-21) and 44 zone-hours of missing actual load. They break the 7-day renewable-ratio feature for the following week.
- **Zero-forecast runs:** 1,577 zone-hours nulled, as described above.
- **NO5:** no wind or solar in the data, so its renewable features are constant.
- **Warm-up rows:** lag and rolling features are null for the first hours of each zone, by construction.
- **Forecast vintage:** the data holds one forecast per hour, with no record of when it was published or revised. The pipeline assumes the forecast for day D was available before the auction closed, which is standard practice but still an assumption.

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

### 5. Pull the data and build the tables

```bash
python scripts/pull_historical_data.py
python scripts/build_dataset.py
python scripts/build_features.py
```

The first full pull takes a while (tens of minutes per data type). Finished past years are cached permanently; the current year is re-fetched on every run.

To reproduce the modeling results:

```bash
python scripts/evaluate_baselines.py   # naive baselines per year, zone and weekday
python scripts/evaluate_models.py      # the model ladder on the cross-validation folds
python scripts/tune_xgboost.py         # random search (long: about 39 model fits)
python scripts/final_test.py           # the 2026 test, run once
python scripts/explain_model.py        # SHAP importance
```

- `data/raw/` — one Parquet file per data type and year.
- `data/processed/hourly_dataset.parquet` — all sources joined by zone and hour. Contains same-hour actuals, so it is not model-safe.
- `data/processed/feature_table.parquet` — the forecast-safe, model-ready feature table.

### 6. Run the tests

```bash
pytest                                   # 38 passed with the data built, 5 skipped without it
pytest --cov=energy_forecaster --cov-report=term-missing
pre-commit install                       # run the checks on every commit
```

### 7. Or use Docker

```bash
docker build -t energy-forecaster .
docker run --rm energy-forecaster                                   # tests, without data
docker run --rm -v "$PWD/data:/app/data" energy-forecaster          # tests with the local data
docker run --rm --env-file .env -v "$PWD/data:/app/data" energy-forecaster \
    python scripts/pull_historical_data.py                          # pull data; the token is passed at run time
```

The ENTSO-E token is never copied into the image (`.env` is in `.dockerignore`).

## Project Status

- [x] Phase 1: Data ingestion, DuckDB warehouse layer, hourly modeling dataset
- [x] Phase 2: Feature engineering and leakage-free preprocessing pipeline
- [x] Phase 3: Model training & evaluation
- [x] Phase 4: Testing, CI/CD, Docker
- [ ] Phase 5: Streamlit dashboard & deployment
