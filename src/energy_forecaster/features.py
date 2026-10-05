import math
from pathlib import Path

import holidays
import polars as pl

from energy_forecaster.config import EUROPEAN_ZONES, REFERENCE_TIMEZONE, Zone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_dataset.parquet"
FEATURE_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_table.parquet"


def load_hourly_dataset(
    path: Path = PROCESSED_DATA_PATH, timezone: str = REFERENCE_TIMEZONE
) -> pl.DataFrame:
    """Load the joined hourly dataset, showing timestamps in `timezone`, sorted per zone."""
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("timestamp").dt.convert_time_zone(timezone))
        .sort(["zone", "timestamp"])
    )


def add_local_time(df: pl.DataFrame, zones: dict[str, Zone] = EUROPEAN_ZONES) -> pl.DataFrame:
    """Add `local_time`: each row's wall-clock time in its own zone's timezone, no tz attached.

    A Polars datetime column holds a single time zone, so rows from zones in different
    time zones can't share one tz-aware column. Converting to the zone's clock and then
    stripping the zone label gives a plain value every row can share.
    """
    tz_by_zone = {code: zone.timezone for code, zone in zones.items()}
    row_timezone = pl.col("zone").replace_strict(tz_by_zone)  # raises on an unknown zone
    local_time = pl.coalesce(
        [
            pl.when(row_timezone == tz).then(
                pl.col("timestamp").dt.convert_time_zone(tz).dt.replace_time_zone(None)
            )
            for tz in sorted(set(tz_by_zone.values()))
        ]
    )
    return df.with_columns(local_time.alias("local_time"))


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add calendar features from each row's local time (requires `add_local_time` first)."""
    df = df.with_columns(
        pl.col("local_time").dt.hour().alias("hour"),
        pl.col("local_time").dt.weekday().alias("day_of_week"),  # 1=Monday ... 7=Sunday
        pl.col("local_time").dt.month().alias("month"),
    )
    df = df.with_columns((pl.col("day_of_week") >= 6).alias("is_weekend"))
    return df.with_columns(
        (2 * math.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
        (2 * math.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
        (2 * math.pi * pl.col("day_of_week") / 7).sin().alias("dow_sin"),
        (2 * math.pi * pl.col("day_of_week") / 7).cos().alias("dow_cos"),
        (2 * math.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * math.pi * pl.col("month") / 12).cos().alias("month_cos"),
    )


def _holiday_calendar(country: str, years: range) -> holidays.HolidayBase:
    """Return the public-holiday calendar for a country over the given years."""
    if country == "SE":
        # Sweden's calendar counts every Sunday as a holiday unless told otherwise.
        return holidays.Sweden(years=years, include_sundays=False)
    return holidays.country_holidays(country, years=years)


def _build_holiday_table(countries: list[str], years: range) -> pl.DataFrame:
    """Build a (country, date, is_holiday) lookup table from the `holidays` package."""
    rows = [
        {"country": country, "date": holiday_date, "is_holiday": True}
        for country in countries
        for holiday_date in _holiday_calendar(country, years)
    ]
    return pl.DataFrame(
        rows, schema={"country": pl.String, "date": pl.Date, "is_holiday": pl.Boolean}
    )


def add_holiday_features(df: pl.DataFrame) -> pl.DataFrame:
    """Flag public holidays by each row's local calendar date (requires `add_local_time` first)."""
    years = df["local_time"].dt.year()
    first_year, last_year = years.min(), years.max()
    if not isinstance(first_year, int) or not isinstance(last_year, int):
        raise TypeError("Cannot determine the year range: the dataset is empty")

    holiday_table = _build_holiday_table(
        df["country"].unique().to_list(), range(first_year, last_year + 1)
    )
    return (
        df.with_columns(pl.col("local_time").dt.date().alias("date"))
        .join(holiday_table, on=["country", "date"], how="left")
        .with_columns(pl.col("is_holiday").fill_null(False))
        .drop("date")
        .sort(["zone", "timestamp"])
    )


def add_lag_features(df: pl.DataFrame, lags: tuple[int, ...] = (24, 48, 168)) -> pl.DataFrame:
    """Add price lags in hours, computed separately within each zone.

    Only lags >= 24h are used: at day-ahead auction time, prices are known
    through the end of the previous day, but not for earlier hours of the
    target day itself.
    """
    return df.sort(["zone", "timestamp"]).with_columns(
        [pl.col("price_eur_mwh").shift(lag).over("zone").alias(f"price_lag_{lag}h") for lag in lags]
    )


def add_price_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add rolling mean price features, built on the 24h-lagged price so they stay forecast-safe.

    A rolling window over the raw price would include the hour being predicted.
    Rolling over `price_lag_24h` instead gives "average price over the day/week
    ending 24h before the target hour" -- every value already known safely
    before any auction closes. Requires add_lag_features to have run first.
    """
    return df.sort(["zone", "timestamp"]).with_columns(
        pl.col("price_lag_24h")
        .rolling_mean(window_size=24)
        .over("zone")
        .alias("price_rolling_24h_avg"),
        pl.col("price_lag_24h")
        .rolling_mean(window_size=168)
        .over("zone")
        .alias("price_rolling_7d_avg"),
    )


_NON_GENERATION_COLUMNS = {
    "timestamp",
    "zone",
    "zone_name",
    "country",
    "price_eur_mwh",
    "load_mw",
    "load_forecast_mw",
}
_WIND_SOLAR_TECHNOLOGIES = ("Wind Onshore", "Wind Offshore", "Solar")


def add_renewable_ratio_features(
    df: pl.DataFrame, generation_cols: list[str], lag_hours: int = 48
) -> pl.DataFrame:
    """Add rolling mean/std of the wind+solar share of actual generation, lagged for safety.

    Actual generation isn't known until well after the hour it describes, same as
    load actuals -- a `lag_hours` shift (48h, the safe margin established earlier
    for actuals) is applied before rolling, so every value in the window is safely
    known before any auction for the target day closes.
    """

    renewable_cols = [c for c in generation_cols if c in _WIND_SOLAR_TECHNOLOGIES]
    df = df.sort(["zone", "timestamp"]).with_columns(
        pl.sum_horizontal(renewable_cols).alias("_renewable_mw"),
        pl.sum_horizontal(generation_cols).alias("_total_generation_mw"),
    )

    df = df.with_columns(
        pl.when(pl.col("_total_generation_mw") > 0)
        .then(pl.col("_renewable_mw") / pl.col("_total_generation_mw"))
        .otherwise(None)
        .alias("_renewable_ratio")
    )
    df = df.with_columns(
        pl.col("_renewable_ratio").shift(lag_hours).over("zone").alias("_renewable_ratio_lagged")
    )
    df = df.with_columns(
        pl.col("_renewable_ratio_lagged")
        .rolling_mean(window_size=168)
        .over("zone")
        .alias("renewable_ratio_rolling_7d_mean"),
        pl.col("_renewable_ratio_lagged")
        .rolling_std(window_size=168)
        .over("zone")
        .alias("renewable_ratio_rolling_7d_std"),
    )
    return df.drop(
        "_renewable_mw", "_total_generation_mw", "_renewable_ratio", "_renewable_ratio_lagged"
    )


def add_forecast_renewable_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add forecasted renewable supply as a share of forecasted demand -- safe at full value."""
    forecast_cols = [c for c in df.columns if c.startswith("forecast_")]
    return df.with_columns(
        pl.sum_horizontal(forecast_cols).alias("forecast_renewable_mw"),
        (pl.sum_horizontal(forecast_cols) / pl.col("load_forecast_mw")).alias(
            "forecast_renewable_share_of_load"
        ),
    )


def get_generation_columns(df: pl.DataFrame) -> list[str]:
    """Return the actual-generation technology columns of the freshly loaded hourly dataset.

    Call this immediately after load_hourly_dataset(), before any other add_*
    feature function runs. Those functions add non-generation columns (hour,
    price_lag_24h, local_time, ...) that this by-elimination heuristic cannot
    tell apart from a genuine generation technology column once they exist.
    """
    return [
        c for c in df.columns if c not in _NON_GENERATION_COLUMNS and not c.startswith("forecast_")
    ]


def build_feature_table(path: Path = PROCESSED_DATA_PATH) -> pl.DataFrame:
    """Build the full model-ready feature table from the joined hourly dataset."""
    df = load_hourly_dataset(path)
    generation_cols = get_generation_columns(df)  # captured before any other feature is added
    df = add_local_time(df)
    df = add_temporal_features(df)
    df = add_holiday_features(df)
    df = add_lag_features(df)
    df = add_price_rolling_features(df)
    df = add_renewable_ratio_features(df, generation_cols)
    df = add_forecast_renewable_features(df)
    df = null_suspect_forecast_runs(df)  # must run after add_forecast_renewable_features
    return df.drop(generation_cols)


def null_suspect_forecast_runs(df: pl.DataFrame, min_hours: int = 12) -> pl.DataFrame:
    """Null the renewable-forecast columns inside long runs of exactly zero.

    In a zone that otherwise reports wind/solar forecasts, a forecast of exactly 0 MW
    for half a day or more is missing or placeholder data, not weather: such runs start
    at midnight, span whole days, and coincide across a country's zones. Both origins
    (hours the API omitted that ingestion zero-filled, and zeros the TSO published) look
    identical in stored data, so both are treated as missing.
    """
    forecast_cols = [c for c in df.columns if c.startswith("forecast_")]
    df = df.sort(["zone", "timestamp"]).with_columns(
        (pl.col("forecast_renewable_mw") == 0).alias("_is_zero")
    )
    df = df.with_columns(
        (pl.col("_is_zero") != pl.col("_is_zero").shift(1).over("zone"))
        .fill_null(True)
        .cast(pl.Int32)
        .alias("_new_run")
    )
    df = df.with_columns(pl.col("_new_run").cum_sum().over("zone").alias("_run_id"))
    df = df.with_columns(pl.len().over(["zone", "_run_id"]).alias("_run_hours"))

    has_signal = (pl.col("forecast_renewable_mw") > 0).any().over("zone")
    suspect = pl.col("_is_zero") & (pl.col("_run_hours") >= min_hours) & has_signal
    return df.with_columns(
        [pl.when(suspect).then(None).otherwise(pl.col(c)).alias(c) for c in forecast_cols]
    ).drop("_is_zero", "_new_run", "_run_id", "_run_hours")


def load_feature_table(path: Path = FEATURE_TABLE_PATH) -> pl.DataFrame:
    """Load the model-ready feature table, timestamps shown in the reference time zone."""
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("timestamp").dt.convert_time_zone(REFERENCE_TIMEZONE))
        .sort(["zone", "timestamp"])
    )
