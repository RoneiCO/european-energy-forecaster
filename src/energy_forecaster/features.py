import math
from pathlib import Path

import holidays
import polars as pl

from energy_forecaster.config import EUROPEAN_ZONES, REFERENCE_TIMEZONE, Zone

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_dataset.parquet"


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
