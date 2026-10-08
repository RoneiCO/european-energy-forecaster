import random

import polars as pl

from energy_forecaster.features import _read_raw_hourly, _total_if_complete

# Per-zone forecast column -> the matching actual-generation column in the hourly dataset
NORDIC_FORECAST_COLUMNS = {
    "forecast_wind_onshore_mw": "Wind Onshore",
    "forecast_solar_mw": "Solar",
    "forecast_wind_offshore_mw": "Wind Offshore",
}
GERMAN_ACTUAL_COLUMNS = ("Wind Onshore", "Wind Offshore", "Solar")


def nordic_error_scales(raw: pl.DataFrame) -> pl.DataFrame:
    """Standard deviation of (forecast - actual) in MW, per zone and technology.

    `raw` is the joined hourly dataset, which holds both forecasts and actuals. Hours where the
    forecast is exactly zero while generation is not are missing-data zeros, not forecast
    error, and are excluded.
    """
    forecast_total = pl.sum_horizontal(list(NORDIC_FORECAST_COLUMNS))
    actual_total = pl.sum_horizontal(list(NORDIC_FORECAST_COLUMNS.values()))
    usable = ~((forecast_total == 0) & (actual_total > 0))
    return raw.group_by("zone").agg(
        [
            (pl.col(forecast) - pl.col(actual)).filter(usable).std().fill_null(0.0).alias(forecast)
            for forecast, actual in NORDIC_FORECAST_COLUMNS.items()
        ]
    )


def german_error_sd(df: pl.DataFrame) -> float:
    """Standard deviation (MW) of the German wind+solar day-ahead forecast error.

    `df` is the feature table, which carries the forecast as `de_forecast_renewable_mw`;
    the actuals come from the raw German generation files.
    """
    generation = _read_raw_hourly("external_generation")
    present = [c for c in GERMAN_ACTUAL_COLUMNS if c in generation.columns]
    actual = generation.select(
        "timestamp",
        pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in present]).alias("actual_mw"),
    )
    forecast = df.select("timestamp", "de_forecast_renewable_mw").unique("timestamp").drop_nulls()
    sd = (
        forecast.join(actual, on="timestamp", how="inner")
        .select((pl.col("de_forecast_renewable_mw") - pl.col("actual_mw")).std())
        .item()
    )
    if sd is None:
        raise ValueError("No overlapping hours between German forecasts and actual generation")
    return sd


def perturb_forecast_features(
    df: pl.DataFrame,
    nordic_scales: pl.DataFrame,
    german_sd: float,
    k: float,
    seed: int = 0,
) -> pl.DataFrame:
    """Add noise to the wind/solar forecast features, scaled by k times each forecast's error.

    Per-zone forecasts get independent Gaussian noise with a standard deviation of k times
    that zone's measured forecast error; the German forecast gets one draw per hour, shared
    by all zones. Derived features (renewable totals and shares, system and country totals,
    the German share) are recomputed from the noisy base forecasts so they stay consistent.
    Nulls stay null. At k = 0 the output must equal the input; the evaluation script asserts it.
    """
    rng = random.Random(seed)
    draws = {
        name: pl.Series(f"_noise_{name}", [rng.gauss(0.0, 1.0) for _ in range(df.height)])
        for name in (*NORDIC_FORECAST_COLUMNS, "german")
    }
    sd_by_zone = {
        col: dict(zip(nordic_scales["zone"], nordic_scales[col], strict=True))
        for col in NORDIC_FORECAST_COLUMNS
    }
    noisy = df.with_columns(*draws.values()).with_columns(
        *[
            (
                pl.col(col)
                + k
                * pl.col("zone").replace_strict(sd_by_zone[col], return_dtype=pl.Float64)
                * pl.col(f"_noise_{col}")
            )
            .clip(lower_bound=0.0)
            .alias(col)
            for col in NORDIC_FORECAST_COLUMNS
        ],
        (
            pl.col("de_forecast_renewable_mw")
            + k * german_sd * pl.col("_noise_german").first().over("timestamp")
        )
        .clip(lower_bound=0.0)
        .alias("de_forecast_renewable_mw"),
    )
    return (
        noisy.with_columns(
            pl.when(pl.col("forecast_renewable_mw").is_null())
            .then(None)
            .otherwise(pl.sum_horizontal(list(NORDIC_FORECAST_COLUMNS)))
            .alias("forecast_renewable_mw")
        )
        .with_columns(
            (pl.col("forecast_renewable_mw") / pl.col("load_forecast_mw")).alias(
                "forecast_renewable_share_of_load"
            ),
            (pl.col("de_forecast_renewable_mw") / pl.col("de_load_forecast_mw")).alias(
                "de_forecast_renewable_share_of_load"
            ),
            _total_if_complete("forecast_renewable_mw", "timestamp").alias(
                "system_forecast_renewable_mw"
            ),
            _total_if_complete("forecast_renewable_mw", ["timestamp", "country"]).alias(
                "country_forecast_renewable_mw"
            ),
        )
        .drop([s.name for s in draws.values()])
    )
