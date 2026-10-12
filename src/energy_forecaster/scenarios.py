import polars as pl

from energy_forecaster.features import _total_if_complete

# Per-zone forecast column -> the matching actual-generation column in the hourly dataset
NORDIC_FORECAST_COLUMNS = {
    "forecast_wind_onshore_mw": "Wind Onshore",
    "forecast_solar_mw": "Solar",
    "forecast_wind_offshore_mw": "Wind Offshore",
}
WIND_COLUMNS = ("forecast_wind_onshore_mw", "forecast_wind_offshore_mw")


def recompute_forecast_derived(df: pl.DataFrame) -> pl.DataFrame:
    """Rebuild every feature derived from the base day-ahead forecasts.

    Renewable totals, shares of load, and the country and system totals must stay
    consistent with the base forecasts. Nulls stay null: a zone without a forecast makes
    the totals that include it missing, never partial.
    """
    # Two steps on purpose: expressions inside one with_columns read the table as it was
    # before the call, so the totals have to wait for the updated forecast_renewable_mw.
    rebuilt = df.with_columns(
        pl.when(pl.col("forecast_renewable_mw").is_null())
        .then(None)
        .otherwise(pl.sum_horizontal(list(NORDIC_FORECAST_COLUMNS)))
        .alias("forecast_renewable_mw")
    )
    return rebuilt.with_columns(
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
        _total_if_complete("load_forecast_mw", "timestamp").alias("system_load_forecast_mw"),
        _total_if_complete("load_forecast_mw", ["timestamp", "country"]).alias(
            "country_load_forecast_mw"
        ),
    )


def apply_scenario(
    df: pl.DataFrame,
    *,
    wind: float = 1.0,
    solar: float = 1.0,
    load: float = 1.0,
    german_renewables: float = 1.0,
) -> pl.DataFrame:
    """Scale the day-ahead forecasts by the given factors and rebuild the derived features.

    `df` must hold all zones for every timestamp it contains, because the country and
    system totals add the zones up. Lags and rolling statistics are history and are left
    alone. With every factor at 1.0 the stored features are reproduced.
    """
    if min(wind, solar, load, german_renewables) < 0:
        raise ValueError("Scenario factors cannot be negative")
    scaled = df.with_columns(
        *[(pl.col(column) * wind).alias(column) for column in WIND_COLUMNS],
        (pl.col("forecast_solar_mw") * solar).alias("forecast_solar_mw"),
        (pl.col("load_forecast_mw") * load).alias("load_forecast_mw"),
        (pl.col("de_forecast_renewable_mw") * german_renewables).alias("de_forecast_renewable_mw"),
    )
    return recompute_forecast_derived(scaled)
