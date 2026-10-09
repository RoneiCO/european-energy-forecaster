from datetime import date

import polars as pl

from energy_forecaster.features import (
    add_holiday_features,
    add_lag_features,
    add_local_time,
    null_suspect_forecast_runs,
)


def _hours(last_day: date) -> pl.Series:
    """Hourly timestamps from 1 March 2025, 00:00 up to 00:00 on `last_day`, Stockholm time."""
    return pl.datetime_range(
        date(2025, 3, 1),
        last_day,
        interval="1h",
        time_zone="Europe/Stockholm",
        eager=True,
    )


def _price_frame(zone: str, first_price: float) -> pl.DataFrame:
    """169 hourly rows whose price at row i is first_price + i."""
    return pl.DataFrame({"timestamp": _hours(date(2025, 3, 8))}).with_columns(
        pl.lit(zone).alias("zone"),
        (first_price + pl.int_range(pl.len())).cast(pl.Float64).alias("price_eur_mwh"),
    )


def _forecast_frame(zone: str, values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": _hours(date(2025, 3, 2)),  # 25 rows
            "forecast_renewable_mw": values,
            "forecast_wind_onshore_mw": values,
        }
    ).with_columns(pl.lit(zone).alias("zone"))


def test_lags_are_computed_inside_each_zone() -> None:
    frame = pl.concat([_price_frame("SE_1", 0.0), _price_frame("SE_2", 1000.0)])
    lagged = add_lag_features(frame)
    for zone, first_price in [("SE_1", 0.0), ("SE_2", 1000.0)]:
        part = lagged.filter(pl.col("zone") == zone)
        assert part["price_lag_24h"].head(24).null_count() == 24  # warm-up, never another zone
        assert part["price_lag_24h"][24] == first_price
        assert part["price_lag_24h"][100] == first_price + 76
        assert part["price_lag_168h"][168] == first_price


def test_local_time_follows_each_zones_own_clock() -> None:
    # One instant, 23:00 on 30 December in Stockholm, for a Swedish and a Finnish zone.
    frame = pl.DataFrame(
        {"timestamp": ["2025-12-30 23:00:00"] * 2, "zone": ["SE_3", "FI"]}
    ).with_columns(pl.col("timestamp").str.to_datetime().dt.replace_time_zone("Europe/Stockholm"))
    result = add_local_time(frame).sort("zone")  # FI sorts before SE_3
    assert result["zone"].to_list() == ["FI", "SE_3"]
    assert result["local_time"].dt.hour().to_list() == [0, 23]  # Finland is an hour ahead
    assert result["local_time"].dt.day().to_list() == [31, 30]


def test_holiday_flags_exclude_swedish_sundays() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": ["2025-03-02 12:00:00", "2025-12-25 12:00:00", "2025-12-06 12:00:00"],
            "zone": ["SE_3", "SE_3", "FI"],
            "country": ["SE", "SE", "FI"],
        }
    ).with_columns(pl.col("timestamp").str.to_datetime().dt.replace_time_zone("Europe/Stockholm"))
    result = add_holiday_features(add_local_time(frame)).with_columns(
        pl.col("local_time").dt.date().cast(pl.String).alias("day")
    )
    flags = dict(zip(result["day"], result["is_holiday"], strict=True))
    assert flags == {
        "2025-03-02": False,  # an ordinary Sunday in Sweden
        "2025-12-25": True,  # Christmas Day
        "2025-12-06": True,  # Finland's Independence Day
    }


def test_a_twelve_hour_zero_run_is_nulled_and_a_shorter_one_is_kept() -> None:
    values = [5.0] * 4 + [0.0] * 12 + [7.0] * 6 + [0.0] * 3
    result = null_suspect_forecast_runs(_forecast_frame("DK_1", values))
    expected_nulls = [False] * 4 + [True] * 12 + [False] * 9
    assert result["forecast_renewable_mw"].is_null().to_list() == expected_nulls
    assert result["forecast_wind_onshore_mw"].is_null().to_list() == expected_nulls
    assert result["forecast_renewable_mw"].tail(3).to_list() == [0.0, 0.0, 0.0]


def test_a_zone_that_never_reports_a_signal_is_left_alone() -> None:
    result = null_suspect_forecast_runs(_forecast_frame("NO_5", [0.0] * 25))
    assert result["forecast_renewable_mw"].null_count() == 0
