from datetime import UTC, date, datetime

import polars as pl
import pytest

from energy_forecaster.dashboard import complete_days, day_summary, select_day, zone_context


def _hourly(start: date, end: date) -> pl.DataFrame:
    stamps = pl.datetime_range(start, end, "1h", time_zone="UTC", eager=True)
    return pl.DataFrame({"timestamp": stamps}).with_columns(zone=pl.lit("SE_1"))


def test_select_day_uses_the_stockholm_calendar():
    result = select_day(_hourly(date(2023, 12, 30), date(2024, 1, 3)), date(2024, 1, 1))
    assert result.height == 24
    assert result["timestamp"].min() == datetime(2023, 12, 31, 23, tzinfo=UTC)


def test_select_day_handles_spring_forward():
    result = select_day(_hourly(date(2024, 3, 29), date(2024, 4, 2)), date(2024, 3, 31))
    assert result.height == 23


def test_select_day_handles_fall_back():
    result = select_day(_hourly(date(2024, 10, 25), date(2024, 10, 29)), date(2024, 10, 27))
    assert result.height == 25


def test_day_summary_matches_hand_calculation():
    s = day_summary([10, 20, 30], [12, 18, 36], [11, 19, 31])
    assert s["mae_yesterday"] == pytest.approx(10 / 3)
    assert s["mae_model"] == pytest.approx(1.0)
    assert s["relative_mae"] == pytest.approx(0.3)


def test_day_summary_ignores_rows_with_missing_values():
    s = day_summary([10, 20, float("nan")], [12, 18, 36], [11, 19, 31])
    assert s["mae_yesterday"] == pytest.approx(2.0)
    assert s["mae_model"] == pytest.approx(1.0)


def test_day_summary_rejects_all_missing():
    with pytest.raises(ValueError, match="No rows"):
        day_summary([float("nan")], [1.0], [1.0])


def test_complete_days_drops_partial_days():
    def local_hours(start: date, end: date) -> pl.Series:
        return pl.datetime_range(
            start, end, "1h", time_zone="Europe/Stockholm", eager=True, closed="left"
        )

    stamps = pl.concat(
        [
            local_hours(
                date(2024, 3, 30), date(2024, 4, 1)
            ),  # 30 March: 24 h, 31 March: 23 h (DST)
            local_hours(date(2024, 4, 1), date(2024, 4, 2)).head(10),  # only 10 hours of 1 April
        ]
    )
    df = pl.DataFrame({"timestamp": stamps}).with_columns(zone=pl.lit("SE_1"))
    assert complete_days(df) == [date(2024, 3, 30), date(2024, 3, 31)]


def test_complete_days_uses_the_local_calendar():
    stamps = pl.datetime_range(
        date(2024, 1, 10),
        date(2024, 1, 11),
        "1h",
        time_zone="Europe/Stockholm",
        eager=True,
        closed="left",
    ).dt.convert_time_zone("UTC")
    df = pl.DataFrame({"timestamp": stamps}).with_columns(zone=pl.lit("SE_1"))
    assert complete_days(df, min_hours=24) == [date(2024, 1, 10)]


def _rows(zone: str, day: int, actual, predicted, yesterday) -> pl.DataFrame:
    stamps = [datetime(2024, 1, day, hour, tzinfo=UTC) for hour in (12, 13)]
    return pl.DataFrame(
        {
            "timestamp": stamps,
            "zone": [zone] * 2,
            "actual": actual,
            "predicted": predicted,
            "yesterday": yesterday,
        }
    )


def test_zone_context_pools_errors_and_counts_days():
    predictions = pl.concat(
        [
            _rows("A", 10, [200, 200], [190, 210], [100, 300]),  # model 20 vs baseline 200: a win
            _rows("A", 11, [100, 100], [70, 130], [80, 120]),  # model 60 vs baseline 40: a loss
            _rows("A", 12, [100, 100], [90, 110], [90, 110]),  # 20 vs 20: a tie, not a win
            _rows("B", 10, [0, 0], [1000, 1000], [0, 0]),  # another zone, must be ignored
        ]
    )
    result = zone_context(predictions, "A")
    assert result["pooled_relative_mae"] == pytest.approx(100 / 260)
    assert result["win_rate"] == pytest.approx(1 / 3)
    assert result["days"] == 3


def test_zone_context_rejects_unknown_zone():
    with pytest.raises(ValueError, match="No predictions"):
        zone_context(_rows("A", 10, [1, 1], [1, 1], [1, 1]), "Z")
