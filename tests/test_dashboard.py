from datetime import UTC, date, datetime

import polars as pl
import pytest

from energy_forecaster.dashboard import day_summary, select_day


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
