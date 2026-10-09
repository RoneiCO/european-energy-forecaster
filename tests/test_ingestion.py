import itertools

import pandas as pd
import pytest

from energy_forecaster.ingestion import (
    _assert_no_interior_gaps,
    _coalesce_duplicate_columns,
    _flatten_generation_columns,
    _normalize_timestamp_column,
    _price_windows,
)

UTC = "UTC"


def _series(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(1.0, index=index)


# --- request windows: a year-long request silently loses a point -----------------


def test_a_year_is_split_into_three_windows() -> None:
    windows = _price_windows(pd.Timestamp("2025-01-01", tz=UTC), pd.Timestamp("2026-01-01", tz=UTC))
    assert len(windows) == 3


def test_no_window_comes_near_a_full_year() -> None:
    windows = _price_windows(pd.Timestamp("2025-01-01", tz=UTC), pd.Timestamp("2026-01-01", tz=UTC))
    assert all(end - start < pd.Timedelta(days=200) for start, end in windows)


def test_windows_overlap_and_cover_the_whole_range() -> None:
    start, end = pd.Timestamp("2025-01-01", tz=UTC), pd.Timestamp("2026-01-01", tz=UTC)
    windows = _price_windows(start, end)
    assert len(windows) > 1  # otherwise the overlap loop below checks nothing
    assert windows[0][0] < start
    assert windows[-1][1] > end
    for (_, previous_end), (next_start, _) in itertools.pairwise(windows):
        assert next_start < previous_end  # padding makes neighbours overlap, so no gap


def test_a_short_range_needs_a_single_window() -> None:
    windows = _price_windows(pd.Timestamp("2025-03-01", tz=UTC), pd.Timestamp("2025-03-31", tz=UTC))
    assert len(windows) == 1


# --- the gap guard ---------------------------------------------------------------


def test_a_complete_hourly_series_passes_the_gap_guard() -> None:
    _assert_no_interior_gaps(
        _series(pd.date_range("2025-03-01", periods=6, freq="h", tz=UTC)), "SE_1"
    )


def test_a_missing_hour_is_caught() -> None:
    index = pd.date_range("2025-03-01", periods=6, freq="h", tz=UTC).delete(3)  # 03:00 lost
    with pytest.raises(ValueError, match="SE_1: points missing"):
        _assert_no_interior_gaps(_series(index), "SE_1")


def test_a_missing_quarter_hour_is_caught() -> None:
    index = pd.date_range("2025-10-01", periods=12, freq="15min", tz=UTC).delete(5)
    with pytest.raises(ValueError, match="points missing"):
        _assert_no_interior_gaps(_series(index), "DK_1")


def test_the_switch_from_hourly_to_quarter_hourly_is_not_a_gap() -> None:
    hourly = pd.date_range("2025-09-30 21:00", periods=3, freq="h", tz=UTC)
    quarter_hourly = pd.date_range("2025-10-01 00:00", periods=8, freq="15min", tz=UTC)
    _assert_no_interior_gaps(_series(hourly.union(quarter_hourly)), "NO_1")


# --- duplicate generation columns ------------------------------------------------


def test_complementary_duplicate_columns_are_merged() -> None:
    nan = float("nan")
    frame = pd.DataFrame(
        [[1.0, nan, 5.0], [nan, 2.0, 6.0]], columns=["Biomass", "Biomass", "Solar"]
    )
    merged = _coalesce_duplicate_columns(frame)
    assert list(merged.columns) == ["Biomass", "Solar"]
    assert merged["Biomass"].tolist() == [1.0, 2.0]


def test_conflicting_duplicate_columns_raise() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["Biomass", "Biomass"])
    with pytest.raises(ValueError, match="Conflicting"):
        _coalesce_duplicate_columns(frame)


def test_a_frame_without_duplicates_is_returned_unchanged() -> None:
    frame = pd.DataFrame([[1.0, 2.0]], columns=["Biomass", "Solar"])
    pd.testing.assert_frame_equal(_coalesce_duplicate_columns(frame), frame)


# --- tuple-shaped generation columns ---------------------------------------------


def test_a_true_multiindex_keeps_only_actual_aggregated() -> None:
    columns = pd.MultiIndex.from_tuples(
        [
            ("Wind Onshore", "Actual Aggregated"),
            ("Hydro Pumped Storage", "Actual Consumption"),
            ("Hydro Pumped Storage", "Actual Aggregated"),
        ]
    )
    frame = pd.DataFrame([[1.0, 2.0, 3.0]], columns=columns)
    assert list(_flatten_generation_columns(frame).columns) == [
        "Wind Onshore",
        "Hydro Pumped Storage",
    ]


def test_a_flat_index_of_tuples_is_handled_too() -> None:
    columns = pd.Index(
        [("Biomass", "Actual Aggregated"), "Solar", ("Marine", "Actual Consumption")],
        tupleize_cols=False,
    )
    frame = pd.DataFrame([[1.0, 2.0, 3.0]])
    frame.columns = columns
    assert list(_flatten_generation_columns(frame).columns) == ["Biomass", "Solar"]


# --- timestamps across daylight saving time --------------------------------------


def test_mixed_offsets_become_one_local_timezone_with_correct_calendar_years() -> None:
    frame = pd.DataFrame({"timestamp": ["2025-12-31 23:00:00+00:00", "2025-07-01 10:00:00+02:00"]})
    result = _normalize_timestamp_column(frame)
    assert str(result["timestamp"].dt.tz) == "Europe/Stockholm"
    # 23:00 UTC on 31 December is already 00:00 on 1 January in Stockholm
    assert result["timestamp"].dt.year.tolist() == [2026, 2025]
