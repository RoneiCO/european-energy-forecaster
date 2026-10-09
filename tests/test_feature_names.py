import re

from energy_forecaster.preprocessing import ALL_FEATURE_COLUMNS, TARGET_COLUMN


def test_the_target_and_same_hour_actual_load_are_not_features() -> None:
    assert {TARGET_COLUMN, "load_mw"}.isdisjoint(ALL_FEATURE_COLUMNS)


def test_price_lags_are_at_least_24_hours() -> None:
    # The auction for day D closes at 12:00 on D-1, so a shorter lag would use unknown prices.
    for name in ALL_FEATURE_COLUMNS:
        match = re.search(r"price_lag_(\d+)h$", name)
        if match:
            assert int(match.group(1)) >= 24, name


def test_there_are_no_duplicate_feature_names() -> None:
    assert len(ALL_FEATURE_COLUMNS) == len(set(ALL_FEATURE_COLUMNS))
