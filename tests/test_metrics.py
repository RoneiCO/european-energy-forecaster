import polars as pl
import pytest

from energy_forecaster.metrics import relative_mae, score

ACTUAL = pl.Series([50.0, 60.0, 40.0])
YESTERDAY = pl.Series([40.0, 70.0, 50.0])
MODEL = pl.Series([48.0, 58.0, 44.0])


def test_score_matches_hand_calculation() -> None:
    # absolute errors 2, 2, 4: MAE = 8/3; squared errors 4, 4, 16: RMSE = sqrt(24/3)
    result = score(ACTUAL, MODEL)
    assert result["mae"] == pytest.approx(8 / 3)
    assert result["rmse"] == pytest.approx(8**0.5)


def test_relative_mae_matches_the_worked_example() -> None:
    ratio = relative_mae(score(ACTUAL, MODEL)["mae"], score(ACTUAL, YESTERDAY)["mae"])
    assert ratio == pytest.approx(8 / 3 / 10)  # 0.267


def test_a_baseline_scored_against_itself_is_exactly_one() -> None:
    assert relative_mae(7.5, 7.5) == 1.0
