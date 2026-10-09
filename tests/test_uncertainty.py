import numpy as np
import pytest

from energy_forecaster.uncertainty import bootstrap_ratio


def _synthetic_days() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    baseline = 288 * rng.uniform(18, 32, 273)
    return baseline * rng.uniform(0.6, 0.9, 273), baseline


def test_interval_brackets_the_point_estimate() -> None:
    model, baseline = _synthetic_days()
    low, mid, high = bootstrap_ratio(model, baseline)
    assert low <= model.sum() / baseline.sum() <= high
    assert low <= mid <= high


def test_identical_series_give_exactly_one() -> None:
    series = np.array([1.0, 2.0, 3.0])
    assert bootstrap_ratio(series, series) == (1.0, 1.0, 1.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same days"):
        bootstrap_ratio([1.0], [1.0, 2.0])


def test_same_seed_gives_the_same_interval() -> None:
    model, baseline = _synthetic_days()
    assert bootstrap_ratio(model, baseline, seed=3) == bootstrap_ratio(model, baseline, seed=3)
