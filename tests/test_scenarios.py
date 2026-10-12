import polars as pl
import pytest
from polars.testing import assert_frame_equal

from energy_forecaster.scenarios import apply_scenario


def _frame(onshore_se2: float | None = 200.0) -> pl.DataFrame:
    """Three zones at one hour: SE_1 and SE_2 (Sweden), NO_1 (Norway)."""
    return pl.DataFrame(
        {
            "timestamp": [1, 1, 1],
            "zone": ["SE_1", "SE_2", "NO_1"],
            "country": ["SE", "SE", "NO"],
            "forecast_wind_onshore_mw": [100.0, onshore_se2, 350.0],
            "forecast_wind_offshore_mw": [0.0, 0.0, 0.0],
            "forecast_solar_mw": [50.0, 0.0, 100.0],
            "forecast_renewable_mw": [150.0, onshore_se2, 450.0],
            "load_forecast_mw": [1000.0, 500.0, 2000.0],
            "de_forecast_renewable_mw": [1000.0] * 3,
            "de_load_forecast_mw": [5000.0] * 3,
        }
    )


def test_derived_features_follow_the_scaled_forecasts() -> None:
    result = apply_scenario(_frame(), wind=2.0, solar=0.0, german_renewables=0.5)
    # SE_1: 100*2 + 0 = 200   SE_2: 200*2 = 400   NO_1: 350*2 = 700
    assert result["forecast_renewable_mw"].to_list() == [200.0, 400.0, 700.0]
    assert result["forecast_renewable_share_of_load"].to_list() == pytest.approx([0.2, 0.8, 0.35])
    assert result["system_forecast_renewable_mw"].to_list() == [1300.0] * 3
    assert result["country_forecast_renewable_mw"].to_list() == [600.0, 600.0, 700.0]
    assert result["de_forecast_renewable_share_of_load"].to_list() == pytest.approx([0.1] * 3)
    assert result["system_load_forecast_mw"].to_list() == [3500.0] * 3
    assert result["country_load_forecast_mw"].to_list() == [1500.0, 1500.0, 2000.0]


def test_more_load_lowers_the_renewable_share_and_raises_the_totals() -> None:
    result = apply_scenario(_frame(), load=2.0)
    assert result["forecast_renewable_share_of_load"].to_list() == pytest.approx(
        [150 / 2000, 200 / 1000, 450 / 4000]
    )
    assert result["system_load_forecast_mw"].to_list() == [7000.0] * 3


def test_a_missing_forecast_makes_the_totals_that_include_it_missing() -> None:
    result = apply_scenario(_frame(onshore_se2=None), wind=2.0)
    assert result["forecast_renewable_mw"].to_list() == [250.0, None, 800.0]
    assert result["system_forecast_renewable_mw"].to_list() == [None, None, None]
    assert result["country_forecast_renewable_mw"].to_list() == [None, None, 800.0]


def test_scaling_up_and_back_down_returns_to_the_start() -> None:
    there_and_back = apply_scenario(apply_scenario(_frame(), wind=2.0), wind=0.5)
    assert_frame_equal(there_and_back, apply_scenario(_frame()))


def test_negative_factors_are_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        apply_scenario(_frame(), wind=-0.1)
