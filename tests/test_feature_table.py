import polars as pl
import pytest

from energy_forecaster.config import EUROPEAN_ZONES
from energy_forecaster.features import FEATURE_TABLE_PATH, load_feature_table
from energy_forecaster.preprocessing import (
    ALL_FEATURE_COLUMNS,
    BOOLEAN_FEATURES,
    CYCLICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
)


@pytest.fixture(scope="module")
def feature_table() -> pl.DataFrame:
    if not FEATURE_TABLE_PATH.exists():
        pytest.skip("feature_table.parquet not built (run scripts/build_features.py)")
    return load_feature_table()


def test_every_model_feature_is_in_the_table(feature_table: pl.DataFrame) -> None:
    assert set(ALL_FEATURE_COLUMNS) <= set(feature_table.columns)


def test_there_is_one_row_per_zone_and_hour(feature_table: pl.DataFrame) -> None:
    assert feature_table.select("zone", "timestamp").n_unique() == feature_table.height


def test_all_twelve_zones_are_present(feature_table: pl.DataFrame) -> None:
    assert set(feature_table["zone"].unique()) == set(EUROPEAN_ZONES)


def test_price_lag_24h_is_really_the_price_24_hours_earlier(feature_table: pl.DataFrame) -> None:
    earlier = feature_table.select(
        "zone",
        (pl.col("timestamp") + pl.duration(hours=24)).alias("timestamp"),
        pl.col(TARGET_COLUMN).alias("price_24h_earlier"),
    )
    joined = feature_table.join(earlier, on=["zone", "timestamp"], how="inner").drop_nulls(
        ["price_lag_24h", "price_24h_earlier"]
    )
    with_lag = feature_table.drop_nulls("price_lag_24h").height
    assert joined.height > 0.99 * with_lag  # nearly every row has a counterpart 24 hours back
    assert (joined["price_lag_24h"] == joined["price_24h_earlier"]).all()


def test_the_base_preprocessor_produces_the_documented_33_columns(
    feature_table: pl.DataFrame,
) -> None:
    sample = feature_table.sample(5000, seed=0).select(ALL_FEATURE_COLUMNS).to_pandas()
    transformed = build_preprocessor().fit_transform(sample)
    expected = (
        len(NUMERIC_FEATURES) + len(CYCLICAL_FEATURES) + len(BOOLEAN_FEATURES) + len(EUROPEAN_ZONES)
    )
    assert transformed.shape[1] == expected == 33
