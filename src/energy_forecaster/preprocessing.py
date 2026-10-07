from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price_eur_mwh"
ID_COLUMNS = ["timestamp", "zone_name", "country"]
CYCLICAL_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
BOOLEAN_FEATURES = ["is_weekend", "is_holiday"]
CATEGORICAL_FEATURES = ["zone"]
NUMERIC_FEATURES = [
    "load_forecast_mw",
    "forecast_wind_onshore_mw",
    "forecast_solar_mw",
    "forecast_wind_offshore_mw",
    "forecast_renewable_mw",
    "forecast_renewable_share_of_load",
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_24h_avg",
    "price_rolling_7d_avg",
    "renewable_ratio_rolling_7d_mean",
    "renewable_ratio_rolling_7d_std",
]
MODEL_FEATURE_COLUMNS = (
    NUMERIC_FEATURES + CATEGORICAL_FEATURES + CYCLICAL_FEATURES + BOOLEAN_FEATURES
)

CROSS_ZONE_FORECAST_FEATURES = [
    "system_forecast_renewable_mw",
    "system_load_forecast_mw",
    "country_forecast_renewable_mw",
    "country_load_forecast_mw",
]
CROSS_ZONE_PRICE_FEATURES = ["system_price_lag_24h"]

CROSS_ZONE_FEATURES = CROSS_ZONE_FORECAST_FEATURES + CROSS_ZONE_PRICE_FEATURES

# Extra numeric features adopted after the ablation: forecast totals only.
# The price-level feature failed the adoption rule (gain of at least 0.01 in every split).
ADOPTED_EXTRA_FEATURES = CROSS_ZONE_FORECAST_FEATURES

EXTERNAL_FEATURES = [
    "de_price_lag_24h",
    "de_load_forecast_mw",
    "de_forecast_renewable_mw",
    "de_forecast_renewable_share_of_load",
]
ALL_FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS + CROSS_ZONE_FEATURES + EXTERNAL_FEATURES


def build_preprocessor(extra_numeric: Sequence[str] = ()) -> ColumnTransformer:
    """Build (but do not fit) the leakage-free preprocessing pipeline.

    `extra_numeric` adds columns to the scaled group, for feature experiments.

    Must be fit only on a training split, never on the full dataset -- fitting
    StandardScaler on data that includes the test period leaks the test period's
    distribution into the model before evaluation, even though no row of test
    data is directly visible to it. See Phase 3 for the temporal split this
    pipeline is fit against.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), [*NUMERIC_FEATURES, *extra_numeric]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("passthrough", "passthrough", CYCLICAL_FEATURES + BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )
