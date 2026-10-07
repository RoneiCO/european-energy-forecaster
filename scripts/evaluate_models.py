import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import (
    build_ridge_pipeline,
    build_xgboost_pipeline,
    evaluate_pipeline,
)
from energy_forecaster.preprocessing import (
    ADOPTED_EXTRA_FEATURES,
    EXTERNAL_FEATURES,
    WIND_SOLAR_FORECAST_FEATURES,
)
from energy_forecaster.splits import CV_SPLITS


def main() -> None:
    df = load_feature_table()
    models = {
        "ridge": build_ridge_pipeline(),
        "xgb_level": build_xgboost_pipeline(),
        "xgb_forecasts": build_xgboost_pipeline(extra_numeric=ADOPTED_EXTRA_FEATURES),
        "xgb_germany": build_xgboost_pipeline(
            extra_numeric=[*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]
        ),
        "xgb_no_wind_solar_fc": build_xgboost_pipeline(
            extra_numeric=[*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES],
            drop_numeric=WIND_SOLAR_FORECAST_FEATURES,
        ),
    }
    results = pl.concat(
        [evaluate_pipeline(pipeline, df, CV_SPLITS, name) for name, pipeline in models.items()]
    ).sort(["split", "model"])
    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(results)


if __name__ == "__main__":
    main()
