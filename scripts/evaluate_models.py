import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import (
    build_ridge_pipeline,
    build_xgboost_pipeline,
    evaluate_pipeline,
)
from energy_forecaster.splits import CV_SPLITS


def main() -> None:
    df = load_feature_table()
    results = pl.concat(
        [
            evaluate_pipeline(build_ridge_pipeline(), df, CV_SPLITS, "ridge"),
            evaluate_pipeline(build_xgboost_pipeline(), df, CV_SPLITS, "xgb_level"),
            evaluate_pipeline(
                build_xgboost_pipeline(), df, CV_SPLITS, "xgb_change", predict_change=True
            ),
        ]
    ).sort(["split", "model"])
    with pl.Config(tbl_rows=20):
        print(results)


if __name__ == "__main__":
    main()