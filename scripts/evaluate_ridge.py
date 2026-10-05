import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_ridge_pipeline, evaluate_pipeline
from energy_forecaster.splits import CV_SPLITS


def main() -> None:
    results = evaluate_pipeline(build_ridge_pipeline(), load_feature_table(), CV_SPLITS, "ridge")
    with pl.Config(tbl_rows=20):
        print(results)


if __name__ == "__main__":
    main()
