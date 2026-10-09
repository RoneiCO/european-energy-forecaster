from pathlib import Path

import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_xgboost_pipeline, predict_split
from energy_forecaster.preprocessing import ADOPTED_EXTRA_FEATURES, EXTERNAL_FEATURES
from energy_forecaster.splits import CV_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES = [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]


def add_errors(pred: pl.DataFrame) -> pl.DataFrame:
    return pred.with_columns(
        (pl.col("actual") - pl.col("predicted")).abs().alias("model_error"),
        (pl.col("actual") - pl.col("yesterday")).abs().alias("yesterday_error"),
    )


def summarize(pred: pl.DataFrame, by: str) -> pl.DataFrame:
    """Model vs yesterday MAE for each value of the column `by`."""
    return (
        pred.group_by(by)
        .agg(
            pl.col("actual").mean().round(1).alias("mean_price"),
            pl.col("model_error").mean().round(1).alias("mae"),
            pl.col("yesterday_error").mean().round(1).alias("mae_yesterday"),
            (pl.col("model_error").mean() / pl.col("yesterday_error").mean())
            .round(3)
            .alias("relative_mae"),
        )
        .sort(by)
    )


def main() -> None:
    df = load_feature_table()
    test_path = PROJECT_ROOT / "data" / "processed" / "test_predictions_full.parquet"
    test = add_errors(pl.read_parquet(test_path))

    cv_frames = []
    for split in CV_SPLITS:
        predictions, _ = predict_split(build_xgboost_pipeline(extra_numeric=FEATURES), df, split)
        cv_frames.append(predictions)
    cv = add_errors(pl.concat(cv_frames))

    by_zone = (
        summarize(cv, "zone")
        .select("zone", pl.col("relative_mae").alias("cv"))
        .join(
            summarize(test, "zone").select("zone", pl.col("relative_mae").alias("test")),
            on="zone",
        )
        .with_columns((pl.col("test") - pl.col("cv")).round(3).alias("change"))
        .sort("change")
    )
    by_month = summarize(test.with_columns(pl.col("timestamp").dt.month().alias("month")), "month")

    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(by_zone)
        print(by_month)


if __name__ == "__main__":
    main()
