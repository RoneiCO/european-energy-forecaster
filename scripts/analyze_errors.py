import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_xgboost_pipeline, predict_split
from energy_forecaster.preprocessing import CROSS_ZONE_FEATURES
from energy_forecaster.splits import CV_SPLITS


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
    frames = []
    for split in CV_SPLITS:
        predictions, _ = predict_split(
            build_xgboost_pipeline(extra_numeric=CROSS_ZONE_FEATURES), df, split
        )
        frames.append(predictions.with_columns(pl.lit(split.name).alias("split")))

    pred = pl.concat(frames).with_columns(
        pl.col("timestamp").dt.hour().alias("hour"),
        (pl.col("actual") - pl.col("predicted")).abs().alias("model_error"),
        (pl.col("actual") - pl.col("yesterday")).abs().alias("yesterday_error"),
    )

    with pl.Config(tbl_rows=30):
        print(summarize(pred, "zone"))
        print(summarize(pred, "hour"))

    n_worst = pred.height // 100  # the worst 1% of hours
    print(
        pred.select(
            (
                pl.col("model_error").sort(descending=True).head(n_worst).sum()
                / pl.col("model_error").sum()
            )
            .round(3)
            .alias("model_worst_1pct_share"),
            (
                pl.col("yesterday_error").sort(descending=True).head(n_worst).sum()
                / pl.col("yesterday_error").sum()
            )
            .round(3)
            .alias("yesterday_worst_1pct_share"),
        )
    )

    pred = pred.with_columns((pl.col("actual") - pl.col("predicted")).alias("residual"))

    # Mean of (actual - predicted) per split: negative means the model over-predicts
    print(
        pred.group_by("split")
        .agg(pl.col("residual").mean().round(1).alias("mean_residual"))
        .sort("split")
    )

    # Do the zones' errors move together? Correlation of hourly residuals between zones.
    wide = (
        pred.pivot(on="zone", index="timestamp", values="residual").drop_nulls().drop("timestamp")
    )
    corr = (
        wide.corr()
        .with_columns(pl.Series("zone", wide.columns))
        .select("zone", pl.exclude("zone"))
        .with_columns(pl.exclude("zone").round(2))
    )
    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(corr)


if __name__ == "__main__":
    main()
