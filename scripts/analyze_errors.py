import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_xgboost_pipeline, predict_split
from energy_forecaster.preprocessing import ADOPTED_EXTRA_FEATURES, EXTERNAL_FEATURES
from energy_forecaster.splits import CV_SPLITS

PAIRS = [("DK_1", "DK_2"), ("SE_3", "SE_4"), ("DK_1", "NO_2"), ("NO_1", "NO_2"), ("DK_2", "SE_4")]


def collect_predictions(df: pl.DataFrame, extra_numeric: list[str]) -> pl.DataFrame:
    """Out-of-sample predictions on every CV split, stacked, with residuals and absolute errors."""
    frames = []
    for split in CV_SPLITS:
        pipeline = build_xgboost_pipeline(extra_numeric=extra_numeric)
        predictions, _ = predict_split(pipeline, df, split)
        frames.append(predictions.with_columns(pl.lit(split.name).alias("split")))
    return pl.concat(frames).with_columns(
        (pl.col("actual") - pl.col("predicted")).alias("residual"),
        (pl.col("actual") - pl.col("predicted")).abs().alias("model_error"),
        (pl.col("actual") - pl.col("yesterday")).abs().alias("yesterday_error"),
    )


def relative_mae_by_zone(pred: pl.DataFrame, label: str) -> pl.DataFrame:
    return pred.group_by("zone").agg(
        (pl.col("model_error").mean() / pl.col("yesterday_error").mean()).round(3).alias(label)
    )


def pair_correlations(pred: pl.DataFrame) -> dict[tuple[str, str], float]:
    """Correlation of hourly residuals for selected zone pairs."""
    wide = pred.pivot(on="zone", index="timestamp", values="residual").drop_nulls()
    return {pair: round(wide.select(pl.corr(*pair)).item(), 2) for pair in PAIRS}


def main() -> None:
    df = load_feature_table()
    reference = collect_predictions(df, ADOPTED_EXTRA_FEATURES)
    germany = collect_predictions(df, [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES])

    by_zone = (
        relative_mae_by_zone(reference, "rel_forecasts")
        .join(relative_mae_by_zone(germany, "rel_germany"), on="zone")
        .with_columns((pl.col("rel_forecasts") - pl.col("rel_germany")).round(3).alias("gain"))
        .sort("gain", descending=True)
    )
    with pl.Config(tbl_rows=20):
        print(by_zone)

    ref_corr, ger_corr = pair_correlations(reference), pair_correlations(germany)
    for pair in PAIRS:
        print(pair, "forecasts:", ref_corr[pair], "germany:", ger_corr[pair])


if __name__ == "__main__":
    main()
