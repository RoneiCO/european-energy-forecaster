import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.metrics import score
from energy_forecaster.splits import CV_SPLITS, TEST_SPLIT, apply_split

NAIVE_FORECASTS: dict[str, pl.Expr] = {
    "yesterday": pl.col("price_lag_24h"),
    "last_week": pl.col("price_lag_168h"),
    # Standard benchmark: yesterday for Tue-Fri, the same day last week for Sat-Mon
    "weekly_mix": pl.when(pl.col("day_of_week").is_in([6, 7, 1]))
    .then(pl.col("price_lag_168h"))
    .otherwise(pl.col("price_lag_24h")),
}


def main() -> None:
    df = load_feature_table()
    rows: list[dict[str, str | int | float]] = []
    for split in [*CV_SPLITS, TEST_SPLIT]:
        train, evaluation = apply_split(df, split)
        for name, forecast in NAIVE_FORECASTS.items():
            scored = evaluation.select(
                pl.col("price_eur_mwh").alias("actual"), forecast.alias("naive")
            ).drop_nulls()
            result = score(scored["actual"], scored["naive"])
            rows.append(
                {
                    "split": split.name,
                    "train_rows": train.height,
                    "eval_rows": scored.height,
                    "baseline": name,
                    "mae": round(result["mae"], 1),
                    "rmse": round(result["rmse"], 1),
                }
            )
    with pl.Config(tbl_rows=20):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
