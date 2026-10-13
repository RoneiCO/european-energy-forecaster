"""Day-level view of the 2026 test predictions: how typical is a single day?"""

import polars as pl

from energy_forecaster.config import REFERENCE_TIMEZONE

pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(-1)

preds = pl.read_parquet("data/processed/test_predictions_full.parquet").with_columns(
    day=pl.col("timestamp").dt.convert_time_zone(REFERENCE_TIMEZONE).dt.date(),
    model_err=(pl.col("actual") - pl.col("predicted")).abs(),
    base_err=(pl.col("actual") - pl.col("yesterday")).abs(),
)

# 1. Tie-out: the pooled ratio must reproduce the reported headline.
print(
    "pooled relative MAE:",
    round(preds.select(pl.col("model_err").sum() / pl.col("base_err").sum()).item(), 4),
)

daily = (
    preds.group_by("zone", "day")
    .agg(model_mae=pl.col("model_err").mean(), base_mae=pl.col("base_err").mean())
    .with_columns(
        relative=pl.col("model_mae") / pl.col("base_mae"),
        win=pl.col("model_mae") < pl.col("base_mae"),
    )
)
print(
    "share of zone-days where the model beats yesterday:",
    round(daily.select(pl.col("win").mean()).item(), 3),
)

# 2. Pooled ratio, win rate and mean-of-daily-ratios per zone.
by_zone = (
    preds.group_by("zone")
    .agg(pooled=pl.col("model_err").sum() / pl.col("base_err").sum())
    .join(
        daily.group_by("zone").agg(
            win_rate=pl.col("win").mean(),
            mean_of_daily=pl.col("relative").mean(),
            p90_daily=pl.col("relative").quantile(0.9),
        ),
        on="zone",
    )
    .sort("pooled")
)
print(by_zone.with_columns(pl.col(pl.Float64).round(3)))

# 3. The last two weeks of DK_1, to tie out against the app.
last = daily.filter(pl.col("zone") == "DK_1").sort("day").tail(14)
print(last.with_columns(pl.col(pl.Float64).round(2)))
