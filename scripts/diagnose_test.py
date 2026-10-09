import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import BASELINE_COLUMN
from energy_forecaster.preprocessing import TARGET_COLUMN


def main() -> None:
    df = load_feature_table().drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])
    by_year = (
        df.group_by(pl.col("timestamp").dt.year().alias("year"))
        .agg(
            pl.col(TARGET_COLUMN).mean().round(1).alias("mean_price"),
            pl.col(TARGET_COLUMN).std().round(1).alias("price_sd"),
            (pl.col(TARGET_COLUMN) - pl.col(BASELINE_COLUMN))
            .abs()
            .mean()
            .round(1)
            .alias("mae_yesterday"),
        )
        .sort("year")
    )
    zone_year = (
        df.with_columns(pl.col("timestamp").dt.year().alias("year"))
        .filter(pl.col("year") >= 2023)
        .group_by("zone", "year")
        .agg(pl.col(TARGET_COLUMN).mean().round(1))
        .pivot(on="year", index="zone", values=TARGET_COLUMN)
        .with_columns(
            (pl.col("2026") - (pl.col("2023") + pl.col("2024") + pl.col("2025")) / 3)
            .round(1)
            .alias("rise_2026")
        )
        .sort("zone")
    )
    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(zone_year)
    print(by_year)


if __name__ == "__main__":
    main()
