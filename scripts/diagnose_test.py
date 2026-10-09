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
    print(by_year)


if __name__ == "__main__":
    main()
