from pathlib import Path

import joblib
import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.preprocessing import ALL_FEATURE_COLUMNS
from energy_forecaster.scenarios import apply_scenario
from energy_forecaster.splits import TEST_SPLIT, apply_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_full.joblib"
DRIVERS = ["wind", "solar", "load", "german_renewables"]
FACTORS = [0.5, 0.75, 1.0, 1.25, 1.5]


def mean_forecast_by_zone(model, table: pl.DataFrame) -> pl.DataFrame:
    predicted = model.predict(table.select(ALL_FEATURE_COLUMNS).to_pandas())
    return (
        table.select("zone")
        .with_columns(pl.Series("predicted", predicted))
        .group_by("zone")
        .agg(pl.col("predicted").mean())
        .sort("zone")
    )


def main() -> None:
    model = joblib.load(MODEL_PATH)
    _, test = apply_split(load_feature_table(), TEST_SPLIT)
    base = mean_forecast_by_zone(model, test)

    rows = []
    for driver in DRIVERS:
        for factor in FACTORS:
            scenario = mean_forecast_by_zone(model, apply_scenario(test, **{driver: factor}))
            change = scenario["predicted"] - base["predicted"]
            mean_change = change.mean()
            if not isinstance(mean_change, float):
                raise TypeError(
                    f"Expected a float mean for {driver} at factor {factor}, got {type(mean_change).__name__}"
                )
            rows.append(
                {
                    "driver": driver,
                    "factor": factor,
                    "mean_change": round(mean_change, 2),
                    **{z: round(c, 1) for z, c in zip(base["zone"], change, strict=True)},
                }
            )
    with pl.Config(tbl_rows=30, tbl_cols=-1):
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
