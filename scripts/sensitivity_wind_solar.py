import polars as pl
from polars.testing import assert_frame_equal
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from energy_forecaster.features import load_feature_table, load_hourly_dataset
from energy_forecaster.metrics import relative_mae, score
from energy_forecaster.models import BASELINE_COLUMN, HISTORY_COLUMNS, build_xgboost_pipeline
from energy_forecaster.preprocessing import (
    ADOPTED_EXTRA_FEATURES,
    ALL_FEATURE_COLUMNS,
    EXTERNAL_FEATURES,
    TARGET_COLUMN,
    WIND_SOLAR_FORECAST_FEATURES,
)
from energy_forecaster.sensitivity import (
    german_error_sd,
    nordic_error_scales,
    perturb_forecast_features,
)
from energy_forecaster.splits import CV_SPLITS, apply_split

NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0]


def main() -> None:
    df = load_feature_table()
    nordic_scales = nordic_error_scales(load_hourly_dataset())
    german_sd = german_error_sd(df)
    german_mean = df.select(pl.col("de_forecast_renewable_mw").mean()).item()
    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(nordic_scales.sort("zone"))
    print(
        f"German forecast error SD: {german_sd:.0f} MW ({german_sd / german_mean:.1%} of the mean)"
    )

    pipeline = build_xgboost_pipeline(extra_numeric=[*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES])
    models: dict[str, Pipeline] = {}
    for split in CV_SPLITS:  # fit once per split, on clean features
        train, _ = apply_split(df, split)
        train = train.drop_nulls([TARGET_COLUMN, *HISTORY_COLUMNS])
        models[split.name] = clone(pipeline).fit(
            train.select(ALL_FEATURE_COLUMNS).to_pandas(), train[TARGET_COLUMN].to_numpy()
        )

    evaluation_years = df.filter(pl.col("timestamp").dt.year().is_between(2023, 2025))
    rows: list[dict[str, str | float]] = []
    for k in NOISE_LEVELS:
        noisy = perturb_forecast_features(evaluation_years, nordic_scales, german_sd, k)
        if k == 0.0:  # self-test: zero noise must rebuild exactly the stored features
            assert_frame_equal(
                evaluation_years.select(WIND_SOLAR_FORECAST_FEATURES),
                noisy.select(WIND_SOLAR_FORECAST_FEATURES),
            )
        for split in CV_SPLITS:
            _, evaluation = apply_split(noisy, split)
            evaluation = evaluation.drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])
            predicted = models[split.name].predict(
                evaluation.select(ALL_FEATURE_COLUMNS).to_pandas()
            )
            model_mae = score(evaluation[TARGET_COLUMN], pl.Series(predicted))["mae"]
            baseline_mae = score(evaluation[TARGET_COLUMN], evaluation[BASELINE_COLUMN])["mae"]
            rows.append(
                {
                    "split": split.name,
                    "k": str(k),
                    "relative_mae": round(relative_mae(model_mae, baseline_mae), 3),
                }
            )
    print(pl.DataFrame(rows).pivot(on="k", index="split", values="relative_mae"))


if __name__ == "__main__":
    main()
