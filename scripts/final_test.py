from pathlib import Path

import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.metrics import relative_mae, score
from energy_forecaster.models import build_xgboost_pipeline, predict_split
from energy_forecaster.preprocessing import (
    ADOPTED_EXTRA_FEATURES,
    EXTERNAL_FEATURES,
    WIND_SOLAR_FORECAST_FEATURES,
)
from energy_forecaster.splits import TEST_SPLIT
from energy_forecaster.uncertainty import bootstrap_ratio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES = [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]

MODELS = {
    "full": build_xgboost_pipeline(extra_numeric=FEATURES),
    # Copy the exact definition of `xgb_no_wind_solar_fc` from evaluate_models.py here.
    "strict": build_xgboost_pipeline(
        extra_numeric=FEATURES, drop_numeric=WIND_SOLAR_FORECAST_FEATURES
    ),
}


def main() -> None:
    df = load_feature_table()
    for name, pipeline in MODELS.items():
        predictions, train_rows = predict_split(pipeline, df, TEST_SPLIT)
        predictions.write_parquet(
            PROJECT_ROOT / "data" / "processed" / f"test_predictions_{name}.parquet"
        )
        model_scores = score(predictions["actual"], predictions["predicted"])
        baseline_scores = score(predictions["actual"], predictions["yesterday"])

        daily = (
            predictions.with_columns(
                pl.col("timestamp").dt.date().alias("day"),
                (pl.col("actual") - pl.col("predicted")).abs().alias("model_error"),
                (pl.col("actual") - pl.col("yesterday")).abs().alias("yesterday_error"),
            )
            .group_by("day")
            .agg(pl.col("model_error").sum(), pl.col("yesterday_error").sum())
            .sort("day")
        )
        low, mid, high = bootstrap_ratio(daily["model_error"], daily["yesterday_error"])

        print(
            f"{name}: train_rows={train_rows} eval_rows={predictions.height} days={daily.height}\n"
            f"  MAE={model_scores['mae']:.1f} RMSE={model_scores['rmse']:.1f} "
            f"rmse_to_mae={model_scores['rmse'] / model_scores['mae']:.2f}\n"
            f"  relative MAE={relative_mae(model_scores['mae'], baseline_scores['mae']):.3f} "
            f"95% interval [{low:.3f}, {high:.3f}] (bootstrap median {mid:.3f})"
        )


if __name__ == "__main__":
    main()
