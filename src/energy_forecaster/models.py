import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from energy_forecaster.metrics import relative_mae, score
from energy_forecaster.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
)
from energy_forecaster.splits import Split, apply_split

# Null only in a zone's first hours, before any history exists. Training rows missing
# these are dropped rather than imputed: a median "yesterday's price" is not a real lag.
HISTORY_COLUMNS = [
    "price_lag_24h",
    "price_lag_48h",
    "price_lag_168h",
    "price_rolling_24h_avg",
    "price_rolling_7d_avg",
]
BASELINE_COLUMN = "price_lag_24h"


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Preprocessing, median imputation of remaining nulls, then Ridge regression."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("impute", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def evaluate_pipeline(
    pipeline: Pipeline, df: pl.DataFrame, splits: list[Split], name: str
) -> pl.DataFrame:
    """Fit a fresh copy of `pipeline` on each split's training rows, score it on the rest."""
    rows: list[dict[str, str | int | float]] = []
    for split in splits:
        train, evaluation = apply_split(df, split)
        train = train.drop_nulls([TARGET_COLUMN, *HISTORY_COLUMNS])
        evaluation = evaluation.drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])

        model = clone(pipeline)  # an unfitted copy: nothing carries over between splits
        model.fit(
            train.select(MODEL_FEATURE_COLUMNS).to_pandas(),
            train[TARGET_COLUMN].to_numpy(),
        )
        predicted = model.predict(evaluation.select(MODEL_FEATURE_COLUMNS).to_pandas())

        model_scores = score(evaluation[TARGET_COLUMN], pl.Series(predicted))
        baseline_scores = score(evaluation[TARGET_COLUMN], evaluation[BASELINE_COLUMN])
        rows.append(
            {
                "split": split.name,
                "model": name,
                "train_rows": train.height,
                "eval_rows": evaluation.height,
                "mae": round(model_scores["mae"], 1),
                "rmse": round(model_scores["rmse"], 1),
                "yesterday_mae": round(baseline_scores["mae"], 1),
                "relative_mae": round(relative_mae(model_scores["mae"], baseline_scores["mae"]), 3),
            }
        )
    return pl.DataFrame(rows)
