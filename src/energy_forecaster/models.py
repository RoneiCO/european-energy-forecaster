from collections.abc import Sequence

import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from energy_forecaster.metrics import relative_mae, score
from energy_forecaster.preprocessing import (
    ALL_FEATURE_COLUMNS,
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


def predict_split(
    pipeline: Pipeline, df: pl.DataFrame, split: Split, predict_change: bool = False
) -> tuple[pl.DataFrame, int]:
    """Fit a fresh copy of `pipeline` on a split's training rows and predict its evaluation rows.

    Returns one row per evaluation hour (timestamp, zone, actual, yesterday, predicted)
    and the number of training rows used.
    """
    train, evaluation = apply_split(df, split)
    train = train.drop_nulls([TARGET_COLUMN, *HISTORY_COLUMNS])
    evaluation = evaluation.drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])

    y_train = train[TARGET_COLUMN].to_numpy()
    if predict_change:
        y_train = y_train - train[BASELINE_COLUMN].to_numpy()

    model = clone(pipeline)  # an unfitted copy: nothing carries over between splits
    model.fit(train.select(ALL_FEATURE_COLUMNS).to_pandas(), y_train)
    predicted = model.predict(evaluation.select(ALL_FEATURE_COLUMNS).to_pandas())
    if predict_change:
        predicted = predicted + evaluation[BASELINE_COLUMN].to_numpy()

    predictions = evaluation.select(
        "timestamp",
        "zone",
        pl.col(TARGET_COLUMN).alias("actual"),
        pl.col(BASELINE_COLUMN).alias("yesterday"),
    ).with_columns(pl.Series("predicted", predicted))
    return predictions, train.height


def evaluate_pipeline(
    pipeline: Pipeline,
    df: pl.DataFrame,
    splits: list[Split],
    name: str,
    predict_change: bool = False,
) -> pl.DataFrame:
    """Score a pipeline on each split: MAE and RMSE, absolute and relative to yesterday."""
    rows: list[dict[str, str | int | float]] = []
    for split in splits:
        predictions, train_rows = predict_split(pipeline, df, split, predict_change)
        model_scores = score(predictions["actual"], predictions["predicted"])
        baseline_scores = score(predictions["actual"], predictions["yesterday"])
        rows.append(
            {
                "split": split.name,
                "model": name,
                "train_rows": train_rows,
                "eval_rows": predictions.height,
                "mae": round(model_scores["mae"], 1),
                "rmse": round(model_scores["rmse"], 1),
                "rmse_to_mae": round(model_scores["rmse"] / model_scores["mae"], 2),
                "relative_mae": round(relative_mae(model_scores["mae"], baseline_scores["mae"]), 3),
                "relative_rmse": round(model_scores["rmse"] / baseline_scores["rmse"], 3),
            }
        )
    return pl.DataFrame(rows)


def build_xgboost_pipeline(
    n_estimators: int = 400,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    min_child_weight: float = 1.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    extra_numeric: Sequence[str] = (),
    drop_numeric: Sequence[str] = (),
) -> Pipeline:
    """Preprocessing, then gradient-boosted trees trained to minimize absolute error.

    No imputer: XGBoost handles missing values natively, learning at each split
    which direction rows with a missing value should go.
    """
    model = XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        tree_method="hist",
        n_jobs=-1,
        random_state=0,
    )
    return Pipeline(
        [("preprocess", build_preprocessor(extra_numeric, drop_numeric)), ("model", model)]
    )
