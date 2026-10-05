import polars as pl
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

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
    pipeline: Pipeline,
    df: pl.DataFrame,
    splits: list[Split],
    name: str,
    predict_change: bool = False,
) -> pl.DataFrame:
    """Fit a fresh copy of `pipeline` on each split's training rows, score it on the rest.

    With predict_change, the model learns price minus yesterday's price, and
    yesterday's price is added back to its predictions.
    """
    rows: list[dict[str, str | int | float]] = []
    for split in splits:
        train, evaluation = apply_split(df, split)
        train = train.drop_nulls([TARGET_COLUMN, *HISTORY_COLUMNS])
        evaluation = evaluation.drop_nulls([TARGET_COLUMN, BASELINE_COLUMN])

        y_train = train[TARGET_COLUMN].to_numpy()
        if predict_change:
            y_train = y_train - train[BASELINE_COLUMN].to_numpy()

        model = clone(pipeline)  # an unfitted copy: nothing carries over between splits
        model.fit(train.select(MODEL_FEATURE_COLUMNS).to_pandas(), y_train)
        predicted = model.predict(evaluation.select(MODEL_FEATURE_COLUMNS).to_pandas())
        if predict_change:
            predicted = predicted + evaluation[BASELINE_COLUMN].to_numpy()

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
                "rmse_to_mae": round(model_scores["rmse"] / model_scores["mae"], 2),
                "relative_mae": round(relative_mae(model_scores["mae"], baseline_scores["mae"]), 3),
                "relative_rmse": round(model_scores["rmse"] / baseline_scores["rmse"], 3),
            }
        )
    return pl.DataFrame(rows)


def build_xgboost_pipeline(
    n_estimators: int = 400, learning_rate: float = 0.05, max_depth: int = 6
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
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=0,
    )
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
