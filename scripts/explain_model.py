import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.base import clone

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import HISTORY_COLUMNS, build_xgboost_pipeline
from energy_forecaster.preprocessing import (
    ADOPTED_EXTRA_FEATURES,
    ALL_FEATURE_COLUMNS,
    EXTERNAL_FEATURES,
    TARGET_COLUMN,
)
from energy_forecaster.splits import CV_SPLITS, apply_split

FEATURES = [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]
SAMPLE_ROWS = 20_000


def feature_group(name: str) -> str:
    """Collapse one-hot zone columns into a single 'zone' group."""
    return "zone" if name.startswith("categorical__zone") else name.split("__", 1)[-1]


def main() -> None:
    df = load_feature_table()
    train, evaluation = apply_split(df, CV_SPLITS[-1])  # fit on 2021-2024, explain 2025
    train = train.drop_nulls([TARGET_COLUMN, *HISTORY_COLUMNS])
    evaluation = evaluation.sample(SAMPLE_ROWS, seed=0)

    pipeline = clone(build_xgboost_pipeline(extra_numeric=FEATURES))
    pipeline.fit(train.select(ALL_FEATURE_COLUMNS).to_pandas(), train[TARGET_COLUMN].to_numpy())

    preprocess, model = pipeline[:-1], pipeline[-1]
    names = list(preprocess.get_feature_names_out())
    X = preprocess.transform(evaluation.select(ALL_FEATURE_COLUMNS).to_pandas())

    contribs = model.get_booster().predict(xgb.DMatrix(X, feature_names=names), pred_contribs=True)
    shap_values, bias = contribs[:, :-1], contribs[:, -1]

    # Tie-out: the contributions plus the bias must rebuild the model's own prediction.
    rebuilt = shap_values.sum(axis=1) + bias
    print("max |rebuilt - predict|:", float(np.abs(rebuilt - model.predict(X)).max()))

    importance = (
        pl.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .with_columns(pl.col("feature").map_elements(feature_group, return_dtype=pl.String))
        .group_by("feature")
        .agg(pl.col("mean_abs_shap").sum().round(2))
        .sort("mean_abs_shap", descending=True)
        .with_columns(
            (pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum()).round(3).alias("share")
        )
    )
    with pl.Config(tbl_rows=50, tbl_cols=-1):
        print(importance)
    german = importance.filter(
        pl.col("feature").is_in(
            [
                "de_price_lag_24h",
                "de_load_forecast_mw",
                "de_forecast_renewable_mw",
                "de_forecast_renewable_share_of_load",
            ]
        )
    )
    print("German features share:", round(german["share"].sum(), 3))


if __name__ == "__main__":
    main()
