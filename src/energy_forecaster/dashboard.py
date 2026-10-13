"""Pure helpers for the dashboard, kept free of Streamlit so they can be tested."""

from datetime import date

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from energy_forecaster.config import REFERENCE_TIMEZONE
from energy_forecaster.preprocessing import ALL_FEATURE_COLUMNS


def select_day(df: pl.DataFrame, day: date) -> pl.DataFrame:
    """All zones' rows for one calendar day on the reference (Stockholm) clock."""
    local_date = pl.col("timestamp").dt.convert_time_zone(REFERENCE_TIMEZONE).dt.date()
    return df.filter(local_date == day).sort("zone", "timestamp")


def predict(model: Pipeline, df: pl.DataFrame) -> np.ndarray:
    prediction = model.predict(df.select(ALL_FEATURE_COLUMNS).to_pandas())
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    return np.asarray(prediction)


def day_summary(actual, yesterday, predicted) -> dict[str, float]:
    """MAE of the baseline and the model on rows where all three values exist."""
    actual, yesterday, predicted = (
        np.asarray(x, dtype=float) for x in (actual, yesterday, predicted)
    )
    mask = ~(np.isnan(actual) | np.isnan(yesterday) | np.isnan(predicted))
    if not mask.any():
        raise ValueError("No rows with an actual, a baseline and a forecast")
    mae_yesterday = float(np.mean(np.abs(actual[mask] - yesterday[mask])))
    mae_model = float(np.mean(np.abs(actual[mask] - predicted[mask])))
    relative = mae_model / mae_yesterday if mae_yesterday > 0 else float("nan")
    return {"mae_yesterday": mae_yesterday, "mae_model": mae_model, "relative_mae": relative}


def complete_days(df: pl.DataFrame, min_hours: int = 23) -> list[date]:
    """Local days with at least `min_hours` distinct hours (23 admits the spring DST day)."""
    local_date = pl.col("timestamp").dt.convert_time_zone(REFERENCE_TIMEZONE).dt.date()
    hours = df.group_by(local_date.alias("day")).agg(pl.col("timestamp").n_unique().alias("hours"))
    return hours.filter(pl.col("hours") >= min_hours).sort("day")["day"].to_list()


def zone_context(predictions: pl.DataFrame, zone: str) -> dict[str, float]:
    """Pooled relative MAE and the share of days the model beat 'yesterday', for one zone.

    `predictions` needs columns timestamp, zone, actual, predicted, yesterday.
    """
    local_date = pl.col("timestamp").dt.convert_time_zone(REFERENCE_TIMEZONE).dt.date()
    rows = predictions.filter(pl.col("zone") == zone).with_columns(
        day=local_date,
        model_err=(pl.col("actual") - pl.col("predicted")).abs(),
        base_err=(pl.col("actual") - pl.col("yesterday")).abs(),
    )
    if rows.is_empty():
        raise ValueError(f"No predictions for zone {zone}")
    daily = rows.group_by("day").agg(pl.col("model_err").mean(), pl.col("base_err").mean())
    return {
        "pooled_relative_mae": rows.select(
            pl.col("model_err").sum() / pl.col("base_err").sum()
        ).item(),
        "win_rate": daily.select((pl.col("model_err") < pl.col("base_err")).mean()).item(),
        "days": float(daily.height),
    }
