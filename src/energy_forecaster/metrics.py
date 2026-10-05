import polars as pl
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def score(y_true: pl.Series, y_pred: pl.Series) -> dict[str, float]:
    """MAE and RMSE in EUR/MWh. Rows with a missing value must be dropped beforehand."""
    true, pred = y_true.to_numpy(), y_pred.to_numpy()
    return {
        "mae": mean_absolute_error(true, pred),
        "rmse": root_mean_squared_error(true, pred),
    }


def relative_mae(model_mae: float, baseline_mae: float) -> float:
    """Model MAE divided by the baseline's: below 1 means the model beats the baseline."""
    return model_mae / baseline_mae
