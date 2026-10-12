import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import sklearn
import xgboost

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_xgboost_pipeline, fit_on_split
from energy_forecaster.preprocessing import (
    ADOPTED_EXTRA_FEATURES,
    ALL_FEATURE_COLUMNS,
    EXTERNAL_FEATURES,
)
from energy_forecaster.splits import TEST_SPLIT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
TEST_PREDICTIONS = PROJECT_ROOT / "data" / "processed" / "test_predictions_full.parquet"
FEATURES = [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]


def main() -> None:
    df = load_feature_table()
    model, evaluation, train_rows = fit_on_split(
        build_xgboost_pipeline(extra_numeric=FEATURES), df, TEST_SPLIT
    )

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "xgb_full.joblib"
    joblib.dump(model, model_path)

    card = {
        "trained_on": f"{TEST_SPLIT.train_start} up to, not including, {TEST_SPLIT.train_end}",
        "train_rows": train_rows,
        "extra_numeric_features": FEATURES,
        "reported_test_2026": {  # copied from scripts/final_test.py, run once
            "relative_mae": 0.776,
            "interval_95": [0.737, 0.815],
            "strict_model_relative_mae": 0.877,
        },
        "library_versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "polars": pl.__version__,
        },
    }
    (MODEL_DIR / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    print(f"saved {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Tie-out: the reloaded model must reproduce the predictions stored by final_test.py
    reloaded = joblib.load(model_path)
    again = reloaded.predict(evaluation.select(ALL_FEATURE_COLUMNS).to_pandas())
    stored = pl.read_parquet(TEST_PREDICTIONS)["predicted"].to_numpy()
    print("max |reloaded - stored|:", float(np.abs(again - stored).max()))


if __name__ == "__main__":
    main()
