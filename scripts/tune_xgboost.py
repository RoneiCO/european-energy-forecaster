import random
from pathlib import Path
from typing import Any

import polars as pl

from energy_forecaster.features import load_feature_table
from energy_forecaster.models import build_xgboost_pipeline, evaluate_pipeline
from energy_forecaster.preprocessing import ADOPTED_EXTRA_FEATURES, EXTERNAL_FEATURES
from energy_forecaster.splits import CV_SPLITS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "processed" / "tuning_results.csv"

DEFAULTS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1,
}
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 400, 800],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 10, 50],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [1, 10],
}
N_CONFIGS = 12
FEATURES = [*ADOPTED_EXTRA_FEATURES, *EXTERNAL_FEATURES]


def sample_configs(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """The defaults first, as the reference, then `n` distinct random configurations."""
    rng = random.Random(seed)
    configs = [dict(DEFAULTS)]
    seen = {tuple(DEFAULTS.values())}
    while len(configs) < n + 1:
        config = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
        if tuple(config.values()) not in seen:
            seen.add(tuple(config.values()))
            configs.append(config)
    return configs


def main() -> None:
    df = load_feature_table()
    rows: list[dict[str, Any]] = []
    for i, config in enumerate(sample_configs(N_CONFIGS)):
        results = evaluate_pipeline(
            build_xgboost_pipeline(**config, extra_numeric=FEATURES), df, CV_SPLITS, f"cfg_{i}"
        )
        by_split = dict(zip(results["split"], results["relative_mae"], strict=True))
        mean = round(sum(by_split.values()) / len(by_split), 4)
        rows.append({"config": i, **config, **by_split, "mean": mean})
        print(rows[-1], flush=True)
        pl.DataFrame(rows).write_csv(RESULTS_PATH)  # saved after every config, so nothing is lost

    with pl.Config(tbl_rows=20, tbl_cols=-1):
        print(pl.DataFrame(rows).sort("mean"))


if __name__ == "__main__":
    main()
