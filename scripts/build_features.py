import logging
from pathlib import Path

from energy_forecaster.features import build_feature_table

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_table.parquet"


def main() -> None:
    df = build_feature_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_PATH)
    logger.info("Saved %d rows, %d columns to %s", df.height, df.width, OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
