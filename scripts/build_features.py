import logging

from energy_forecaster.features import FEATURE_TABLE_PATH, build_feature_table

logger = logging.getLogger(__name__)


def main() -> None:
    df = build_feature_table()
    FEATURE_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(FEATURE_TABLE_PATH)
    logger.info("Saved %d rows, %d columns to %s", df.height, df.width, FEATURE_TABLE_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
