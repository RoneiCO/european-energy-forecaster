import logging
from pathlib import Path

from energy_forecaster.warehouse import get_connection

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "hourly_dataset.parquet"

QUERY = """
WITH combined AS (
    SELECT
        p.timestamp,
        p.zone,
        p.zone_name,
        p.country,
        p.price_eur_mwh,
        l.load_mw,
        g.* EXCLUDE (timestamp, zone, zone_name, country)
    FROM prices p
    INNER JOIN load_data l USING (zone, timestamp)
    INNER JOIN generation g USING (zone, timestamp)
),
filled AS (
    SELECT
        timestamp, zone, zone_name, country, price_eur_mwh, load_mw,
        COALESCE(COLUMNS(* EXCLUDE (timestamp, zone, zone_name, country, price_eur_mwh, load_mw)), 0)
    FROM combined
)
SELECT
    *,
    AVG(price_eur_mwh) OVER (
        PARTITION BY zone ORDER BY timestamp
        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
    ) AS price_rolling_24h_avg,
    AVG(price_eur_mwh) OVER (
        PARTITION BY zone ORDER BY timestamp
        ROWS BETWEEN 167 PRECEDING AND CURRENT ROW
    ) AS price_rolling_7d_avg
FROM filled
ORDER BY zone, timestamp
"""


def build_dataset() -> None:
    """Join prices, load, and generation into one hourly modeling table and persist it."""
    con = get_connection()
    result = con.sql(QUERY)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(str(OUTPUT_PATH))

    row_count = con.sql(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT_PATH}')").fetchone()
    if row_count is None:
        raise RuntimeError(f"Failed to count rows in {OUTPUT_PATH}")

    logger.info("Saved %d rows to %s", row_count[0], OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
