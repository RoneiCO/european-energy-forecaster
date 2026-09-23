import logging
from pathlib import Path

import pandas as pd

from energy_forecaster.ingestion import (
    NORDIC_ZONES,
    fetch_all_zone_generation,
    fetch_all_zone_load,
    fetch_all_zone_prices,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def pull_and_cache_year(data_type: str, fetch_fn, zones: dict[str, str], year: int) -> None:
    """Fetch one calendar year of data for all zones and cache it as Parquet, unless already cached."""
    out_path = RAW_DATA_DIR / data_type / f"year={year}.parquet"  # One file per data type per year.
    current_year = pd.Timestamp.now().year
    if out_path.exists() and year < current_year:
        logger.info("Skipping %s %d, already cached at %s", data_type, year, out_path)
        return

    tz = "Europe/Stockholm"
    start = pd.Timestamp(f"{year}-01-01", tz=tz)
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz=tz), pd.Timestamp.now(tz=tz))

    logger.info("Fetching %s for %d...", data_type, year)
    df = fetch_fn(zones, start, end)
    df = df[df["timestamp"].dt.year == year]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def run_historical_pull(zones: dict[str, str] = NORDIC_ZONES, first_year: int = 2021) -> None:
    """Pull and cache prices, load, and generation for every year from first_year through now."""
    current_year = pd.Timestamp.now().year
    fetchers = {
        "prices": fetch_all_zone_prices,
        "load": fetch_all_zone_load,
        "generation": fetch_all_zone_generation,
    }
    for data_type, fetch_fn in fetchers.items():
        for year in range(first_year, current_year + 1):
            pull_and_cache_year(data_type, fetch_fn, zones, year)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_historical_pull()
