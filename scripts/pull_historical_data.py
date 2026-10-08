import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from energy_forecaster.config import EUROPEAN_ZONES, EXTERNAL_ZONES, REFERENCE_TIMEZONE, Zone
from energy_forecaster.ingestion import (
    fetch_all_zone_generation,
    fetch_all_zone_load,
    fetch_all_zone_load_forecast,
    fetch_all_zone_prices,
    fetch_all_zone_wind_solar_forecast,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def pull_and_cache_year(data_type: str, fetch_fn, zones: dict[str, Zone], year: int) -> None:
    """Fetch one calendar year of data for all zones and cache it as Parquet, unless already cached."""
    out_path = RAW_DATA_DIR / data_type / f"year={year}.parquet"  # One file per data type per year.
    current_year = pd.Timestamp.now().year
    if out_path.exists() and year < current_year:
        logger.info("Skipping %s %d, already cached at %s", data_type, year, out_path)
        return

    tz = REFERENCE_TIMEZONE
    start = pd.Timestamp(f"{year}-01-01", tz=tz)
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz=tz), pd.Timestamp.now(tz=tz))

    logger.info("Fetching %s for %d...", data_type, year)
    df = fetch_fn(zones, start, end)
    df = df[df["timestamp"].dt.year == year]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)


def run_historical_pull(zones: dict[str, Zone] = EUROPEAN_ZONES, first_year: int = 2021) -> None:
    """Pull and cache prices, load, and generation for every year from first_year through now."""
    current_year = pd.Timestamp.now().year
    fetchers: dict[str, tuple[Callable[..., pd.DataFrame], dict[str, Zone]]] = {
        "prices": (fetch_all_zone_prices, zones),
        "load": (fetch_all_zone_load, zones),
        "generation": (fetch_all_zone_generation, zones),
        "load_forecast": (fetch_all_zone_load_forecast, zones),
        "wind_solar_forecast": (fetch_all_zone_wind_solar_forecast, zones),
        "external_prices": (fetch_all_zone_prices, EXTERNAL_ZONES),
        "external_load_forecast": (fetch_all_zone_load_forecast, EXTERNAL_ZONES),
        "external_wind_solar_forecast": (fetch_all_zone_wind_solar_forecast, EXTERNAL_ZONES),
        "external_generation": (fetch_all_zone_generation, EXTERNAL_ZONES),
    }
    for data_type, (fetch_fn, fetch_zones) in fetchers.items():
        for year in range(first_year, current_year + 1):
            pull_and_cache_year(data_type, fetch_fn, fetch_zones, year)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_historical_pull()
