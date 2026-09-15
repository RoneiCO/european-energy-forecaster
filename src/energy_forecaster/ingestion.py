import logging
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

load_dotenv()

logger = logging.getLogger(__name__)

NORDIC_ZONES = {
    "SE_1": "Sweden (Luleå)",
    "SE_2": "Sweden (Sundsvall)",
    "SE_3": "Sweden (Stockholm)",
    "SE_4": "Sweden (Malmö)",
    "NO_1": "Norway (Oslo)",
    "NO_2": "Norway (Kristiansand)",
    "NO_3": "Norway (Trondheim)",
    "NO_4": "Norway (Tromsø)",
    "NO_5": "Norway (Bergen)",
    "DK_1": "Denmark (West)",
    "DK_2": "Denmark (East)",
    "FI": "Finland",
}


def get_client() -> EntsoePandasClient:
    """Create an ENTSO-E client using the API key from the environment.

    Args:
        None

    Returns:
        EntsoePandasClient: An instance of the ENTSO-E client.

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    api_key = os.environ["ENTSOE_API_KEY"]
    return EntsoePandasClient(api_key=api_key)


def fetch_day_ahead_prices(zone_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch day-ahead electricity prices for a bidding zone, resampled to hourly.

    Args:
        zone_code: ENTSO-E bidding zone code, e.g. "SE_4".
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).

    Returns:
        Hourly prices in EUR/MWh, indexed by timestamp. Source data from
        before Oct 2025 is hourly; from Oct 2025 onward it's 15-minute and
        gets averaged into hourly buckets here for a consistent series.

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    client = get_client()
    prices = client.query_day_ahead_prices(zone_code, start=start, end=end)
    return prices.resample("1h").mean()


def fetch_all_zone_prices(
    zones: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch day-ahead electricity prices for all specified bidding zones, resampled to hourly.

    Args:
        zones: A dictionary mapping ENTSO-E bidding zone codes to human-readable names.
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).
        max_retries: Maximum number of retries for fetching prices in case of failure.

    Returns:
        A DataFrame containing hourly prices in EUR/MWh for all specified zones,
        timestamp index converted to a normal column, with additional columns for zone code, zone name, and country.

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    frames = []
    for zone_code, zone_name in zones.items():
        prices = None
        for attempt in range(1, max_retries + 1):
            try:
                prices = fetch_day_ahead_prices(zone_code, start, end)
                break
            except (requests.exceptions.RequestException, NoMatchingDataError) as exc:
                logger.warning(
                    "Attempt %d/%d failed for %s: %s", attempt, max_retries, zone_code, exc
                )
                if attempt < max_retries:
                    time.sleep(2**attempt)

        if prices is None:
            logger.error("Giving up on %s after %d attempts", zone_code, max_retries)
            continue

        zone_df = prices.rename("price_eur_mwh").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone_name
        zone_df["country"] = zone_code.split("_")[0]
        frames.append(zone_df)

    return pd.concat(frames).reset_index(names="timestamp")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start = pd.Timestamp("2026-09-01", tz="Europe/Stockholm")
    end = pd.Timestamp("2026-09-04", tz="Europe/Stockholm")
    all_prices = fetch_all_zone_prices(NORDIC_ZONES, start, end)
    print(all_prices.head())
    print(all_prices["zone"].value_counts())
