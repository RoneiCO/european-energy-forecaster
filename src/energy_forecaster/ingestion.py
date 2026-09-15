import logging
import os
import time
from collections.abc import Callable

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from typing import TypeVar

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", pd.Series, pd.DataFrame)

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
    return prices.resample("1h").mean() # standardize to hourly resolution.


def fetch_all_zone_prices(
    zones: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch hourly day-ahead prices for every zone in `zones`, concatenated into one table.

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
        prices = _fetch_with_retry(fetch_day_ahead_prices, zone_code, start, end, max_retries)
        if prices is None:
            continue
        zone_df = prices.rename("price_eur_mwh").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone_name
        zone_df["country"] = zone_code.split("_")[0]
        frames.append(zone_df)
    return pd.concat(frames).reset_index(names="timestamp")


def fetch_all_zone_load(
    zones: dict[str, str], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
) -> pd.DataFrame:
    """Fetch hourly load (demand) for every zone in `zones`, concatenated into one table.

    Args:
        zones: A dictionary mapping ENTSO-E bidding zone codes to human-readable names.
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).
        max_retries: Maximum number of retries for fetching load in case of failure.

    Returns:
        A DataFrame containing hourly load values in MW for all specified zones,

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """

    frames = []
    for zone_code, zone_name in zones.items():
        load = _fetch_with_retry(fetch_load, zone_code, start, end, max_retries)
        if load is None:
            continue
        zone_df = load.rename("load_mw").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone_name
        zone_df["country"] = zone_code.split("_")[0]
        frames.append(zone_df)
    return pd.concat(frames).reset_index(names="timestamp")


def fetch_all_zone_generation(
    zones: dict[str, str], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
) -> pd.DataFrame:
    """Fetch hourly generation by source for every zone, concatenated with missing techs as 0.

    Args:
        zones: A dictionary mapping ENTSO-E bidding zone codes to human-readable names.
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).
        max_retries: Maximum number of retries for fetching generation data in case of failure.

    Returns:
        A DataFrame containing hourly generation mix values in MW for all specified zones.
        The columns represent different generation types (e.g., solar, wind, nuclear, etc.).

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    frames = []
    for zone_code, zone_name in zones.items():
        generation = _fetch_with_retry(fetch_generation_mix, zone_code, start, end, max_retries)
        if generation is None:
            continue
        generation = generation.rename_axis("timestamp").reset_index()
        generation["zone"] = zone_code
        generation["zone_name"] = zone_name
        generation["country"] = zone_code.split("_")[0]
        frames.append(generation)

    combined = pd.concat(frames, ignore_index=True)
    tech_cols = [
        c for c in combined.columns if c not in ("timestamp", "zone", "zone_name", "country")
    ]
    combined[tech_cols] = combined[tech_cols].fillna(0) # Fill missing generation types with 0 MW
    return combined


def fetch_load(
    zone_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Fetch electricity load for a bidding zone, resampled to hourly.

    Args:
        zone_code: ENTSO-E bidding zone code, e.g. "SE_4"
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).

    Returns:
        A Series containing hourly load values in MW for the specified zone.

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    client = get_client()
    load = client.query_load(zone_code, start=start, end=end)
    return load.iloc[:, 0].resample("1h").mean()


def fetch_generation_mix(
    zone_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Fetch electricity generation mix for a bidding zone, resampled to hourly.

    Args:
        zone_code: ENTSO-E bidding zone code, e.g. "SE_4"
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).

    Returns:
        A DataFrame containing hourly generation mix values in MW for the specified zone.
        The columns represent different generation types (e.g., solar, wind, nuclear, etc.).

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    client = get_client()
    generation = client.query_generation(zone_code, start=start, end=end)
    # entsoe-py sometimes labels columns with two levels (e.g., the generation type, and whether it's "Actual Aggregated" output vs. "Actual Consumption"
    # For now, we will just take the first level of the column names to simplify the DataFrame.
    if isinstance(generation.columns, pd.MultiIndex):
        generation.columns = generation.columns.get_level_values(0)
    return generation.resample("1h").mean()


def _fetch_with_retry(
    fetch_fn: Callable[[str, pd.Timestamp, pd.Timestamp], T],
    zone_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_retries: int = 3,
) -> T | None:
    """Call fetch_fn for a zone, retrying with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn(zone_code, start, end)
        except (requests.exceptions.RequestException, NoMatchingDataError) as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, zone_code, exc)
            if attempt < max_retries:
                time.sleep(2**attempt)
    logger.error("Giving up on %s after %d attempts", zone_code, max_retries)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    start = pd.Timestamp("2026-09-01", tz="Europe/Stockholm")
    end = pd.Timestamp("2026-09-04", tz="Europe/Stockholm")

    all_prices = fetch_all_zone_prices(NORDIC_ZONES, start, end)
    print(all_prices.head())
    print(all_prices["zone"].value_counts())

    load = fetch_load("SE_4", start, end)
    print(load.head())

    generation = fetch_generation_mix("SE_4", start, end)
    print(generation.head())
    print(generation.columns.tolist())
