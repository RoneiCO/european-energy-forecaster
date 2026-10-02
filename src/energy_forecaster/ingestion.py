import logging
import os
import time
from collections.abc import Callable

import pandas as pd
import requests
from dotenv import load_dotenv
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from energy_forecaster.config import Zone

load_dotenv()

logger = logging.getLogger(__name__)

_PRICE_CHUNK_DAYS = 150
_PRICE_CHUNK_PADDING = pd.Timedelta(days=2)


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
    """Fetch day-ahead prices for a bidding zone in short windows, resampled to hourly."""
    client = get_client()
    pieces = [
        client.query_day_ahead_prices(zone_code, start=w_start, end=w_end)
        for w_start, w_end in _price_windows(start, end)
    ]
    prices = pd.concat(pieces)
    prices = prices[~prices.index.duplicated(keep="first")].sort_index()
    prices = prices[(prices.index >= start) & (prices.index <= end)]
    _assert_no_interior_gaps(prices, zone_code)
    return prices.resample("1h").mean()


def fetch_all_zone_prices(
    zones: dict[str, Zone],
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
    for zone_code, zone in zones.items():
        prices = _fetch_with_retry(fetch_day_ahead_prices, zone_code, start, end, max_retries)
        if prices is None:
            continue
        zone_df = prices.rename("price_eur_mwh").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone.name
        zone_df["country"] = zone.country
        frames.append(zone_df)
    return _normalize_timestamp_column(pd.concat(frames).reset_index(names="timestamp"))


def fetch_all_zone_load(
    zones: dict[str, Zone], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
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
    for zone_code, zone in zones.items():
        load = _fetch_with_retry(fetch_load, zone_code, start, end, max_retries)
        if load is None:
            continue
        zone_df = load.rename("load_mw").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone.name
        zone_df["country"] = zone.country
        frames.append(zone_df)
    return _normalize_timestamp_column(pd.concat(frames).reset_index(names="timestamp"))


def fetch_all_zone_generation(
    zones: dict[str, Zone], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
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
    for zone_code, zone in zones.items():
        generation = _fetch_with_retry(fetch_generation_mix, zone_code, start, end, max_retries)
        if generation is None:
            continue
        generation = generation.rename_axis("timestamp").reset_index()
        generation["zone"] = zone_code
        generation["zone_name"] = zone.name
        generation["country"] = zone.country
        frames.append(generation)

    combined = pd.concat(frames, ignore_index=True)
    tech_cols = [
        c for c in combined.columns if c not in ("timestamp", "zone", "zone_name", "country")
    ]
    combined[tech_cols] = combined[tech_cols].fillna(0)  # Fill missing generation types with 0 MW
    return _normalize_timestamp_column(combined)


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
    """Fetch hourly generation by source (MW), keeping only power delivered to the grid.

    Drops the "Actual Consumption" subtype (pumped-storage charging draw) — out of
    scope for a generation-mix feature. Net storage behavior could be a useful
    feature later, but isn't needed for now.

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

    generation = _flatten_generation_columns(generation)
    generation = _coalesce_duplicate_columns(generation)

    return generation.resample("1h").mean()


def _fetch_with_retry[T: (pd.Series, pd.DataFrame)](
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


def _normalize_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the timestamp column to one consistent timezone-aware dtype.

    entsoe-py returns timestamps as a fixed UTC offset that changes across
    DST transitions (+01:00 in winter, +02:00 in summer for Stockholm). A
    column spanning both can't be stored as a uniform datetime64[ns, tz]
    dtype and silently degrades to `object` dtype instead, breaking `.dt`
    accessors. Parsing with utc=True first forces one representation; then
    tz_convert restores local Stockholm time for correct calendar semantics.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Europe/Stockholm")
    return df


def _flatten_generation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize generation columns to plain technology names.

    entsoe-py sometimes returns combined (technology, subtype) column labels as a
    true pandas MultiIndex, and sometimes as a flat Index whose individual elements
    are tuples -- inconsistently, depending on the zone and which technologies it
    reports. Checking `isinstance(col, tuple)` per column catches both shapes;
    checking `isinstance(df.columns, pd.MultiIndex)` only catches the first one.
    Keeps only "Actual Aggregated" (power delivered to the grid), dropping
    "Actual Consumption" (e.g. pumped-storage charging draw).
    """
    keep_cols, new_names = [], []
    for col in df.columns:
        if isinstance(col, tuple):
            technology, subtype = col
            if subtype == "Actual Aggregated":
                keep_cols.append(col)
                new_names.append(technology)
        else:
            keep_cols.append(col)
            new_names.append(col)
    df = df[keep_cols]
    df.columns = new_names
    return df


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate-named columns into one, keeping the first non-null value per row.

    entsoe-py sometimes reports the same technology under two parallel column
    representations (e.g. a bare "Biomass" and a tupled ("Biomass", "Actual
    Aggregated")) for the same zone/year, with only one of the two ever actually
    populated. Raises if both versions are ever non-null on the same row, since
    that would mean the two sources genuinely disagree -- a real conflict, not a
    safe-to-merge duplicate.
    """
    if not df.columns.duplicated().any():
        return df

    for name in df.columns[df.columns.duplicated()].unique():
        subset = df.loc[:, df.columns == name]
        if (subset.notna().sum(axis=1) > 1).any():
            raise ValueError(f"Conflicting non-null values across duplicate '{name}' columns")

    return df.T.groupby(level=0).first().T


def fetch_load_forecast(zone_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch the day-ahead forecasted load (demand) for a bidding zone, resampled to hourly.

    Args:
        zone_code: ENTSO-E bidding zone code, e.g. "SE_4".
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).

    Returns:
        A Series containing hourly forecasted load values in MW for the specified zone.

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    client = get_client()
    forecast = client.query_load_forecast(zone_code, start=start, end=end)
    return forecast.iloc[:, 0].resample("1h").mean()


def fetch_wind_solar_forecast(
    zone_code: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Fetch the day-ahead forecasted wind/solar generation for a bidding zone, resampled to hourly.

    Args:
        zone_code: ENTSO-E bidding zone code, e.g. "SE_4".
        start: Start of the query window (must be timezone-aware).
        end: End of the query window (must be timezone-aware).

    Returns:
        A DataFrame containing hourly forecasted generation in MW for the specified zone,
        with columns named "forecast_<technology>_mw" (e.g. "forecast_solar_mw",
        "forecast_wind_onshore_mw").

    Raises:
        KeyError: If ENTSOE_API_KEY is not set in the environment.
    """
    client = get_client()
    forecast = client.query_wind_and_solar_forecast(zone_code, start=start, end=end)
    forecast = _coalesce_duplicate_columns(forecast)
    forecast = forecast.resample("1h").mean()
    forecast.columns = [f"forecast_{c.lower().replace(' ', '_')}_mw" for c in forecast.columns]
    return forecast


def fetch_all_zone_load_forecast(
    zones: dict[str, Zone], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
) -> pd.DataFrame:
    """Fetch hourly day-ahead forecasted load for every zone, concatenated into one table."""
    frames = []
    for zone_code, zone in zones.items():
        forecast = _fetch_with_retry(fetch_load_forecast, zone_code, start, end, max_retries)
        if forecast is None:
            continue
        zone_df = forecast.rename("load_forecast_mw").to_frame()
        zone_df["zone"] = zone_code
        zone_df["zone_name"] = zone.name
        zone_df["country"] = zone.country
        frames.append(zone_df)
    return _normalize_timestamp_column(pd.concat(frames).reset_index(names="timestamp"))


def fetch_all_zone_wind_solar_forecast(
    zones: dict[str, Zone], start: pd.Timestamp, end: pd.Timestamp, max_retries: int = 3
) -> pd.DataFrame:
    """Fetch hourly day-ahead forecasted wind/solar generation for every zone, missing techs as 0."""
    frames = []
    for zone_code, zone in zones.items():
        forecast = _fetch_with_retry(fetch_wind_solar_forecast, zone_code, start, end, max_retries)
        if forecast is None:
            continue
        forecast = forecast.rename_axis("timestamp").reset_index()
        forecast["zone"] = zone_code
        forecast["zone_name"] = zone.name
        forecast["country"] = zone.country
        frames.append(forecast)

    combined = pd.concat(frames, ignore_index=True)
    tech_cols = [
        c for c in combined.columns if c not in ("timestamp", "zone", "zone_name", "country")
    ]
    combined[tech_cols] = combined[tech_cols].fillna(0)
    return _normalize_timestamp_column(combined)


def _price_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end] into windows of at most _PRICE_CHUNK_DAYS days, each padded.

    A prices request spanning a full calendar year silently loses one point (the first
    point of the day before its one-year mark). Requests this short were tested clean,
    and the padding keeps any loss at a window's edge, where trimming discards it.
    """
    windows = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(days=_PRICE_CHUNK_DAYS), end)
        windows.append((chunk_start - _PRICE_CHUNK_PADDING, chunk_end + _PRICE_CHUNK_PADDING))
        chunk_start = chunk_end
    return windows


def _assert_no_interior_gaps(series: pd.Series, label: str) -> None:
    """Raise if a point is missing between two present points, at any cadence.

    A lost point makes one step between neighbors larger than the steps on both sides
    of it. The switch from hourly to 15-minute data is not flagged: its step is not
    larger than the step before it.
    """
    steps = series.index.to_series().diff()
    bump = (steps > steps.shift(1)) & (steps > steps.shift(-1))
    if bump.any():
        examples = [str(t) for t in series.index[bump.to_numpy()][:3]]
        raise ValueError(f"{label}: points missing before {', '.join(examples)}")
