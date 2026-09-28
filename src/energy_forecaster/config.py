from dataclasses import dataclass

# Project-wide reference clock: cuts the data into calendar-year files and is the
# default display zone of `timestamp`. It does NOT drive per-zone local-time
# features (see Zone.timezone). Changing it after data is cached shifts the year
# boundaries, so delete data/raw and re-pull if you change it.
REFERENCE_TIMEZONE = "Europe/Stockholm"


@dataclass(frozen=True)
class Zone:
    name: str
    country: str  # ISO 3166-1 alpha-2 code, used for public-holiday calendars
    timezone: str  # IANA name, used for local-time features (hour, weekday, holidays)


EUROPEAN_ZONES: dict[str, Zone] = {
    "SE_1": Zone("Sweden (Luleå)", "SE", "Europe/Stockholm"),
    "SE_2": Zone("Sweden (Sundsvall)", "SE", "Europe/Stockholm"),
    "SE_3": Zone("Sweden (Stockholm)", "SE", "Europe/Stockholm"),
    "SE_4": Zone("Sweden (Malmö)", "SE", "Europe/Stockholm"),
    "NO_1": Zone("Norway (Oslo)", "NO", "Europe/Oslo"),
    "NO_2": Zone("Norway (Kristiansand)", "NO", "Europe/Oslo"),
    "NO_3": Zone("Norway (Trondheim)", "NO", "Europe/Oslo"),
    "NO_4": Zone("Norway (Tromsø)", "NO", "Europe/Oslo"),
    "NO_5": Zone("Norway (Bergen)", "NO", "Europe/Oslo"),
    "DK_1": Zone("Denmark (West)", "DK", "Europe/Copenhagen"),
    "DK_2": Zone("Denmark (East)", "DK", "Europe/Copenhagen"),
    "FI": Zone("Finland", "FI", "Europe/Helsinki"),
}
