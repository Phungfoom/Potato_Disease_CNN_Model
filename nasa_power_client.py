import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


@dataclass
class NasaPowerConfig:
    parameters: List[str]
    community: str = "AG"  # Agriculture
    format: str = "JSON"


DEFAULT_CONFIG = NasaPowerConfig(
    parameters=[
        "T2M",   # Air temperature at 2m
        "RH2M",  # Relative humidity at 2m
        "PRECTOT",  # Precipitation
        "ALLSKY_SFC_SW_DWN",  # Solar irradiance
    ]
)


def _build_datestring(date: datetime.date) -> str:
    return date.strftime("%Y%m%d")


def fetch_nasa_power_features(
    lat: float,
    lon: float,
    date: datetime.date,
    config: NasaPowerConfig = DEFAULT_CONFIG,
) -> Optional[Dict[str, float]]:
    """Fetch a small set of NASA POWER daily features for a given location and date."""
    start = _build_datestring(date)
    end = start

    params = {
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "parameters": ",".join(config.parameters),
        "community": config.community,
        "format": config.format,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        print(f"[NASA POWER] Request failed for ({lat}, {lon}, {date}): {exc}")
        return None

    try:
        daily_values = next(iter(data["properties"]["parameter"].values()))
        key = next(iter(daily_values))
        features: Dict[str, float] = {}
        for param in config.parameters:
            series = data["properties"]["parameter"].get(param)
            if series is None or key not in series:
                continue
            value = series[key]
            if value is None:
                continue
            features[param] = float(value)
        return features or None
    except Exception as exc:
        print(f"[NASA POWER] Failed to parse response for ({lat}, {lon}, {date}): {exc}")
        return None

