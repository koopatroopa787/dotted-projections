"""
backend/world.py
Joins across live planetary data — the part no single API can answer.

The design rule here is intersection, not blend. "Aurora probability >= 10%,
sun below the horizon, cloud cover < 40%" is a claim someone can walk outside
and check. A weighted composite of the same three numbers would look more
sophisticated and mean nothing, which is the exact failure this codebase's
sibling project was built to demonstrate.

Sources, all free and keyless:
  NOAA OVATION   aurora probability on a 1-degree global grid, refreshed ~5 min
  Open-Meteo     cloud cover and temperature at arbitrary points, bulk in one call
  solar geometry computed here — no API can be wrong about where the sun is
"""
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

AURORA_URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "DottedProjections/0.1 (https://dotte.duckdns.org)"}

AURORA_CACHE_SECONDS = 300      # NOAA regenerates every ~5 minutes
WEATHER_CACHE_SECONDS = 600
REQUEST_TIMEOUT = 30.0

# Open-Meteo takes comma-separated coordinates; keep the URL a sane length.
MAX_WEATHER_POINTS = 90

_cache: Dict[str, Any] = {}


# ── Solar geometry ────────────────────────────────────────────────────────────

def solar_elevation(lat: float, lon: float, when: Optional[datetime] = None) -> float:
    """
    Sun's altitude in degrees. Negative means below the horizon.

    NOAA's low-precision algorithm: good to about a tenth of a degree, which is
    far inside what "is it dark there" needs.
    """
    now = when or datetime.now(timezone.utc)
    day = now.timetuple().tm_yday
    hour = now.hour + now.minute / 60 + now.second / 3600

    frac = 2 * math.pi / 365 * (day - 1 + (hour - 12) / 24)
    eq_time = 229.18 * (
        0.000075 + 0.001868 * math.cos(frac) - 0.032077 * math.sin(frac)
        - 0.014615 * math.cos(2 * frac) - 0.040849 * math.sin(2 * frac)
    )
    decl = (
        0.006918 - 0.399912 * math.cos(frac) + 0.070257 * math.sin(frac)
        - 0.006758 * math.cos(2 * frac) + 0.000907 * math.sin(2 * frac)
        - 0.002697 * math.cos(3 * frac) + 0.00148 * math.sin(3 * frac)
    )

    time_offset = eq_time + 4 * lon
    true_solar = (hour * 60 + time_offset) % 1440
    hour_angle = math.radians(true_solar / 4 - 180)

    lat_r = math.radians(lat)
    cos_zenith = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    )
    return math.degrees(math.asin(max(-1.0, min(1.0, cos_zenith))))


def is_dark(lat: float, lon: float, when: Optional[datetime] = None,
            below: float = -6.0) -> bool:
    """Darker than civil twilight — the point at which faint aurora is visible."""
    return solar_elevation(lat, lon, when) < below


# ── Sources ───────────────────────────────────────────────────────────────────

async def fetch_aurora() -> Tuple[List[List[int]], Optional[str]]:
    """NOAA's aurora grid as [lon, lat, probability], plus its forecast time."""
    hit = _cache.get("aurora")
    if hit and time.time() - hit["at"] < AURORA_CACHE_SECONDS:
        return hit["grid"], hit["forecast_time"]
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as c:
            res = await c.get(AURORA_URL)
            res.raise_for_status()
            payload = res.json()
    except Exception:
        return (hit["grid"], hit["forecast_time"]) if hit else ([], None)

    grid = payload.get("coordinates") or []
    forecast_time = payload.get("Forecast Time")
    _cache["aurora"] = {"at": time.time(), "grid": grid, "forecast_time": forecast_time}
    return grid, forecast_time


async def fetch_conditions(points: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """Cloud cover and temperature at up to MAX_WEATHER_POINTS locations, one call."""
    if not points:
        return []
    points = points[:MAX_WEATHER_POINTS]
    key = "wx:" + ",".join(f"{a:.1f}:{b:.1f}" for a, b in points)
    hit = _cache.get(key)
    if hit and time.time() - hit["at"] < WEATHER_CACHE_SECONDS:
        return hit["rows"]

    params = {
        "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
        "current": "cloud_cover,temperature_2m",
    }
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as c:
            res = await c.get(WEATHER_URL, params=params)
            res.raise_for_status()
            data = res.json()
    except Exception:
        return hit["rows"] if hit else []

    # One coordinate returns an object; several return a list.
    if isinstance(data, dict):
        data = [data]
    rows = []
    for entry in data:
        current = entry.get("current") or {}
        rows.append({
            "lat": entry.get("latitude"),
            "lon": entry.get("longitude"),
            "cloud": current.get("cloud_cover"),
            "temp": current.get("temperature_2m"),
        })
    _cache[key] = {"at": time.time(), "rows": rows}
    return rows


# ── The join ──────────────────────────────────────────────────────────────────

async def aurora_tonight(
    min_probability: int = 10,
    max_cloud: int = 40,
    limit: int = 40,
) -> Dict[str, Any]:
    """
    Where the aurora is actually visible right now.

    Three conditions, all necessary, none of them a weighting:
      1. NOAA puts aurora probability at or above the threshold
      2. the sun is below civil twilight there, so the sky is dark
      3. Open-Meteo says the cloud cover is low enough to see through

    Filtering by 1 and 2 first is not just an optimisation — it is what keeps
    the weather call to a single request instead of hundreds.
    """
    grid, forecast_time = await fetch_aurora()
    now = datetime.now(timezone.utc)

    candidates = []
    for lon, lat, prob in grid:
        if prob < min_probability:
            continue
        # NOAA publishes longitude 0..359; the rest of the world uses -180..180.
        lon180 = lon - 360 if lon > 180 else lon
        if not is_dark(lat, lon180, now):
            continue
        candidates.append({"lat": float(lat), "lon": float(lon180), "prob": int(prob)})

    candidates.sort(key=lambda c: -c["prob"])
    # Thin to roughly one point per 4 degrees so a single call covers the whole
    # visible band rather than a hundred neighbours of the same hotspot.
    spread: List[Dict[str, Any]] = []
    for c in candidates:
        if all(abs(c["lat"] - s["lat"]) > 4 or abs(c["lon"] - s["lon"]) > 6
               for s in spread):
            spread.append(c)
        if len(spread) >= MAX_WEATHER_POINTS:
            break

    conditions = await fetch_conditions([(c["lat"], c["lon"]) for c in spread])
    visible = []
    for c, wx in zip(spread, conditions):
        cloud = wx.get("cloud")
        if cloud is None or cloud > max_cloud:
            continue
        visible.append({**c, "cloud": int(cloud), "temp": wx.get("temp")})

    visible.sort(key=lambda v: (-v["prob"], v["cloud"]))
    return {
        "visible": visible[:limit],
        "dark_candidates": len(candidates),
        "grid_points": len(grid),
        "forecast_time": forecast_time,
        "min_probability": min_probability,
        "max_cloud": max_cloud,
    }


def summarise_aurora(result: Dict[str, Any]) -> str:
    """Built from the counts, never written by the model."""
    visible = result["visible"]
    if not result["grid_points"]:
        return "AURORA FEED UNAVAILABLE"
    if not result["dark_candidates"]:
        return f"NO AURORA ABOVE {result['min_probability']}% ON THE NIGHT SIDE"
    if not visible:
        return (
            f"{result['dark_candidates']} DARK SPOTS HAVE AURORA "
            f"BUT ALL ARE CLOUDED OVER"
        )
    best = visible[0]
    return (
        f"{len(visible)} CLEAR DARK SPOTS "
        f"BEST {best['prob']}% AURORA {best['cloud']}% CLOUD"
    )
