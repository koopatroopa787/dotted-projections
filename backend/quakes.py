"""
backend/quakes.py
Live earthquakes from USGS, and the filter the director's model drives.

Division of labour, on purpose:

  the model  turns fuzzy human phrasing into a structured query
             ("any big ones near japan overnight" -> min_magnitude 4.5,
              place japan, window day). Open-vocabulary place names and words
              like "big" or "serious" are exactly what a regex cannot do and a
              small model can.

  this file  does the filtering and writes the summary from the real numbers.
             The model never states a count or a magnitude, so it cannot
             hallucinate one.

USGS feeds are public, no key, updated every minute.
"""
import time
from typing import Any, Dict, List, Optional

import httpx

FEEDS = {
    "hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
    "week": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson",
}
DEFAULT_WINDOW = "day"
CACHE_SECONDS = 60          # USGS refreshes about once a minute
REQUEST_TIMEOUT = 20.0
MAX_RESULTS = 60

_cache: Dict[str, Any] = {}


async def fetch_quakes(window: str = DEFAULT_WINDOW) -> List[Dict[str, Any]]:
    """Every quake USGS lists for the window, newest first. Cached for a minute."""
    window = window if window in FEEDS else DEFAULT_WINDOW
    hit = _cache.get(window)
    if hit and time.time() - hit["at"] < CACHE_SECONDS:
        return hit["quakes"]

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            res = await client.get(FEEDS[window])
            res.raise_for_status()
            payload = res.json()
    except Exception:
        # Serve a stale cache rather than nothing; an old quake list is still
        # true, it is just not the latest.
        return hit["quakes"] if hit else []

    quakes = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2 or props.get("mag") is None:
            continue
        quakes.append({
            "id": feature.get("id"),
            "mag": round(float(props["mag"]), 1),
            "place": props.get("place") or "unknown",
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "depth_km": round(float(coords[2]), 1) if len(coords) > 2 else None,
            "time": props.get("time"),
            "url": props.get("url"),
        })

    quakes.sort(key=lambda q: q["time"] or 0, reverse=True)
    _cache[window] = {"at": time.time(), "quakes": quakes}
    return quakes


def filter_quakes(
    quakes: List[Dict[str, Any]],
    min_magnitude: float = 0.0,
    place: str = "",
    limit: int = 25,
    sort: str = "magnitude",
) -> List[Dict[str, Any]]:
    """Apply the model's parsed query. Pure, so it is testable without network."""
    needle = (place or "").strip().lower()
    out = [
        q for q in quakes
        if q["mag"] >= min_magnitude
        and (not needle or needle in q["place"].lower())
    ]
    if sort == "recent":
        out.sort(key=lambda q: q["time"] or 0, reverse=True)
    else:
        out.sort(key=lambda q: q["mag"], reverse=True)
    return out[: max(1, min(limit, MAX_RESULTS))]


def summarise(
    found: List[Dict[str, Any]], place: str, min_magnitude: float, window: str
) -> str:
    """
    The headline, built from the actual numbers.

    Deliberately not model-written: a count or a magnitude invented by a 0.5B
    model would be indistinguishable from a real one, and this is the half that
    has to be true.
    """
    where = f" near {place}" if place else ""
    span = {"hour": "in the last hour", "day": "today", "week": "this week"}.get(
        window, "today"
    )
    if not found:
        floor = f" above M{min_magnitude:g}" if min_magnitude else ""
        return f"no quakes{floor}{where} {span}".upper()

    strongest = max(found, key=lambda q: q["mag"])
    noun = "quake" if len(found) == 1 else "quakes"
    return (
        f"{len(found)} {noun}{where} {span} strongest M{strongest['mag']:g}"
    ).upper()
