"""
backend/aircraft.py
Live aircraft from OpenSky, and place names resolved to boxes by Nominatim.

Two decisions worth stating.

One global fetch, filtered locally. OpenSky bills anonymous callers in credits
(~400/day) and charges per query regardless of area, so asking it separately
for Europe and then Japan costs twice. The whole sky is one request; every
region query after that is free.

A four-minute cache. That is not laziness about freshness: the map's dot pitch
is 2 degrees, roughly 220 km, and an airliner covers about 60 km in four
minutes. The cache is finer than anything the display can show, and it keeps
the day's usage inside the anonymous budget.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

STATES_URL = "https://opensky-network.org/api/states/all"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS = {"User-Agent": "DottedProjections/0.1 (https://dotte.duckdns.org)"}

CACHE_SECONDS = 240
GEOCODE_CACHE_SECONDS = 86400
REQUEST_TIMEOUT = 45.0

# OpenSky's state vector layout.
I_CALLSIGN, I_COUNTRY, I_LON, I_LAT, I_ALT, I_ONGROUND, I_VEL, I_TRACK = 1, 2, 5, 6, 7, 8, 9, 10

REGION_BOXES: Dict[str, Tuple[float, float, float, float]] = {
    # south, north, west, east — used before spending a geocode call.
    "europe": (34.0, 72.0, -25.0, 45.0),
    "asia": (-10.0, 60.0, 60.0, 150.0),
    "americas": (-56.0, 72.0, -170.0, -34.0),
    "africa": (-35.0, 38.0, -18.0, 52.0),
    "oceania": (-48.0, 0.0, 110.0, 180.0),
}

_cache: Dict[str, Any] = {}


async def fetch_aircraft() -> Dict[str, Any]:
    """Every airborne aircraft OpenSky can see. One call, cached."""
    hit = _cache.get("states")
    if hit and time.time() - hit["at"] < CACHE_SECONDS:
        return {**hit, "stale": False}

    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as c:
            res = await c.get(STATES_URL)
            res.raise_for_status()
            payload = res.json()
            remaining = res.headers.get("x-rate-limit-remaining")
    except Exception:
        # Stale positions are still real positions; nothing is better.
        return {**hit, "stale": True} if hit else {
            "planes": [], "at": 0, "remaining": None, "stale": True
        }

    planes = []
    for s in payload.get("states") or []:
        if s[I_ONGROUND] or s[I_LON] is None or s[I_LAT] is None:
            continue
        planes.append({
            "callsign": (s[I_CALLSIGN] or "").strip() or None,
            "country": s[I_COUNTRY],
            "lon": float(s[I_LON]),
            "lat": float(s[I_LAT]),
            "alt_m": s[I_ALT],
            "vel_ms": s[I_VEL],
            "track": s[I_TRACK],
        })

    entry = {"planes": planes, "at": time.time(),
             "remaining": remaining, "fetched": payload.get("time")}
    _cache["states"] = entry
    return {**entry, "stale": False}


async def geocode_box(place: str) -> Optional[Tuple[float, float, float, float]]:
    """Place name to (south, north, west, east). Known regions skip the call."""
    key = (place or "").strip().lower()
    if not key:
        return None
    if key in REGION_BOXES:
        return REGION_BOXES[key]

    cached = _cache.get("geo:" + key)
    if cached and time.time() - cached["at"] < GEOCODE_CACHE_SECONDS:
        return cached["box"]

    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=25.0) as c:
            res = await c.get(NOMINATIM_URL,
                              params={"q": place, "format": "json", "limit": 1})
            res.raise_for_status()
            rows = res.json()
    except Exception:
        return None
    if not rows:
        _cache["geo:" + key] = {"at": time.time(), "box": None}
        return None

    south, north, west, east = (float(v) for v in rows[0]["boundingbox"])
    box = (south, north, west, east)
    _cache["geo:" + key] = {"at": time.time(), "box": box}
    return box


def in_box(plane: Dict[str, Any], box: Tuple[float, float, float, float]) -> bool:
    south, north, west, east = box
    if not (south <= plane["lat"] <= north):
        return False
    if west <= east:
        return west <= plane["lon"] <= east
    # A box that crosses the antimeridian, e.g. Fiji or the Pacific.
    return plane["lon"] >= west or plane["lon"] <= east


def filter_aircraft(
    planes: List[Dict[str, Any]],
    box: Optional[Tuple[float, float, float, float]] = None,
    limit: int = 400,
) -> List[Dict[str, Any]]:
    """Planes inside the box, thinned evenly so the map is not a solid blob."""
    inside = [p for p in planes if box is None or in_box(p, box)]
    if len(inside) <= limit:
        return inside
    # Even stride rather than the first N: taking the head would show one
    # airline's block of consecutive state vectors, not the sky.
    step = len(inside) / limit
    return [inside[int(i * step)] for i in range(limit)]


def summarise(shown: int, total_in_area: int, total: int, place: str,
              stale: bool, remaining: Optional[str]) -> str:
    where = f" over {place}" if place else " worldwide"
    if not total:
        return "AIRCRAFT FEED UNAVAILABLE"
    if not total_in_area:
        return f"NO AIRCRAFT TRACKED{where}".upper()
    bits = [f"{total_in_area} AIRCRAFT AIRBORNE{where}".upper()]
    if shown < total_in_area:
        bits.append(f"SHOWING {shown}")
    if stale:
        bits.append("CACHED")
    return " ".join(bits)
