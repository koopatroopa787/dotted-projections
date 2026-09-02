"""
backend/markets.py
The world's stock exchanges, where they are, whether they are open, and what
they did today.

Division of labour, deliberately:

  the model  turns phrasing into a query — "how is asia doing", "what's open
             right now", "worst market today", "is tokyo trading". Region and
             exchange names are open vocabulary; a regex cannot enumerate them.

  this file  decides what is open and what moved, from timezones and real
             quotes. The model never states a price or a percentage, so it
             cannot invent one.

Quotes come from Yahoo's public chart endpoint: free, no key. Sessions are
local exchange hours; holidays are not modelled, so a market can read "open"
on a public holiday.
"""
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx

# lat/lon are the exchange buildings. open/close are local trading hours,
# main session only (no pre/post, no lunch break for the Asian venues).
EXCHANGES: List[Dict[str, Any]] = [
    {"code": "NYSE",  "city": "New York",    "country": "United States", "region": "americas",
     "lat": 40.71,  "lon": -74.01,  "tz": "America/New_York",   "open": "09:30", "close": "16:00", "symbol": "^GSPC"},
    {"code": "TSX",   "city": "Toronto",     "country": "Canada",        "region": "americas",
     "lat": 43.65,  "lon": -79.38,  "tz": "America/Toronto",    "open": "09:30", "close": "16:00", "symbol": "^GSPTSE"},
    {"code": "BMV",   "city": "Mexico City", "country": "Mexico",        "region": "americas",
     "lat": 19.43,  "lon": -99.13,  "tz": "America/Mexico_City","open": "08:30", "close": "15:00", "symbol": "^MXX"},
    {"code": "B3",    "city": "Sao Paulo",   "country": "Brazil",        "region": "americas",
     "lat": -23.55, "lon": -46.63,  "tz": "America/Sao_Paulo",  "open": "10:00", "close": "17:00", "symbol": "^BVSP"},
    {"code": "LSE",   "city": "London",      "country": "United Kingdom","region": "europe",
     "lat": 51.51,  "lon": -0.13,   "tz": "Europe/London",      "open": "08:00", "close": "16:30", "symbol": "^FTSE"},
    {"code": "EPA",   "city": "Paris",       "country": "France",        "region": "europe",
     "lat": 48.86,  "lon": 2.35,    "tz": "Europe/Paris",       "open": "09:00", "close": "17:30", "symbol": "^FCHI"},
    {"code": "FWB",   "city": "Frankfurt",   "country": "Germany",       "region": "europe",
     "lat": 50.11,  "lon": 8.68,    "tz": "Europe/Berlin",      "open": "09:00", "close": "17:30", "symbol": "^GDAXI"},
    {"code": "SIX",   "city": "Zurich",      "country": "Switzerland",   "region": "europe",
     "lat": 47.37,  "lon": 8.54,    "tz": "Europe/Zurich",      "open": "09:00", "close": "17:30", "symbol": "^SSMI"},
    {"code": "BME",   "city": "Madrid",      "country": "Spain",         "region": "europe",
     "lat": 40.42,  "lon": -3.70,   "tz": "Europe/Madrid",      "open": "09:00", "close": "17:30", "symbol": "^IBEX"},
    {"code": "MIL",   "city": "Milan",       "country": "Italy",         "region": "europe",
     "lat": 45.46,  "lon": 9.19,    "tz": "Europe/Rome",        "open": "09:00", "close": "17:30", "symbol": "FTSEMIB.MI"},
    {"code": "OMX",   "city": "Stockholm",   "country": "Sweden",        "region": "europe",
     "lat": 59.33,  "lon": 18.07,   "tz": "Europe/Stockholm",   "open": "09:00", "close": "17:30", "symbol": "^OMX"},
    {"code": "JSE",   "city": "Johannesburg","country": "South Africa",  "region": "africa",
     "lat": -26.20, "lon": 28.05,   "tz": "Africa/Johannesburg","open": "09:00", "close": "17:00", "symbol": "^J203.JO"},
    {"code": "NSE",   "city": "Mumbai",      "country": "India",         "region": "asia",
     "lat": 19.08,  "lon": 72.88,   "tz": "Asia/Kolkata",       "open": "09:15", "close": "15:30", "symbol": "^NSEI"},
    {"code": "SGX",   "city": "Singapore",   "country": "Singapore",     "region": "asia",
     "lat": 1.29,   "lon": 103.85,  "tz": "Asia/Singapore",     "open": "09:00", "close": "17:00", "symbol": "^STI"},
    {"code": "HKEX",  "city": "Hong Kong",   "country": "Hong Kong",     "region": "asia",
     "lat": 22.32,  "lon": 114.17,  "tz": "Asia/Hong_Kong",     "open": "09:30", "close": "16:00", "symbol": "^HSI"},
    {"code": "SSE",   "city": "Shanghai",    "country": "China",         "region": "asia",
     "lat": 31.23,  "lon": 121.47,  "tz": "Asia/Shanghai",      "open": "09:30", "close": "15:00", "symbol": "000001.SS"},
    {"code": "TSE",   "city": "Tokyo",       "country": "Japan",         "region": "asia",
     "lat": 35.68,  "lon": 139.69,  "tz": "Asia/Tokyo",         "open": "09:00", "close": "15:00", "symbol": "^N225"},
    {"code": "KRX",   "city": "Seoul",       "country": "South Korea",   "region": "asia",
     "lat": 37.57,  "lon": 126.98,  "tz": "Asia/Seoul",         "open": "09:00", "close": "15:30", "symbol": "^KS11"},
    {"code": "TWSE",  "city": "Taipei",      "country": "Taiwan",        "region": "asia",
     "lat": 25.03,  "lon": 121.57,  "tz": "Asia/Taipei",        "open": "09:00", "close": "13:30", "symbol": "^TWII"},
    {"code": "ASX",   "city": "Sydney",      "country": "Australia",     "region": "oceania",
     "lat": -33.87, "lon": 151.21,  "tz": "Australia/Sydney",   "open": "10:00", "close": "16:00", "symbol": "^AXJO"},
    {"code": "NZX",   "city": "Wellington",  "country": "New Zealand",   "region": "oceania",
     "lat": -41.29, "lon": 174.78,  "tz": "Pacific/Auckland",   "open": "10:00", "close": "16:45", "symbol": "^NZ50"},
]

QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DottedProjections/0.1)"}
CACHE_SECONDS = 120
REQUEST_TIMEOUT = 15.0

_cache: Dict[str, Any] = {}


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_open(exchange: Dict[str, Any], now_utc: Optional[datetime] = None) -> bool:
    """
    Open on a weekday, inside local session hours.

    Public holidays are not modelled — there is no free, reliable, global
    exchange-holiday feed, and inventing one would be worse than being wrong
    a dozen days a year in a way the caller can see.
    """
    now = (now_utc or datetime.now(ZoneInfo("UTC"))).astimezone(
        ZoneInfo(exchange["tz"])
    )
    if now.weekday() >= 5:
        return False
    minute = now.hour * 60 + now.minute
    return _minutes(exchange["open"]) <= minute < _minutes(exchange["close"])


async def _fetch_quote(client: httpx.AsyncClient, symbol: str) -> Optional[Dict[str, Any]]:
    try:
        res = await client.get(
            QUOTE_URL.format(symbol=symbol), params={"interval": "1d", "range": "5d"}
        )
        res.raise_for_status()
        meta = res.json()["chart"]["result"][0]["meta"]
        last = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose") or meta.get("previousClose")
        if last is None or not prev:
            return None
        return {
            "last": round(float(last), 2),
            "change_pct": round(100.0 * (float(last) - float(prev)) / float(prev), 2),
            "currency": meta.get("currency"),
        }
    except Exception:
        return None


async def fetch_markets() -> List[Dict[str, Any]]:
    """Every exchange with its open state and today's move. Cached briefly."""
    hit = _cache.get("markets")
    if hit and time.time() - hit["at"] < CACHE_SECONDS:
        return hit["markets"]

    now = datetime.now(ZoneInfo("UTC"))
    out: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=REQUEST_TIMEOUT) as client:
            for ex in EXCHANGES:
                quote = await _fetch_quote(client, ex["symbol"])
                local = now.astimezone(ZoneInfo(ex["tz"]))
                out.append({
                    **{k: ex[k] for k in
                       ("code", "city", "country", "region", "lat", "lon", "symbol")},
                    "open_now": is_open(ex, now),
                    "local_time": local.strftime("%H:%M"),
                    "session": f"{ex['open']}-{ex['close']}",
                    "last": quote["last"] if quote else None,
                    "change_pct": quote["change_pct"] if quote else None,
                })
    except Exception:
        return hit["markets"] if hit else []

    _cache["markets"] = {"at": time.time(), "markets": out}
    return out


def filter_markets(
    markets: List[Dict[str, Any]],
    region: str = "",
    place: str = "",
    only_open: bool = False,
    sort: str = "region",
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Apply the parsed query. Pure, so it is testable without network."""
    region = (region or "").strip().lower()
    needle = (place or "").strip().lower()

    out = []
    for m in markets:
        if only_open and not m["open_now"]:
            continue
        if region and m["region"] != region:
            continue
        if needle and needle not in (
            m["city"] + " " + m["country"] + " " + m["code"]
        ).lower():
            continue
        out.append(m)

    if sort == "gainers":
        out.sort(key=lambda m: (m["change_pct"] is None, -(m["change_pct"] or 0)))
    elif sort == "losers":
        out.sort(key=lambda m: (m["change_pct"] is None, m["change_pct"] or 0))
    return out[: max(1, limit)]


def summarise(found: List[Dict[str, Any]], only_open: bool, region: str, place: str) -> str:
    """Headline built from the real numbers, never written by the model."""
    where = f" in {region}" if region else (f" in {place}" if place else "")
    if not found:
        return (f"no markets{where}" + (" open now" if only_open else "")).upper()

    if only_open:
        names = ", ".join(m["city"] for m in found[:4])
        more = f" +{len(found) - 4}" if len(found) > 4 else ""
        return f"{len(found)} open now {names}{more}".upper()

    rated = [m for m in found if m["change_pct"] is not None]
    if not rated:
        return f"{len(found)} markets{where}".upper()

    best = max(rated, key=lambda m: m["change_pct"])
    worst = min(rated, key=lambda m: m["change_pct"])
    opens = sum(1 for m in found if m["open_now"])
    return (
        f"{len(found)} markets{where} {opens} open "
        f"best {best['city']} {best['change_pct']:+.1f}% "
        f"worst {worst['city']} {worst['change_pct']:+.1f}%"
    ).upper()
