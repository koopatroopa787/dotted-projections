"""
backend/server.py
The director: free text in, one schema-constrained intent out.

The model never renders anything and never drives motion. It emits a single
small decision; the canvas does all the animating. That is what makes a 0.5B
model on a 2-core ARM box the right size for this job — it is doing
classification, not reasoning.

Run: uvicorn server:app --host 127.0.0.1 --port 8020
"""
import json
import os
import re
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from quakes import fetch_quakes, filter_quakes, summarise
from markets import fetch_markets, filter_markets, summarise as summarise_markets
from world import aurora_tonight, summarise_aurora
from markets import EXCHANGES
from aircraft import (fetch_aircraft, geocode_box, filter_aircraft, in_box,
                      summarise as summarise_aircraft)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("DOTTED_MODEL", "qwen2.5:0.5b-instruct")
REQUEST_TIMEOUT = 25.0

app = FastAPI(title="Dotted Projections director")

# The model is small and will occasionally answer in prose despite the schema.
# These let the request still resolve instead of failing the user's command.
#
# SHAPE is tested BEFORE locate: "show me a dog" contains "show me", which the
# locate pattern would otherwise swallow.
SHAPE_HINTS = re.compile(
    # "an" must precede "a" in the alternation, or "an octopus" matches the "a"
    # and the word comes through as "n octopus".
    r"^\s*(?:draw|sketch|spell|shape|make|render)\s+(?:me\s+)?"
    r"(?:(?:an|a|the)\s+)?(.+?)\s*[.!?]?$"
    r"|^\s*show\s+me\s+(?:(?:an|a|the)\s+)(.+?)\s*[.!?]?$",
    re.IGNORECASE,
)
LOCATE_HINTS = re.compile(
    r"\b(my|me|i am|i'm)\b.*\b(location|position|where|city|here)\b"
    r"|\bwhere\s+am\s+i\b|\bshow\s+me\b|\bfind\s+me\b|\blocate\s+me\b",
    re.IGNORECASE,
)
LATLON = re.compile(
    r"(-?\d{1,2}(?:\.\d+)?)\s*[,/ ]\s*(-?\d{1,3}(?:\.\d+)?)"
)
REVERSE_HINTS = re.compile(r"\b(reverse|invert|inverse|negative)\b", re.IGNORECASE)
# Getting back to the map. Without this every bare word is a shape request,
# so "map" drew the word MAP and there was no way home.
MAP_HINTS = re.compile(
    r"^\s*(the\s+)?(map|world|world map|globe|home|back|reset|clear|earth)\s*[.!?]?$",
    re.IGNORECASE,
)

PLANE_HINTS = re.compile(
    r"\b(plane|planes|aircraft|flight|flights|flying|airborne|aviation|air traffic|jets?)\b",
    re.IGNORECASE,
)
AURORA_HINTS = re.compile(
    r"\b(aurora|northern lights|southern lights|borealis|australis)\b",
    re.IGNORECASE,
)
MARKET_HINTS = re.compile(
    r"\b(market|markets|exchange|exchanges|stock|stocks|bourse|index|indices|"
    r"trading|nasdaq|nyse|ftse|nikkei|sensex|nifty)\b"
    r"|\b(?:how (?:is|are|s)|hows)\b.*\b(?:americas?|europe|asia|africa|oceania|"
    r"doing|performing|looking)\b",
    re.IGNORECASE,
)
# Built from the exchange table rather than typed out, so adding a venue to
# markets.py teaches the router about it automatically. A city on its own is
# ambiguous ("tokyo" could be a shape to draw), so it only routes to markets
# alongside a trading word.
MARKET_PLACES = re.compile(
    r"\b(" + "|".join(
        sorted({re.escape(e[k].lower()) for e in EXCHANGES
                for k in ("city", "country", "code")}, key=len, reverse=True)
    ) + r")\b",
    re.IGNORECASE,
)
MARKET_CONTEXT = re.compile(
    r"\b(open|closed|trading|traded|market|markets|doing|performing|index|"
    r"stocks?|shares?|up|down|close|session)\b",
    re.IGNORECASE,
)

QUAKE_HINTS = re.compile(
    r"\b(quake|quakes|earthquake|earthquakes|seismic|tremor|tremors|richter)\b",
    re.IGNORECASE,
)
CHAT_HINTS = re.compile(
    r"\btalk\s+to\s+me\b|\bsay\s+something\b|\bchat\b|\?\s*$", re.IGNORECASE
)

# The dots spell the reply one word at a time, so a long answer is unreadable.
CHAT_MAX_CHARS = 78
# Enough for the reply to follow the thread; more than this and a 0.5B
# model starts losing the plot rather than gaining context.
CHAT_HISTORY_TURNS = 4
CHAT_PROMPT = (
    "You are the voice of a dot-matrix world map. Answer in ONE finished "
    "sentence of ten words or fewer. Never stop mid-sentence. Plain words, "
    "no punctuation, no emoji, no quotes. "
    "Good: I AM LISTENING TO YOU / THE WORLD IS QUIET TONIGHT"
)


# Tried in order. ipapi.co is the one the brief named, but it rate-limits this
# datacenter address, so it sits second behind a provider that answers.
IP_PROVIDERS = [
    {
        "name": "ipwho.is",
        "url": "https://ipwho.is/{ip}",
        "lat": "latitude", "lon": "longitude", "country": "country",
        "failed": lambda d: d.get("success") is False,
    },
    {
        "name": "ipapi.co",
        "url": "https://ipapi.co/{ip}/json/",
        "lat": "latitude", "lon": "longitude", "country": "country_name",
        "failed": lambda d: bool(d.get("error")),
    },
]


class Command(BaseModel):
    text: str
    # Prior turns, oldest first, as {"you": ..., "map": ...}. The server
    # keeps no state: whoever is talking carries their own thread.
    history: List[Dict[str, str]] = []


def _fallback(text: str) -> Dict[str, Any]:
    """
    Deterministic classification. Every input lands somewhere: a bare word is a
    shape request, because typing "cat" and watching nothing happen is the worst
    thing this interface can do.
    """
    match = LATLON.search(text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"intent": "locate", "params": {"lat": lat, "lon": lon}}

    if MAP_HINTS.match(text):
        return {"intent": "idle", "params": {}}

    if PLANE_HINTS.search(text):
        return {"intent": "aircraft", "params": {}}

    if AURORA_HINTS.search(text):
        return {"intent": "aurora", "params": {}}

    if MARKET_HINTS.search(text) or (
        MARKET_PLACES.search(text) and MARKET_CONTEXT.search(text)
    ):
        return {"intent": "markets", "params": {}}

    if QUAKE_HINTS.search(text):
        return {"intent": "quakes", "params": {}}

    if REVERSE_HINTS.search(text):
        return {"intent": "reverse", "params": {}}

    shape = SHAPE_HINTS.match(text)
    if shape:
        word = (shape.group(1) or shape.group(2) or "").strip()
        # "draw my location" is a locate dressed as a draw.
        if word and not LOCATE_HINTS.search(word):
            return {"intent": "shape", "params": {"word": word[:40]}}

    if LOCATE_HINTS.search(text):
        return {"intent": "locate", "params": {}}

    if CHAT_HINTS.search(text):
        return {"intent": "chat", "params": {}}

    # Anything short and unclaimed is a thing to draw. Longer prose is talk.
    words = text.split()
    if 1 <= len(words) <= 3 and len(text) <= 24:
        return {"intent": "shape", "params": {"word": text.strip()[:24]}}
    return {"intent": "chat", "params": {}}


@app.get("/health")
async def health():
    ok, detail = False, "unreachable"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{OLLAMA_URL}/api/tags")
            tags = res.json().get("models", [])
            names = [m.get("name") for m in tags]
            ok = MODEL in names
            detail = f"model {'loaded' if ok else 'missing'}; have {names}"
    except Exception as exc:
        detail = f"ollama error: {exc}"
    return JSONResponse({"status": "ok", "model": MODEL, "director_ready": ok,
                         "detail": detail})


@app.post("/intent")
async def classify(cmd: Command):
    text = (cmd.text or "").strip()[:400]
    if not text:
        return JSONResponse({"intent": "idle", "params": {}})

    # Rules decide the intent — the 0.5B model is not more reliable than a regex
    # at this, and asked to do it it both regressed ("where am I" -> idle) and
    # invented coordinates. The model's job is writing the chat reply, which is
    # the one thing here a regex cannot do.
    decision = _fallback(text)

    if decision["intent"] == "aircraft":
        q = await _parse_with(PLANE_SCHEMA, PLANE_PROMPT, text, {"place": ""})
        place = _accept_place(q.get("place"), text)
        box = await geocode_box(place) if place else None
        if place and box is None:
            place = ""            # unresolvable name: show the world, say so

        feed = await fetch_aircraft()
        planes = feed["planes"]
        in_area = [p for p in planes if box is None or in_box(p, box)]
        shown = filter_aircraft(planes, box)
        return JSONResponse({
            "intent": "aircraft",
            "params": {
                "aircraft": shown,
                "in_area": len(in_area),
                "total": len(planes),
                "place": place,
                "box": list(box) if box else None,
                "stale": feed.get("stale", False),
                "credits_left": feed.get("remaining"),
                "summary": summarise_aircraft(
                    len(shown), len(in_area), len(planes), place,
                    feed.get("stale", False), feed.get("remaining"),
                ),
            },
            "via": q.get("via", "rule"),
        })

    if decision["intent"] == "aurora":
        q = await _parse_with(
            AURORA_SCHEMA, AURORA_PROMPT, text,
            {"min_probability": 10, "max_cloud": 40},
        )
        prob = max(1, min(100, int(q.get("min_probability") or 10)))
        cloud = max(0, min(100, int(q.get("max_cloud") or 40)))
        # "clear skies" is a statement about cloud, not about aurora strength.
        if not AURORA_STRONG.search(text):
            prob = 5 if AURORA_ANY.search(text) else 10
        if CLEAR_SKY.search(text):
            cloud = min(cloud, 20)
        result = await aurora_tonight(min_probability=prob, max_cloud=cloud)
        return JSONResponse({
            "intent": "aurora",
            "params": {**result, "summary": summarise_aurora(result),
                       "query": {"min_probability": prob, "max_cloud": cloud}},
            "via": q.get("via", "rule"),
        })

    if decision["intent"] == "markets":
        q = await _parse_with(
            MARKET_SCHEMA, MARKET_PROMPT, text,
            {"region": "", "place": "", "only_open": False, "sort": "region"},
        )
        place = _accept_place(q.get("place"), text)
        # A continent is a region, not a city. Left in place it becomes a
        # substring search for "asia" inside city names, which matches nothing
        # and silently empties the result.
        if place in ("americas", "america", "europe", "asia", "africa", "oceania"):
            place = ""
        region = _pin_region(text)
        only_open = bool(q.get("only_open")) or bool(OPEN_NOW.search(text))
        sort = q.get("sort") if q.get("sort") in ("region", "gainers", "losers") else "region"
        if SORT_WORST.search(text):
            sort = "losers"
        elif SORT_BEST.search(text):
            sort = "gainers"

        markets = await fetch_markets()
        found = filter_markets(markets, region=region, place=place,
                               only_open=only_open, sort=sort)
        return JSONResponse({
            "intent": "markets",
            "params": {
                "markets": found,
                "total": len(markets),
                "query": {"region": region, "place": place,
                          "only_open": only_open, "sort": sort},
                "summary": summarise_markets(found, only_open, region, place),
            },
            "via": q.get("via", "rule"),
        })

    if decision["intent"] == "quakes":
        query = await _parse_quake_query(text)
        quakes = await fetch_quakes(query["window"])
        found = filter_quakes(
            quakes,
            min_magnitude=query["min_magnitude"],
            place=query["place"],
            sort=query["sort"],
        )
        return JSONResponse({
            "intent": "quakes",
            "params": {
                "query": query,
                "quakes": found,
                "total_in_feed": len(quakes),
                "summary": summarise(
                    found, query["place"], query["min_magnitude"], query["window"]
                ),
            },
            "via": query["via"],
        })

    if decision["intent"] == "chat":
        reply = await _chat_reply(text, cmd.history or [])
        return JSONResponse({
            "intent": "chat",
            "params": {"reply": reply},
            "via": "model" if reply else "fallback",
        })

    return JSONResponse({**decision, "via": "rule"})



# What the model is genuinely needed for. "big ones near the ring of fire",
# "anything serious overnight", "strongest in chile" — open-vocabulary places
# and fuzzy magnitude words. A regex can match the word "earthquake"; it cannot
# decide that "serious" means about M5.5.
QUAKE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_magnitude": {"type": "number"},
        "place": {"type": "string"},
        "window": {"type": "string", "enum": ["hour", "day", "week"]},
        "sort": {"type": "string", "enum": ["magnitude", "recent"]},
    },
    "required": ["min_magnitude", "place", "window", "sort"],
}

QUAKE_PROMPT = (
    "Turn an earthquake question into a query. Reply with JSON only.\n"
    "min_magnitude: any/all/every = 0, small/minor = 0, notable/big/large = 4.5, "
    "major/serious/strong/severe = 5.5, huge/massive/devastating = 6.5. "
    "If the user names a number, use that number.\n"
    "place: the country, region or ocean named, lowercase, empty string if none. "
    "Never invent a place.\n"
    "window: hour for 'right now' or 'last hour', week for 'this week' or "
    "'recent days', otherwise day.\n"
    "sort: recent if they ask for latest or newest, otherwise magnitude."
)


# Words with one obvious reading are pinned here rather than left to the model.
# It read "minor tremors" as M4.5 and returned nothing, which is the worst
# possible answer to that question. The model keeps the open-vocabulary work —
# place names and phrasings nobody enumerated — and these guard the rest.
MAG_WORDS = [
    (re.compile(r"\b(minor|small|little|tiny|weak|slight)\b", re.I), 0.0),
    (re.compile(r"\b(any|all|every|各)\b", re.I), 0.0),
    (re.compile(r"\b(strongest|biggest|largest)\b", re.I), 0.0),  # sort handles it
    (re.compile(r"\b(huge|massive|devastating|catastrophic)\b", re.I), 6.5),
    (re.compile(r"\b(major|serious|severe)\b", re.I), 5.5),
    (re.compile(r"\b(big|large|notable|significant|strong)\b", re.I), 4.5),
]
EXPLICIT_MAG = re.compile(
    r"(?:above|over|greater than|bigger than|at least|>=?|m)\s*(\d(?:\.\d)?)", re.I
)


def _magnitude_from_words(text: str):
    """An explicit number wins; then the first unambiguous adjective."""
    m = EXPLICIT_MAG.search(text)
    if m:
        try:
            return max(0.0, min(10.0, float(m.group(1))))
        except ValueError:
            pass
    for pattern, value in MAG_WORDS:
        if pattern.search(text):
            return value
    return None


def _clamp_query(parsed: Dict[str, Any], text: str) -> Dict[str, Any]:
    """Keep the model inside sane bounds; fall back where it wandered."""
    try:
        mag = float(parsed.get("min_magnitude", 0) or 0)
    except (TypeError, ValueError):
        mag = 0.0
    pinned = _magnitude_from_words(text)
    if pinned is not None:
        mag = pinned
    place = _accept_place(parsed.get("place"), text)
    window = parsed.get("window")
    sort = parsed.get("sort")
    lowered = text.lower()
    if re.search(r"\b(strongest|biggest|largest)\b", lowered):
        sort = "magnitude"
    if re.search(r"\b(right now|last hour|this hour)\b", lowered):
        window = "hour"
    elif re.search(r"\b(week|past few days|recent days)\b", lowered):
        window = "week"
    elif re.search(r"\b(today|24 hours|overnight|day)\b", lowered):
        window = "day"
    return {
        "min_magnitude": max(0.0, min(10.0, mag)),
        "place": place,
        "window": window if window in ("hour", "day", "week") else "day",
        "sort": sort if sort in ("magnitude", "recent") else "magnitude",
    }


async def _parse_quake_query(text: str) -> Dict[str, Any]:
    payload = {
        "model": MODEL,
        "format": QUAKE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
        "messages": [
            {"role": "system", "content": QUAKE_PROMPT},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            res = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            res.raise_for_status()
            parsed = json.loads(res.json().get("message", {}).get("content", "{}"))
        return {**_clamp_query(parsed, text), "via": "model"}
    except Exception:
        return {"min_magnitude": 0.0, "place": "", "window": "day",
                "sort": "magnitude", "via": "fallback"}



PLANE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"place": {"type": "string"}},
    "required": ["place"],
}
PLANE_PROMPT = (
    "Extract the place a flight question is about. JSON only. "
    "place: the country, region, state or city named, lowercase. "
    "Empty string if the user named nowhere. Never invent a place."
)

AURORA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "min_probability": {"type": "integer"},
        "max_cloud": {"type": "integer"},
    },
    "required": ["min_probability", "max_cloud"],
}
AURORA_PROMPT = (
    "Turn an aurora question into thresholds. JSON only. "
    "min_probability is the NOAA aurora chance in percent: default 10, "
    "use 30 for 'strong' or 'good', 5 for 'any' or 'even faint'. "
    "max_cloud is the most cloud cover allowed in percent: default 40, "
    "use 20 for 'clear skies' or 'cloudless', 80 for 'any weather'."
)

MARKET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "region": {"type": "string",
                   "enum": ["", "americas", "europe", "asia", "africa", "oceania"]},
        "place": {"type": "string"},
        "only_open": {"type": "boolean"},
        "sort": {"type": "string", "enum": ["region", "gainers", "losers"]},
    },
    "required": ["region", "place", "only_open", "sort"],
}
MARKET_PROMPT = (
    "Turn a stock market question into a query. JSON only. "
    "region: one of americas europe asia africa oceania, or empty. "
    "place: a city or country the user named, lowercase, else empty. "
    "Never invent a place. "
    "only_open: true if they ask what is open or trading right now. "
    "sort: gainers if they ask what is up or best, losers if down or worst, "
    "else region."
)


# The model reaches for a plausible value when the user gave none: it answered
# "which markets are open right now" with region=europe, which filtered every
# result away. Anything with one obvious reading is pinned here; the model keeps
# the open-vocabulary work.
REGION_WORDS = {
    "americas": r"\b(americas?|north america|south america|latin america|us|usa|united states)\b",
    "europe": r"\beurope|european\b",
    "asia": r"\basia|asian\b",
    "africa": r"\bafrica|african\b",
    "oceania": r"\b(oceania|australia|pacific)\b",
}
SORT_BEST = re.compile(r"\b(best|top|gainers?|winners?|up|rising|strongest|rally)\b", re.I)
SORT_WORST = re.compile(r"\b(worst|losers?|down|falling|weakest|selling off)\b", re.I)
OPEN_NOW = re.compile(r"\b(open|trading|live)\b", re.I)
AURORA_STRONG = re.compile(r"\b(strong|good|great|intense|best|bright)\b", re.I)
AURORA_ANY = re.compile(r"\b(any|faint|weak|even)\b", re.I)
CLEAR_SKY = re.compile(r"\b(clear|cloudless|no cloud|clear skies)\b", re.I)


def _pin_region(text: str) -> str:
    """A region the user never named is a hallucination, not a default."""
    lowered = text.lower()
    for region, pattern in REGION_WORDS.items():
        if re.search(pattern, lowered, re.I):
            return region
    return ""


# Locative cue: the difference between a question that is asking about a place
# and one that is not.
LOCATIVE = re.compile(
    r"\b(over|in|near|above|around|at|across|within|from)\b", re.IGNORECASE
)


def _accept_place(model_place: str, text: str) -> str:
    """
    Decide whether to trust the model's place.

    Requiring the place to appear verbatim in the text was too strict: it threw
    away the one thing the model can do that a regex cannot. It resolves "the
    land of the rising sun" to japan and "the maple leaf flag" to canada, and
    the verbatim test discarded both. Measured over 28 phrasings, that guard
    was why disabling the model changed 1 answer.

    So: a place echoed in the text is always fine, and an inferred place is
    accepted only when the sentence actually points somewhere. "How many planes
    are in the air" names nowhere and has no locative cue, so an invented
    country is still rejected.
    """
    place = (model_place or "").strip().lower()[:40]
    if not place:
        return ""
    if place in text.lower():
        return place
    return place if LOCATIVE.search(text) else ""


async def _parse_with(schema: Dict[str, Any], prompt: str, text: str,
                      fallback: Dict[str, Any]) -> Dict[str, Any]:
    """One schema-constrained extraction. Falls back rather than failing."""
    payload = {
        "model": MODEL,
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 80},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            res = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            res.raise_for_status()
            parsed = json.loads(res.json().get("message", {}).get("content", "{}"))
        return {**fallback, **{k: v for k, v in parsed.items() if k in fallback},
                "via": "model"}
    except Exception:
        return {**fallback, "via": "fallback"}


async def _chat_reply(text: str, history: List[Dict[str, str]]) -> str:
    """A few words back, or empty if the model is unavailable."""
    messages = [{"role": "system", "content": CHAT_PROMPT}]
    for turn in history[-CHAT_HISTORY_TURNS:]:
        if turn.get("you"):
            messages.append({"role": "user", "content": str(turn["you"])[:200]})
        if turn.get("map"):
            messages.append({"role": "assistant", "content": str(turn["map"])[:60]})
    messages.append({"role": "user", "content": text})

    payload = {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 24},
        "messages": messages,
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            res = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            res.raise_for_status()
            raw = res.json().get("message", {}).get("content", "")
    except Exception:
        return ""

    # Keep only what the dot renderer can draw, and cap the length. The model
    # overshoots "five words or fewer" often enough to matter, and a reply cut
    # mid-sentence reads worse than a short one, so trim on a word boundary.
    # Whole words only. Slicing the joined string by character count cut
    # "WHENEVER" to "WHENEVE", and a dot-drawn half-word just looks broken.
    words = re.findall(r"[A-Za-z0-9']+", raw)[:12]
    out, used = [], 0
    for w in words:
        if used + len(w) + (1 if out else 0) > CHAT_MAX_CHARS:
            break
        used += len(w) + (1 if out else 0)
        out.append(w)
    return " ".join(out).upper()


def _client_ip(request: Request) -> str:
    """
    The visitor's address, not ours. nginx sets X-Forwarded-For; the left-most
    entry is the original client.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


@app.get("/whereami")
async def whereami(request: Request):
    """
    Coarse IP geolocation, used only when the visitor declines the browser
    prompt. Proxied through the backend so the page never talks to a third
    party directly.

    Must look up the VISITOR's IP explicitly: querying ipapi.co without one
    geolocates whichever machine made the call, which is the server, so every
    visitor would be told they are wherever this box is hosted.
    """
    ip = _client_ip(request)
    if not ip or ip.startswith(("127.", "10.", "192.168.", "172.")):
        return JSONResponse(
            {"error": "no public client address to locate"}, status_code=422
        )

    async with httpx.AsyncClient(timeout=8.0) as client:
        for provider in IP_PROVIDERS:
            try:
                res = await client.get(provider["url"].format(ip=ip))
                data = res.json()
            except Exception:
                continue
            if provider["failed"](data):
                continue
            lat, lon = data.get(provider["lat"]), data.get(provider["lon"])
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                continue
            return JSONResponse({
                "lat": lat, "lon": lon,
                "city": data.get("city"),
                "country": data.get(provider["country"]),
                "source": provider["name"],
            })

    return JSONResponse({"error": "all geolocation providers failed"},
                        status_code=502)
