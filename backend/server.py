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
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("DOTTED_MODEL", "qwen2.5:0.5b-instruct")
REQUEST_TIMEOUT = 25.0

app = FastAPI(title="Dotted Projections director")

# Intents are a closed set. Adding one here and in the frontend switch is the
# whole extension path — the model is never asked to invent a shape.
INTENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["locate", "shape", "idle"]},
        "params": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "label": {"type": "string"},
                "word": {"type": "string"},
            },
        },
    },
    "required": ["intent"],
}

SYSTEM_PROMPT = (
    "You classify a command for a world map into exactly one intent.\n\n"
    '"locate" = the user wants THEIR OWN position marked.\n'
    '  yes: "show my location" / "where am I" / "find me" / "mark my city"\n'
    '"shape" = the user wants the dots rearranged into a thing or a word.\n'
    '  yes: "draw a dog" / "spell hello" / "make a heart"\n'
    '  put the thing in params.word\n'
    '"idle" = everything else.\n'
    '  yes: "hello" / "what is the capital of France" / "connect me to Tokyo"\n'
    '       / "tell me about Peru" / random text\n\n'
    "A question about a place is NOT locate — locate is only about the user.\n"
    "Reply with JSON only. Do not add params unless the user gave an explicit "
    "latitude and longitude."
)

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


def _fallback(text: str) -> Dict[str, Any]:
    """Deterministic classification, used when the model is unavailable or vague."""
    match = LATLON.search(text)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"intent": "locate", "params": {"lat": lat, "lon": lon}}

    shape = SHAPE_HINTS.match(text)
    if shape:
        word = (shape.group(1) or shape.group(2) or "").strip()
        # "draw my location" is a locate dressed as a draw.
        if word and not LOCATE_HINTS.search(word):
            return {"intent": "shape", "params": {"word": word[:40]}}

    if LOCATE_HINTS.search(text):
        return {"intent": "locate", "params": {}}
    return {"intent": "idle", "params": {}}


def _coerce(raw: str, allow_coords: bool = True) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    intent = data.get("intent")
    if intent not in ("locate", "shape", "idle"):
        return None
    params = data.get("params")
    params = params if isinstance(params, dict) else {}

    # The model happily emits lat/lon it was never given — 37.7749,-122.4194
    # (San Francisco) for any prompt mentioning a location. Marking a
    # hallucinated city for every visitor is worse than marking nothing, so
    # coordinates survive only when the user actually typed some.
    if not allow_coords:
        params = {k: v for k, v in params.items() if k not in ("lat", "lon")}
    return {"intent": intent, "params": params}


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

    # Rules first, model second. A 0.5B classifier is genuinely useful for the
    # long tail of phrasings, but it is not more reliable than a regex on the
    # phrasings the regex already covers — and asked to handle those it both
    # regressed ("where am I" -> idle) and invented coordinates. So anything
    # unambiguous is decided here and never reaches the model.
    if LATLON.search(text) or SHAPE_HINTS.match(text) or LOCATE_HINTS.search(text):
        return JSONResponse({**_fallback(text), "via": "rule"})

    payload = {
        "model": MODEL,
        "format": INTENT_SCHEMA,   # Ollama constrains sampling to this shape.
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            res = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            res.raise_for_status()
            content = res.json().get("message", {}).get("content", "")
        intent = _coerce(content, allow_coords=False)
        if intent is not None:
            return JSONResponse({**intent, "via": "model"})
    except Exception:
        pass

    return JSONResponse({**_fallback(text), "via": "fallback"})


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
