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

    if decision["intent"] == "chat":
        reply = await _chat_reply(text, cmd.history or [])
        return JSONResponse({
            "intent": "chat",
            "params": {"reply": reply},
            "via": "model" if reply else "fallback",
        })

    return JSONResponse({**decision, "via": "rule"})


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
