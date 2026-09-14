# Dotted Projections

**A world map drawn entirely in dots, steered by a small local LLM.** Type a question in plain English and the map answers with live data — where the aircraft are, where the aurora is visible tonight, which markets are open.

[![Live](https://img.shields.io/badge/live-dotte.duckdns.org-2FA98C?style=for-the-badge)](https://dotte.duckdns.org/)

## What you can ask

| Ask | Shows |
|---|---|
| `flights over europe` | live aircraft positions |
| `where can I see the aurora tonight` | places that are dark, under the aurora oval **and** clear-sky |
| `recent earthquakes in japan` | recent quakes |
| `which markets are open right now` | exchanges currently in session |
| `where am I` | your approximate location |
| `hello`, `dog`, any word | the dots re-form into that text or shape |
| `map` | back to the world |

Data: [OpenSky](https://opensky-network.org/) for aircraft, [NOAA SWPC](https://www.swpc.noaa.gov/) aurora forecast, [Open-Meteo](https://open-meteo.com/) cloud cover, [USGS](https://earthquake.usgs.gov/) earthquakes, Yahoo Finance for markets, and [Nominatim](https://nominatim.openstreetmap.org/) to turn place names into regions.

## The model's actual job

Routing is rules first, and every number is computed, never generated. The model — `qwen2.5:3b-instruct` via Ollama, on CPU — does one thing: turns language into a structured query. "Flights above the land of the rising sun" → `japan`; "markets in the country with the maple leaf flag" → `canada`.

That role was measured, not assumed. With the model switched off, 8 of 12 test questions degrade. The smaller `qwen2.5:0.5b` changed only 1 of 28 answers and resolved places correctly 3 times in 8, against 6 in 8 for the 3b — which is why the 3b runs in production.

The aurora answer is the most interesting join: 65,160 grid points, narrowed to 564 that are both dark and under the oval, narrowed to 5 with clear skies.

## Layout

    build/make_dots.py      land grid generated from Natural Earth polygons (once, at build time)
    frontend/index.html     canvas renderer, morphing and command console
    frontend/dots.json      generated: 4,240 land dots on a 2° grid
    backend/server.py       FastAPI director: routing, intent extraction via Ollama
    backend/world.py        solar elevation, aurora oval, cloud cover
    backend/aircraft.py     OpenSky positions and place-to-bounding-box geocoding
    backend/markets.py      exchange sessions and hours
    backend/quakes.py       USGS earthquake feed
    backend/test_server.py  routing checks

## Run it

    ollama pull qwen2.5:3b-instruct
    pip install fastapi uvicorn httpx pydantic
    cd backend && uvicorn server:app --port 8020
    python3 test_server.py

Set `DOTTED_MODEL` to use a different Ollama model.

## Regenerating the map

    curl -sL -o data/countries.geojson \
      https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
    python3 build/make_dots.py 2.0      # degrees per dot; smaller is denser

Land is detected by point-in-polygon with even-odd ray casting across every ring, so lakes stay empty. A rescue pass gives any country too small for the grid its best free cell, so island nations don't disappear. Antarctica is cropped at `LAT_MIN = -56`.
