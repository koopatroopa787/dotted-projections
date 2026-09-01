# Dotted Projections

A world map rendered entirely as dots, driven by a small local LLM acting as a
director. The model classifies a command into one structured intent; the canvas
does all the animation.

    build/make_dots.py     grid generation from Natural Earth polygons (build-time, once)
    frontend/index.html    canvas renderer + command console
    frontend/dots.json     generated: 10,743 land dots on a 1.2 degree grid
    backend/server.py      FastAPI director -> Ollama (qwen2.5:0.5b-instruct)

## Regenerating the map

    curl -sL -o data/countries.geojson \
      https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
    python3 build/make_dots.py 1.2      # degrees; smaller = denser

Density reference: 2.0deg = 3,861 dots (39KB) - 1.5 = 6,876 (68KB)
1.2 = 10,743 (105KB) - 1.0 = 15,487 (153KB).

Antarctica is cropped (LAT_MIN = -56.0 in make_dots.py); set it to -90 to include it.

## Checks

    python3 backend/test_server.py
