# Dotted Projections

A world map rendered entirely as dots, driven by a small local LLM acting as a
director. The model classifies a command into one structured intent; the canvas
does all the animation.

    build/make_dots.py     grid generation from Natural Earth polygons (build-time, once)
    frontend/index.html    canvas renderer + command console
    frontend/dots.json     generated: 544 land dots on a 6 degree grid
    backend/server.py      FastAPI director -> Ollama (qwen2.5:0.5b-instruct)

## Regenerating the map

    curl -sL -o data/countries.geojson \
      https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson
    python3 build/make_dots.py 6.0      # degrees; smaller = denser

Density reference (cell-coverage sampling): 6.5deg = 478 dots - 6.0 = 544 (8KB)
5.0 = 776 - 4.0 = 969. 6.0 matches the reference image.

Antarctica is cropped (LAT_MIN = -56.0 in make_dots.py); set it to -90 to include it.

## Checks

    python3 backend/test_server.py
