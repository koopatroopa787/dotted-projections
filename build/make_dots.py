"""
build/make_dots.py
Generate the world dot grid from real geographic data, once, at build time.

Samples a regular lat/lon grid against Natural Earth country polygons with
point-in-polygon, so every dot corresponds to an actual coordinate. On an
equirectangular projection an equal angular grid is also an even grid in screen
space, which is what gives the reference image its look — so the geographic
correctness and the visual style want the same sampling.

Each land dot keeps the country it fell in, so "connect me to Japan" can resolve
to real dots rather than an approximate screen position.

Run: python build/make_dots.py [step_degrees]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "countries.geojson"
OUTPUT = ROOT / "frontend" / "dots.json"

# Antarctica is a solid band across the bottom of an equirectangular projection
# and dominates the composition. Standard for this map style is to crop it.
# Set to -90.0 to include it.
LAT_MIN = -56.0
LAT_MAX = 84.0
DEFAULT_STEP = 1.2


def rings_of(geometry):
    """Every linear ring in a Polygon or MultiPolygon, flattened."""
    kind = geometry.get("type")
    coords = geometry.get("coordinates") or []
    if kind == "Polygon":
        return [coords]
    if kind == "MultiPolygon":
        return list(coords)
    return []


def bbox_of(polygon):
    xs, ys = [], []
    for ring in polygon:
        for x, y in ring:
            xs.append(x)
            ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def point_in_polygon(x, y, polygon):
    """
    Even-odd ray casting across every ring of one polygon.

    Counting exterior and interior rings together is what makes holes work:
    a point inside a hole crosses the outer ring once and the hole ring once,
    so it comes out even, i.e. outside. Lakes stay empty.
    """
    inside = False
    for ring in polygon:
        n = len(ring)
        j = n - 1
        for i in range(n):
            xi, yi = ring[i][0], ring[i][1]
            xj, yj = ring[j][0], ring[j][1]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x_cross > x:
                    inside = not inside
            j = i
    return inside


def load_countries():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    entries = []
    for feature in data["features"]:
        props = feature["properties"]
        name = props.get("NAME") or props.get("NAME_LONG") or "Unknown"
        iso = props.get("ISO_A2") or ""
        for polygon in rings_of(feature["geometry"]):
            entries.append({
                "name": name,
                "iso": iso,
                "polygon": polygon,
                "bbox": bbox_of(polygon),
            })
    return entries


def build(step: float):
    countries = load_countries()
    names, name_index = [], {}

    cols, rows, country_ids = [], [], []
    n_cols = int(round(360.0 / step))
    n_rows = int(round((LAT_MAX - LAT_MIN) / step))

    for row in range(n_rows + 1):
        lat = LAT_MAX - row * step
        # Skip polygons that cannot contain this latitude at all.
        band = [c for c in countries if c["bbox"][1] <= lat <= c["bbox"][3]]
        if not band:
            continue
        for col in range(n_cols):
            lon = -180.0 + col * step
            hit = None
            for c in band:
                x0, _, x1, _ = c["bbox"]
                if lon < x0 or lon > x1:
                    continue
                if point_in_polygon(lon, lat, c["polygon"]):
                    hit = c
                    break
            if hit is None:
                continue

            key = hit["name"]
            if key not in name_index:
                name_index[key] = len(names)
                names.append({"name": key, "iso": hit["iso"]})
            cols.append(col)
            rows.append(row)
            country_ids.append(name_index[key])

    return {
        "step": step,
        "lonMin": -180.0,
        "latMax": LAT_MAX,
        "latMin": LAT_MIN,
        "nCols": n_cols,
        "nRows": n_rows + 1,
        "count": len(cols),
        "countries": names,
        # Parallel arrays: dot i is at grid cell (cols[i], rows[i]) in country
        # countryIds[i]. Grid cell -> lon/lat is a multiply in the renderer.
        "cols": cols,
        "rows": rows,
        "countryIds": country_ids,
    }


if __name__ == "__main__":
    step = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_STEP
    grid = build(step)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(grid, separators=(",", ":")), encoding="utf-8")
    size_kb = OUTPUT.stat().st_size / 1024
    print(
        f"step={step}deg  grid={grid['nCols']}x{grid['nRows']}  "
        f"land dots={grid['count']}  countries={len(grid['countries'])}  "
        f"{size_kb:.0f}KB -> {OUTPUT}"
    )
