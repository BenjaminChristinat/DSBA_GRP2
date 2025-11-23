#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-

import json
import unicodedata
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import osmnx as ox
from shapely.geometry import box, Polygon, MultiPolygon
from shapely.ops import unary_union

# ---------------------- CONFIG ----------------------
COUNTRY = "Switzerland"        # Hard limit: we only look inside Switzerland
PADDING_DEG = 0.002            # default padding on each side (≈ 200 m)
MAX_PADDING_DEG = 0.01         # max padding after auto-inflation (≈ 1.1 km)
PADDING_STEP = 0.0005          # inflation step for verification iterations

EFF_OK = 0.60                  # efficiency threshold: area(city)/area(rectangles)
EFF_IMPROVE_MIN = 0.07         # require >= 7% improvement to accept more slices
MAX_SLICES_MAIN = 5            # cap slices for main component (before count check)
WATER_WASTE_SHARE_TO_ALLOW_ONE_BOX = 0.60  # if ≥60% waste is water, keep 1 box

MAX_RECTANGLES_PER_CITY = 10   # hard target (exclaves may force more; we warn)
DECIMALS = 6                   # JSON rounding
# Cities to process:
CITIES = [
    "Zurich", "Geneva", "Basel", "Lausanne", "Bern", "Zug", "Lugano", "Lucerne",
    "Interlaken", "St. Moritz", "Davos", "Montreux", "Vevey", "Nyon", "Rolle",
    "Morges", "Pully", "Sion", "Martigny", "Opfikon", "Carouge", "Lancy", "La Tour-de-Peilz",
]
# ---------------------------------------------------


# ---------- Utilities ----------
def norm(s: str) -> str:
    """Accent-insensitive, casefolded string."""
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII").casefold()

def any_name_matches(row, target: str) -> bool:
    """True if any name-like field matches the target (accent-insensitive)."""
    target_n = norm(target)
    for k, v in row.items():
        if not isinstance(v, str):
            continue
        if k.startswith("name") or "name" in k or k in ("official_name", "alt_name"):
            if norm(v) == target_n:
                return True
    return False

def ensure_multiplolygon(geom):
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    raise ValueError("Geometry is not polygonal.")

def rect_from_bounds(b: Tuple[float, float, float, float]):
    minx, miny, maxx, maxy = b
    return box(minx, miny, maxx, maxy)

def pad_bounds(bounds, pad: float):
    minx, miny, maxx, maxy = bounds
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)

def bounds_to_dict(bounds, prefix: str, idx: int):
    minx, miny, maxx, maxy = bounds
    return (
        f"{prefix}_Zone{idx}",
        dict(
            south=round(miny, DECIMALS),
            north=round(maxy, DECIMALS),
            west=round(minx, DECIMALS),
            east=round(maxx, DECIMALS),
        ),
    )

def union_area(geoms: List):
    if not geoms:
        return 0.0
    return unary_union(geoms).area

def slice_component(component: Polygon, slices: int) -> List[Polygon]:
    """Slice a polygon along its long axis into 'slices' parts."""
    minx, miny, maxx, maxy = component.bounds
    width = maxx - minx
    height = maxy - miny
    polys = []
    if width >= height:
        # vertical slices (east-west city)
        step = width / slices
        for i in range(slices):
            x0 = minx + i * step
            x1 = minx + (i + 1) * step
            polys.append(component.intersection(box(x0, miny, x1, maxy)))
    else:
        # horizontal slices (north-south city)
        step = height / slices
        for i in range(slices):
            y0 = miny + i * step
            y1 = miny + (i + 1) * step
            polys.append(component.intersection(box(minx, y0, maxx, y1)))
    return [p for p in polys if not p.is_empty]

def fetch_commune_geom(city: str) -> MultiPolygon:
    """
    Get the Swiss commune (admin_level=8) geometry for 'city'.
    We strictly filter to Switzerland & admin_level=8 and prefer name-matched features.
    """
    place = f"{city}, {COUNTRY}"
    tags = {"boundary": "administrative", "admin_level": "8"}
    try:
        gdf = ox.features_from_place(place, tags)
    except Exception:
        # fallback: try without country in the place string
        gdf = ox.features_from_place(city, tags)

    if gdf.empty:
        raise RuntimeError(f"No admin_level=8 boundary found for {city}")

    # Keep polygons only
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError(f"Boundary result not polygonal for {city}")

    # Prefer exact name matches; else, pick the one closest to the geocoded city centroid.
    gdf["name_match"] = gdf.apply(lambda r: any_name_matches(r, city), axis=1)
    candidates = gdf[gdf["name_match"]].copy()
    if candidates.empty:
        # No exact name match across name tags; use proximity to geocoded centroid
        place_gdf = ox.geocode_to_gdf(place)
        centroid = place_gdf.geometry.unary_union.centroid
        gdf["dist"] = gdf.geometry.distance(centroid)
        chosen = gdf.sort_values("dist").head(1)
    else:
        chosen = candidates

    geom = unary_union(chosen.geometry.values)
    mp = ensure_multiplolygon(geom)

    # Final guard: ensure we truly have Swiss admin_level=8 (city, not canton)
    # (If OSM data carries admin_level strings, enforce.)
    al = chosen["admin_level"].astype(str).unique().tolist()
    if not any(a == "8" for a in al):
        raise RuntimeError(f"Non-commune admin_level found for {city}: {al}")

    return mp

def water_waste_share(rects: List[Polygon], city_geom: Polygon | MultiPolygon) -> float:
    """
    Estimate how much of the 'waste' (rectangles minus city) is water.
    """
    rect_union = unary_union(rects)
    waste_geom = rect_union.difference(city_geom)
    waste_area = waste_geom.area
    if waste_area <= 0:
        return 0.0

    minx, miny, maxx, maxy = rect_union.bounds
    # Fetch water features in this bbox
    water_tags = [
        {"natural": "water"},
        {"water": "lake"},
        {"water": "reservoir"},
        {"landuse": "reservoir"},
    ]
    water_union = None
    for tags in water_tags:
        try:
            w = ox.features_from_bbox(maxy, miny, maxx, minx, tags)
            if not w.empty:
                w = w[w.geometry.type.isin(["Polygon", "MultiPolygon"])]
                wu = unary_union(w.geometry.values)
                water_union = wu if water_union is None else water_union.union(wu)
        except Exception:
            continue

    if water_union is None:
        return 0.0

    water_in_waste = waste_geom.intersection(water_union).area
    return water_in_waste / waste_area if waste_area > 0 else 0.0

def make_rectangles_for_city(city: str) -> Tuple[dict, int, gpd.GeoSeries]:
    """
    Build rectangles for one city:
    - exclaves each get their own rectangle
    - main component gets smart slices only if efficient
    - auto-inflate padding to ensure 100% coverage
    - enforce <= 10 rectangles when possible (exclaves priority)
    Returns: (dict_of_json_entries, rectangle_count, GeoSeries_of_rectangles)
    """
    mp = fetch_commune_geom(city)
    components = list(mp.geoms)  # polygons
    components.sort(key=lambda g: g.area, reverse=True)
    main = components[0]
    exclaves = components[1:]

    # Start with exclave rectangles (always separate)
    rects = []
    for ex in exclaves:
        rects.append(rect_from_bounds(pad_bounds(ex.bounds, PADDING_DEG)))

    # Main component: start with 1 rectangle
    best_rects_for_main = [rect_from_bounds(pad_bounds(main.bounds, PADDING_DEG))]
    best_eff = main.area / union_area(best_rects_for_main)

    # If lots of waste, consider slicing—unless waste is mostly water.
    if best_eff < EFF_OK:
        water_share = water_waste_share(best_rects_for_main, main)
        if water_share < WATER_WASTE_SHARE_TO_ALLOW_ONE_BOX:
            for slices in range(2, MAX_SLICES_MAIN + 1):
                parts = slice_component(main, slices)
                cand = [rect_from_bounds(pad_bounds(p.bounds, PADDING_DEG)) for p in parts]
                # If slicing produced empties (degenerate), skip
                if not cand:
                    continue
                eff = main.area / union_area(cand)
                # accept only if efficiency gains meaningfully
                if (eff - best_eff) >= EFF_IMPROVE_MIN and eff >= EFF_OK:
                    best_eff = eff
                    best_rects_for_main = cand
                    # keep trying more slices to see if it’s even better
                # stop early if rectangles are already a lot
                if len(cand) + len(rects) >= MAX_RECTANGLES_PER_CITY:
                    break

    all_rects = rects + best_rects_for_main

    # If we exceed the target count, try a simpler main (1 rect), bigger padding
    if len(all_rects) > MAX_RECTANGLES_PER_CITY:
        # Revert main to 1 big rect
        all_rects = rects + [rect_from_bounds(pad_bounds(main.bounds, PADDING_DEG))]
        # If still >10 it’s likely too many exclaves; we keep them (warn later)

    # --- Coverage verification & auto-inflate padding ---
    # Inflate rectangles slightly until union covers 100% of city (including exclaves)
    city_union = unary_union(components)
    pad = PADDING_DEG
    inflated_rects = all_rects[:]
    def covered():
        return city_union.difference(unary_union(inflated_rects)).area <= 1e-12

    while not covered() and pad < MAX_PADDING_DEG:
        pad += PADDING_STEP
        inflated_rects = []
        # Rebuild exclave rects
        for ex in exclaves:
            inflated_rects.append(rect_from_bounds(pad_bounds(ex.bounds, pad)))
        # Rebuild main rects with same slicing we decided above
        if len(best_rects_for_main) == 1:
            inflated_rects.append(rect_from_bounds(pad_bounds(main.bounds, pad)))
        else:
            parts = slice_component(main, len(best_rects_for_main))
            inflated_rects.extend([rect_from_bounds(pad_bounds(p.bounds, pad)) for p in parts])

    # Recount and final check on count
    final_rects = inflated_rects
    count = len(final_rects)

    # If count still >10 and the reason is main slicing, collapse main to 1
    if count > MAX_RECTANGLES_PER_CITY and len(best_rects_for_main) > 1:
        final_rects = []
        for ex in exclaves:
            final_rects.append(rect_from_bounds(pad_bounds(ex.bounds, pad)))
        final_rects.append(rect_from_bounds(pad_bounds(main.bounds, pad)))
        count = len(final_rects)

    # Final coverage assertion (should be covered)
    if city_union.difference(unary_union(final_rects)).area > 1e-12:
        raise RuntimeError(f"Coverage check failed for {city}. Try increasing PADDING_DEG or MAX_PADDING_DEG.")

    # Build JSON entries and a GeoSeries for optional export
    entries = {}
    prefix = city.replace(" ", "_")
    gseries = gpd.GeoSeries(final_rects, crs="EPSG:4326")

    # If we STILL exceed 10 because of many exclaves, we keep them but flag in metadata
    warn = ""
    if count > MAX_RECTANGLES_PER_CITY:
        warn = f" (WARNING: {city} produced {count} rectangles due to exclaves; exclaves kept separate as requested.)"

    idx = 1
    for r in final_rects:
        key, val = bounds_to_dict(r.bounds, prefix, idx)
        entries[key] = val
        idx += 1

    return entries, count, gseries

# --------------- Driver ---------------

def main():
    all_entries = {}
    summary_rows = []
    rect_geojson_features = []

    for city in CITIES:
        print(f"\n=== {city} ===")
        try:
            entries, count, gseries = make_rectangles_for_city(city)
            all_entries.update(entries)
            print(f"Rectangles used: {count}")
            summary_rows.append({"city": city, "rectangles": count})

            # Save per-city rectangles to collect later
            for i, geom in enumerate(gseries, start=1):
                rect_geojson_features.append({
                    "type": "Feature",
                    "properties": {"city": city, "zone": f"{city.replace(' ','_')}_Zone{i}"},
                    "geometry": gpd.GeoSeries([geom], crs="EPSG:4326").__geo_interface__["features"][0]["geometry"],
                })
        except Exception as e:
            print(f"ERROR for {city}: {e}")
            summary_rows.append({"city": city, "rectangles": 0})

    # Write JSON
    Path("swiss_city_zones.json").write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote swiss_city_zones.json with {len(all_entries)} zones.")

    # Write summary CSV
    import csv
    with open("summary_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["city", "rectangles"])
        w.writeheader()
        w.writerows(summary_rows)
    print("Wrote summary_counts.csv")

    # Write rectangles + boundaries for optional visual QA
    rect_fc = {"type": "FeatureCollection", "features": rect_geojson_features}
    Path("rectangles.geojson").write_text(json.dumps(rect_fc, ensure_ascii=False), encoding="utf-8")
    print("Wrote rectangles.geojson (drop into https://geojson.io to visually inspect).")

if __name__ == "__main__":
    main()
