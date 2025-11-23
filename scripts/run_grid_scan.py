#!/usr/bin/env python3
"""Grid-scan Model 2 success probabilities across all cities and concepts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyproj import Transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.features.build_features import (
    COMP_HIGH_RATING_FEATURE,
    COMP_RADIUS_FEATURES,
    DEFAULT_EXTENDED_PATH,
    DEFAULT_MICRO_POI_PATH,
    MICRO_POI_FEATURES,
    add_city_dummies,
    add_log_transforms,
    clip_and_log_distances,
    compute_competition_radius_counts,
    compute_knn_competition_features,
    compute_microlocation_counts,
    compute_nearest_distance_features,
    coords_valid_mask,
    filter_competition_layer,
    load_microlocation_table,
    load_restaurant_table,
    project_points,
)
# Use original selection logic if available, else fallback
try:
    from src.models.train_models import select_feature_columns
except ImportError:
    select_feature_columns = None


@dataclass(frozen=True)
class Concept:
    cuisine_slug: str
    est_flag: str
    price_level: int

    @property
    def cuisine_column(self) -> str:
        return f"cuisine_{self.cuisine_slug}"

    @property
    def name(self) -> str:
        est_slug = self.est_flag.replace("is_", "")
        return f"{self.cuisine_slug}_{est_slug}_p{self.price_level}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Swiss cities for high-potential grid cells with Model 2.")
    parser.add_argument("--city", help="Process only this city.")
    parser.add_argument("--cities", nargs="+", help="Explicit list of cities.")
    parser.add_argument("--step", type=int, default=100, help="Grid spacing in meters.")
    parser.add_argument("--bbox-pad", type=float, default=500.0, help="Padding (m) added to each city bounding box.")
    parser.add_argument("--max-dist-km", type=float, default=1.0, help="Drop grid cells farther than this distance (km).")
    parser.add_argument("--concepts-path", type=Path, help="Optional CSV with columns cuisine,est_flag,price_level.")
    parser.add_argument("--concept-chunk-size", type=int, default=25, help="How many concepts to score at a time.")
    parser.add_argument("--micropoi-path", type=Path, default=DEFAULT_MICRO_POI_PATH)
    parser.add_argument("--competition-path", type=Path, default=DEFAULT_EXTENDED_PATH)
    parser.add_argument("--model-path", type=Path, default=Path("models/model2_lgbm_best.pkl"))
    parser.add_argument("--features-path", type=Path, default=Path("data/features/location_competition_features.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def determine_cities(args: argparse.Namespace, available: List[str]) -> List[str]:
    if args.cities:
        requested = args.cities
    elif args.city:
        requested = available if args.city.lower() == "all" else [args.city]
    else:
        requested = available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"Requested cities not found in feature table: {', '.join(missing)}")
    return sorted(dict.fromkeys(requested))


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        zeros = pd.DataFrame(0, index=df.index, columns=missing)
        df = pd.concat([df, zeros], axis=1)
    return df


def build_city_bbox(coords: np.ndarray, pad_m: float) -> tuple[float, float, float, float]:
    xs = coords[:, 0]
    ys = coords[:, 1]
    return xs.min() - pad_m, xs.max() + pad_m, ys.min() - pad_m, ys.max() + pad_m


def make_grid(bbox: Sequence[float], step: float, city: str) -> pd.DataFrame:
    min_x, max_x, min_y, max_y = bbox
    xs = np.arange(min_x, max_x, step)
    ys = np.arange(min_y, max_y, step)
    centers_x = xs + step / 2
    centers_y = ys + step / 2
    data = []
    for ix, cx in enumerate(centers_x):
        for iy, cy in enumerate(centers_y):
            data.append((city, ix, iy, cx, cy))
    if not data:
        raise ValueError("Grid generation produced zero cells – adjust bbox or step.")
    return pd.DataFrame(data, columns=["__SelectedCity__", "x_idx", "y_idx", "x_lv95", "y_lv95"])


def filter_by_distance(grid: pd.DataFrame, rest_coords: np.ndarray, max_dist_km: float) -> pd.DataFrame:
    if rest_coords.size == 0:
        return grid
    from sklearn.neighbors import KDTree
    tree = KDTree(rest_coords, metric="euclidean")
    coords = grid[["x_lv95", "y_lv95"]].to_numpy()
    distances, _ = tree.query(coords, k=1)
    keep = distances[:, 0] <= (max_dist_km * 1000.0)
    return grid.loc[keep].reset_index(drop=True)


def build_auto_concepts(cuisine_cols: List[str], est_cols: List[str], price_levels: List[int]) -> List[Concept]:
    combos: List[Concept] = []
    for cuisine_col, est_flag, price in product(cuisine_cols, est_cols, price_levels):
        slug = cuisine_col.replace("cuisine_", "")
        combos.append(Concept(slug, est_flag, int(price)))
    return combos


def load_concepts(concepts_path: Path | None, cuisine_cols: List[str], est_cols: List[str], price_levels: List[int]) -> List[Concept]:
    if concepts_path is None:
        return build_auto_concepts(cuisine_cols, est_cols, price_levels)
    df = pd.read_csv(concepts_path)
    concept_list = []
    for row in df.itertuples(index=False):
        concept_list.append(Concept(
            str(row.cuisine).strip().lower().replace(" ", "_"),
            str(row.est_flag).strip(),
            int(row.price_level)
        ))
    return concept_list


def chunked(iterable: Sequence[Concept], size: int) -> Iterator[List[Concept]]:
    for start in range(0, len(iterable), size):
        yield list(iterable[start : start + size])


def main() -> None:
    args = parse_args()
    features = pd.read_parquet(args.features_path)
    # Fix coords
    features["longitude"] = pd.to_numeric(features["longitude"], errors="coerce")
    features["latitude"] = pd.to_numeric(features["latitude"], errors="coerce")
    features = features.dropna(subset=["longitude", "latitude"])

    available_cities = sorted(features["__SelectedCity__"].dropna().unique().tolist())
    city_list = determine_cities(args, available_cities)

    model = joblib.load(args.model_path)
    if hasattr(model, "feature_names_in_"):
        feature_cols = list(model.feature_names_in_)
    elif hasattr(model, "steps") and hasattr(model.steps[-1][1], "feature_names_in_"):
        feature_cols = list(model.steps[-1][1].feature_names_in_)
    else:
        # Fallback
        num_cols = [c for c in features.columns if features[c].dtype in (float, int)]
        feature_cols = num_cols

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)
    poi = load_microlocation_table(args.micropoi_path)
    comp_layer = load_restaurant_table(args.competition_path)
    comp_filtered = filter_competition_layer(comp_layer)
    comp_coords_all, _ = project_points(comp_filtered["longitude"], comp_filtered["latitude"], transformer)
    comp_valid_mask = coords_valid_mask(comp_coords_all)
    comp_valid = comp_filtered.iloc[np.flatnonzero(comp_valid_mask)].reset_index(drop=True)
    comp_coords_valid = comp_coords_all[comp_valid_mask]

    # Prepare Indices
    price_levels = sorted({int(x) for x in features["price_level_num"].dropna().unique()}) or [0]
    cuisine_cols = [col for col in feature_cols if col.startswith("cuisine_")]
    est_cols = [col for col in feature_cols if col.startswith("is_")]
    concepts = load_concepts(args.concepts_path, cuisine_cols, est_cols, price_levels)

    feature_idx = {name: idx for idx, name in enumerate(feature_cols)}
    price_idx = feature_idx.get("price_level_num")
    cuisine_idx = {col: feature_idx.get(col) for col in cuisine_cols if col in feature_idx}
    est_idx = {col: feature_idx.get(col) for col in est_cols if col in feature_idx}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for city in city_list:
        print(f"Processing {city}...")
        city_slice = features[features["__SelectedCity__"] == city].copy()
        coords, mask = project_points(city_slice["longitude"], city_slice["latitude"], transformer)
        coords = coords[mask]
        
        if coords.size == 0: 
            continue

        bbox = build_city_bbox(coords, args.bbox_pad)
        grid = make_grid(bbox, args.step, city)
        grid = filter_by_distance(grid, coords, args.max_dist_km)
        
        if grid.empty: 
            continue

        base = grid.copy()
        base["grid_id"] = (
            base["__SelectedCity__"] + ":" +
            base["x_idx"].astype(int).astype(str) + ":" +
            base["y_idx"].astype(int).astype(str)
        )
        
        # --- COMPUTE FEATURES FOR GRID ---
        core_coords = base[["x_lv95", "y_lv95"]].to_numpy()
        core_mask = np.isfinite(core_coords).all(axis=1)

        compute_microlocation_counts(base, core_coords, core_mask, poi, transformer)
        compute_nearest_distance_features(base, core_coords, core_mask, poi, transformer)
        clip_and_log_distances(base, ("dist_station_m", "dist_hotel_m"))

        if comp_coords_valid.size > 0:
            compute_competition_radius_counts(base, core_coords, core_mask, comp_valid, comp_coords_valid, nn_pos_by_id={})
            # KNN omitted for speed in grid - usually not critical for simple Concept Page
        
        add_city_dummies(base)
        count_cols = [cfg[0] for cfg in MICRO_POI_FEATURES] + [cfg[0] for cfg in COMP_RADIUS_FEATURES] + [COMP_HIGH_RATING_FEATURE[0]]
        add_log_transforms(base, count_cols)
        base = ensure_columns(base, feature_cols)
        if "price_level_num" not in base.columns: base["price_level_num"] = 0

        # --- SAVE FEATURES FOR PAGE 2 ---
        # We save the raw calculated features (dist_station, etc.) before they get lost
        feature_columns_to_keep = [
            "grid_id", "x_lv95", "y_lv95", 
            "dist_station_m", "comp_count_500m", "log_attractions_500m"
        ]
        # Add any others if they exist
        existing_cols = [c for c in feature_columns_to_keep if c in base.columns]
        grid_features = base[existing_cols].copy()
        
        city_slug = city.lower().replace(" ", "_")
        out_features = args.output_dir / f"grid_features_{city_slug}.parquet"
        grid_features.to_parquet(out_features, index=False)
        print(f"   -> Features saved: {out_features}")

        # --- PREDICTIONS FOR PAGE 1/Map ---
        base_matrix = base[feature_cols].to_numpy(copy=True)
        n_cells = len(base)
        grid_meta_cols = ["grid_id", "__SelectedCity__", "x_idx", "y_idx", "x_lv95", "y_lv95"]
        grid_meta = base[grid_meta_cols].copy()
        
        best_prob = np.full(n_cells, -np.inf, dtype=float)
        best_concept = np.array([""] * n_cells, dtype=object)
        
        out_all = args.output_dir / f"grid_concepts_{city_slug}.parquet"
        writer = None

        try:
            # Simplified: Just predict probability, don't save every single row (too big)
            # We only save the 'Best' concept per cell to keep it fast
            meta_values = grid_meta.to_numpy()
            
            for chunk in chunked(concepts, max(1, args.concept_chunk_size)):
                chunk_size = len(chunk)
                X_chunk = np.tile(base_matrix, (chunk_size, 1))
                
                for idx, concept in enumerate(chunk):
                    rows = slice(idx * n_cells, (idx + 1) * n_cells)
                    block = X_chunk[rows]
                    if price_idx is not None: block[:, price_idx] = concept.price_level
                    if concept.cuisine_column in cuisine_idx: block[:, cuisine_idx[concept.cuisine_column]] = 1
                    if concept.est_flag in est_idx: block[:, est_idx[concept.est_flag]] = 1
                
                X_df = pd.DataFrame(X_chunk, columns=feature_cols)
                probs = model.predict_proba(X_df)[:, 1]

                for idx, concept in enumerate(chunk):
                    rows = slice(idx * n_cells, (idx + 1) * n_cells)
                    block_probs = probs[rows]
                    better = block_probs > best_prob
                    best_prob[better] = block_probs[better]
                    best_concept[better] = concept.name
            
            # Save BEST concepts (Page 1 data)
            best_df = grid_meta.copy()
            best_df["best_concept"] = best_concept
            best_df["area_potential"] = best_prob
            
            # Merge features back in for convenience
            best_df = best_df.merge(grid_features, on=["grid_id", "x_lv95", "y_lv95"])
            
            out_best = args.output_dir / f"grid_concepts_{city_slug}.parquet" # Saving as concepts for Page 1
            best_df.to_parquet(out_best, index=False)
            print(f"   -> Predictions saved: {out_best}")

        except Exception as e:
            print(f"Error processing concepts for {city}: {e}")

if __name__ == "__main__":
    main()