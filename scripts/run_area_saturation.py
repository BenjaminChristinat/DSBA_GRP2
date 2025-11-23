#!/usr/bin/env python3
"""Aggregate Model 2 expectations vs observed winners per grid cell."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer

from src.models.train_models import select_feature_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute over/under-saturation metrics per grid cell.")
    parser.add_argument("--step", type=int, default=100, help="Grid spacing (meters) for cell assignment.")
    parser.add_argument("--bbox-pad", type=float, default=500.0, help="Padding (m) for city bounding boxes.")
    parser.add_argument(
        "--rating-thr",
        type=float,
        default=4.3,
        help="Rating threshold used for observed success (matches Model 2 training).",
    )
    parser.add_argument(
        "--city-quantile",
        type=float,
        default=0.70,
        help="City-level review-count quantile to define success labels (matches Model 2 training).",
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/features/location_competition_features.parquet"),
        help="Feature table produced by src/features/build_features.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/model2_lgbm_best.pkl"),
        help="Joblib pipeline for Model 2 (LightGBM).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/area_perf.parquet"),
        help="Where to write the aggregated parquet file.",
    )
    return parser.parse_args()


def prepare_feature_matrix(df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    num_cols, cat_cols = select_feature_columns(df.columns, include_rating_cols=False)
    feature_cols = num_cols + cat_cols
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    model = joblib.load(model_path)
    df["pred_success_prob"] = model.predict_proba(X)[:, 1]
    return df


def compute_observed_success(df: pd.DataFrame, rating_thr: float, quantile: float) -> pd.DataFrame:
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["user_rating_count"] = pd.to_numeric(df["user_rating_count"], errors="coerce")
    thresh = df.groupby("__SelectedCity__")["user_rating_count"].transform(lambda s: s.quantile(quantile))
    df["observed_success"] = ((df["rating"] >= rating_thr) & (df["user_rating_count"] >= thresh)).astype(int)
    return df


def attach_grid_cells(
    df: pd.DataFrame,
    transformer: Transformer,
    step: float,
    pad: float,
) -> pd.DataFrame:
    results: List[pd.DataFrame] = []
    for city, sub in df.groupby("__SelectedCity__", dropna=False):
        lon = pd.to_numeric(sub["longitude"], errors="coerce")
        lat = pd.to_numeric(sub["latitude"], errors="coerce")
        mask = lon.notna() & lat.notna()
        if mask.sum() == 0:
            continue
        lon_vals = lon[mask].to_numpy()
        lat_vals = lat[mask].to_numpy()
        x_vals, y_vals = transformer.transform(lon_vals, lat_vals)
        min_x = x_vals.min() - pad
        min_y = y_vals.min() - pad
        max_x = x_vals.max() + pad
        max_y = y_vals.max() + pad
        if min_x == max_x or min_y == max_y:
            continue
        x_idx = np.floor((x_vals - min_x) / step).astype(int)
        y_idx = np.floor((y_vals - min_y) / step).astype(int)
        x_center = min_x + (x_idx + 0.5) * step
        y_center = min_y + (y_idx + 0.5) * step
        tmp = sub[mask].copy()
        city_name = city if isinstance(city, str) else "Unknown"
        tmp["grid_city"] = city_name
        tmp["grid_x_idx"] = x_idx
        tmp["grid_y_idx"] = y_idx
        tmp["grid_x_lv95"] = x_center
        tmp["grid_y_lv95"] = y_center
        tmp["grid_id"] = tmp["grid_city"] + ":" + tmp["grid_x_idx"].astype(str) + ":" + tmp["grid_y_idx"].astype(str)
        results.append(tmp)
    if not results:
        raise ValueError("No rows with valid coordinates available for grid assignment.")
    return pd.concat(results, ignore_index=True)


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.features_path)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

    df = prepare_feature_matrix(df, args.model_path)
    df = compute_observed_success(df, args.rating_thr, args.city_quantile)
    df = attach_grid_cells(df, transformer, args.step, args.bbox_pad)

    grouped = (
        df.groupby(["grid_id", "grid_city"])
        .agg(
            n_restaurants=("grid_id", "size"),
            expected_successes=("pred_success_prob", "sum"),
            observed_successes=("observed_success", "sum"),
            grid_x_lv95=("grid_x_lv95", "first"),
            grid_y_lv95=("grid_y_lv95", "first"),
            grid_x_idx=("grid_x_idx", "first"),
            grid_y_idx=("grid_y_idx", "first"),
        )
        .reset_index()
    )
    grouped["diff"] = grouped["observed_successes"] - grouped["expected_successes"]
    grouped["ratio"] = grouped["observed_successes"] / grouped["expected_successes"].clip(lower=0.1)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_parquet(args.output_path, index=False)
    print(f"Wrote {args.output_path} ({len(grouped)} rows)")


if __name__ == "__main__":
    main()
