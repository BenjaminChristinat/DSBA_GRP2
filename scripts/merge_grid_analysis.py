#!/usr/bin/env python3
"""
Merge all grid concept/best parquet files into a single dataset.

Usage:
    python scripts/merge_grid_analysis.py \
        --concepts-glob 'data/processed/grid_concepts_*.parquet' \
        --best-glob 'data/processed/grid_best_*.parquet' \
        --output data/processed/grid_analysis_all.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_COLUMNS = [
    "grid_id",
    "__SelectedCity__",
    "x_idx",
    "y_idx",
    "x_lv95",
    "y_lv95",
    "concept_name",
    "cuisine_slug",
    "est_flag",
    "price_level_num",
    "success_prob",
    "best_concept_name",
    "best_area_potential",
    "concept_source_file",
    "best_source_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-city grid outputs into a single parquet file."
    )
    parser.add_argument(
        "--concepts-glob",
        default="data/processed/grid_concepts_*.parquet",
        help="Glob pattern covering all concept parquet files.",
    )
    parser.add_argument(
        "--best-glob",
        default="data/processed/grid_best_*.parquet",
        help="Glob pattern covering all best-concept parquet files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/grid_analysis_all.parquet",
        help="Destination parquet file.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices={"snappy", "zstd", "gzip", "brotli", "none"},
        help="Parquet compression codec (defaults to zstd for better ratio).",
    )
    return parser.parse_args()


def _match_best_file(concept_path: str, best_files: set[str]) -> str:
    stem = Path(concept_path).name.replace("grid_concepts_", "").replace(".parquet", "")
    candidate = f"data/processed/grid_best_{stem}.parquet"
    if candidate in best_files:
        return candidate
    msg = f"Missing best file for '{stem}'. Expected {candidate}"
    raise FileNotFoundError(msg)


def _iter_city_frames(
    concept_paths: Iterable[str], best_paths: set[str]
) -> Iterable[pd.DataFrame]:
    for concept_path in concept_paths:
        concept_df = pd.read_parquet(concept_path).copy()
        concept_df["concept_source_file"] = Path(concept_path).name

        best_path = _match_best_file(concept_path, best_paths)
        best_df = pd.read_parquet(best_path).copy()
        best_df = best_df.rename(
            columns={
                "best_concept": "best_concept_name",
                "area_potential": "best_area_potential",
            }
        )
        best_df["best_source_file"] = Path(best_path).name

        merged = concept_df.merge(
            best_df[
                [
                    "grid_id",
                    "__SelectedCity__",
                    "x_idx",
                    "y_idx",
                    "x_lv95",
                    "y_lv95",
                    "best_concept_name",
                    "best_area_potential",
                    "best_source_file",
                ]
            ],
            on=["grid_id", "__SelectedCity__", "x_idx", "y_idx", "x_lv95", "y_lv95"],
            how="left",
        )
        yield merged[DEFAULT_COLUMNS]


def main() -> None:
    args = parse_args()
    concept_paths = sorted(glob.glob(args.concepts_glob))
    if not concept_paths:
        raise SystemExit(f"No files match {args.concepts_glob}")
    best_paths = set(glob.glob(args.best_glob))
    if not best_paths:
        raise SystemExit(f"No files match {args.best_glob}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    parquet_writer = None
    try:
        for frame in _iter_city_frames(concept_paths, best_paths):
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if parquet_writer is None:
                compression = None if args.compression == "none" else args.compression
                parquet_writer = pq.ParquetWriter(
                    output, table.schema, compression=compression
                )
            parquet_writer.write_table(table)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"Wrote {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
