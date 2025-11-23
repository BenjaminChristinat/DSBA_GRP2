#!/usr/bin/env python3
"""
Helper utilities for pulling Google Maps reviews via Bright Data.

The script covers three stages:
1. Preparing payloads from FINAL_MERGED_WITH_OSM_GOOGLE.csv
2. Triggering + monitoring Bright Data dataset runs
3. Post-processing the downloaded snapshot into a clean time-series CSV

Usage examples are documented in docs/brightdata_reviews.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

DEFAULT_DATASET_ID = "gd_luzfs1dn2oa0teb81"
DEFAULT_INPUT_CSV = Path("data/interim/FINAL_MERGED_WITH_OSM_GOOGLE.csv")
DEFAULT_DAYS_LIMIT = 2557  # ~7 years (365.25 * 7)
DEFAULT_CITIES = ("Zurich", "Geneva", "Nyon")
PLACE_ID_PDP_TEMPLATE = "https://www.google.com/maps/place/?q=place_id:{place_id}"


def _load_place_ids(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return ids or None


def _build_urls(df: pd.DataFrame, use_place_id_url: bool) -> pd.Series:
    if use_place_id_url:
        if df["id"].isna().any():
            raise SystemExit("Cannot build place_id URLs when 'id' has missing values.")
        return df["id"].apply(
            lambda pid: PLACE_ID_PDP_TEMPLATE.format(place_id=pid) if pid else ""
        )
    return df["google_maps_uri"]


def _ensure_payload(df: pd.DataFrame, days_limit: int, urls: pd.Series) -> List[dict]:
    payload = []
    for url in urls:
        if not url:
            continue
        payload.append({"url": url, "days_limit": days_limit})
    if not payload:
        raise SystemExit("No rows remaining for payload. Check your filters/URL choice.")
    return payload


def _filter_restaurants(
    df: pd.DataFrame,
    cities: Iterable[str],
    place_ids: Optional[Iterable[str]],
) -> pd.DataFrame:
    if place_ids:
        df = df[df["id"].isin(place_ids)]
    else:
        df = df[df["__SelectedCity__"].isin(cities)]
    df = df[df["google_maps_uri"].notna()].copy()
    df["user_rating_count"] = df["user_rating_count"].fillna(0)
    return df


def _select_with_target(
    df: pd.DataFrame,
    limit: Optional[int],
    target_total_reviews: Optional[int],
    order_by: str,
    ascending: bool,
) -> pd.DataFrame:
    work = df.copy()
    if target_total_reviews and limit:
        target_per_place = target_total_reviews / limit
        work["_score"] = (work["user_rating_count"] - target_per_place).abs()
        work = work.sort_values("_score")
    else:
        work = work.sort_values(order_by, ascending=ascending)
    if limit:
        work = work.head(limit)
    return work.drop(columns=[c for c in ["_score"] if c in work.columns])


def cmd_prepare(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input_csv)
    place_ids = args.place_ids or _load_place_ids(args.place_ids_file)
    filtered = _filter_restaurants(df, args.cities, place_ids)
    selected = _select_with_target(
        filtered,
        args.limit,
        args.target_total_reviews,
        args.order_by,
        args.ascending,
    )

    urls = _build_urls(selected, args.use_place_id_url)
    selected = selected.assign(brightdata_url=urls)

    payload = _ensure_payload(selected, args.days_limit, urls)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote payload with {len(payload)} urls -> {output_path}")

    if args.selection_output:
        selection_cols = [
            "id",
            "display_name",
            "__SelectedCity__",
            "user_rating_count",
            "google_maps_uri",
            "brightdata_url",
        ]
        sel_path = Path(args.selection_output)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        selected[selection_cols].to_csv(sel_path, index=False)
        print(f"Wrote selection metadata -> {sel_path}")


def _resolve_api_key(args: argparse.Namespace) -> str:
    api_key = args.api_key or os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            "Missing Bright Data API key. "
            "Pass --api-key or export BRIGHTDATA_API_KEY."
        )
    return api_key


def _trigger_dataset(
    api_key: str,
    dataset_id: str,
    payload: List[dict],
    include_errors: bool,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        "https://api.brightdata.com/datasets/v3/trigger",
        headers=headers,
        params={"dataset_id": dataset_id, "include_errors": str(include_errors).lower()},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    snapshot_id = data["snapshot_id"]
    print(f"Triggered dataset. snapshot_id={snapshot_id}")
    return snapshot_id


def _poll_until_ready(
    api_key: str, snapshot_id: str, poll_interval: int, max_wait: int
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    waited = 0
    while True:
        resp = requests.get(
            f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}",
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        status = resp.json().get("status")
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Status: {status}")
        if status == "ready":
            return
        if status == "failed":
            raise SystemExit(f"Bright Data run failed: snapshot_id={snapshot_id}")
        time.sleep(poll_interval)
        waited += poll_interval
        if waited > max_wait:
            raise SystemExit(
                f"Timed out waiting for snapshot {snapshot_id} "
                f"after {waited} seconds."
            )


def _download_snapshot(api_key: str, snapshot_id: str) -> List[dict]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
        headers=headers,
        params={"format": "json"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def cmd_run(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.payload).read_text(encoding="utf-8"))
    api_key = _resolve_api_key(args)
    snapshot_id = _trigger_dataset(
        api_key=api_key,
        dataset_id=args.dataset_id,
        payload=payload,
        include_errors=args.include_errors,
    )
    run_dir = Path(args.output_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshot_id.txt").write_text(snapshot_id, encoding="utf-8")

    metadata = {
        "snapshot_id": snapshot_id,
        "dataset_id": args.dataset_id,
        "payload_file": str(Path(args.payload).resolve()),
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "days_limit": args.days_limit,
        "payload_size": len(payload),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Run metadata stored under {run_dir}")

    if args.no_wait:
        print("Skipping wait/download as requested.")
        return

    _poll_until_ready(api_key, snapshot_id, args.poll_interval, args.max_wait)

    if args.skip_download:
        print("Wait finished; download skipped per flag.")
        return

    records = _download_snapshot(api_key, snapshot_id)
    out_file = run_dir / args.output_json
    out_file.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} reviews -> {out_file}")


def cmd_postprocess(args: argparse.Namespace) -> None:
    reviews_path = Path(args.snapshot_json)
    reviews_raw = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not isinstance(reviews_raw, list):
        raise SystemExit("Snapshot JSON must contain an array of review rows.")

    reviews_df = pd.DataFrame(reviews_raw)
    required_cols = {"place_id", "review_rating", "review_date"}
    missing = required_cols - set(reviews_df.columns)
    if missing:
        raise SystemExit(f"Snapshot missing expected columns: {missing}")

    restaurants = pd.read_csv(args.restaurants_csv)
    if args.cities:
        restaurants = restaurants[restaurants["__SelectedCity__"].isin(args.cities)]

    resto_fields = (
        restaurants.drop_duplicates(subset=["id"])[
            ["id", "display_name", "__SelectedCity__", "google_maps_uri"]
        ]
    ).rename(
        columns={
            "id": "place_id",
            "__SelectedCity__": "city",
            "display_name": "restaurant_name",
        }
    )

    merged = reviews_df.merge(resto_fields, on="place_id", how="left")
    merged = merged[
        [
            "place_id",
            "restaurant_name",
            "city",
            "google_maps_uri",
            "review_date",
            "review_rating",
        ]
    ]
    merged = merged.sort_values(["place_id", "review_date"])

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote {len(merged)} review rows -> {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automate Bright Data Google Maps reviews extraction."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Build payload JSON from the master CSV.")
    prep.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        type=Path,
        help="Path to FINAL_MERGED_WITH_OSM_GOOGLE.csv",
    )
    prep.add_argument(
        "--cities",
        nargs="+",
        default=list(DEFAULT_CITIES),
        help="City filter. Ignored when --place-ids is used.",
    )
    prep.add_argument(
        "--place-ids",
        nargs="+",
        help="Explicit list of Google place IDs to include.",
    )
    prep.add_argument(
        "--place-ids-file",
        type=Path,
        help="File containing one place_id per line.",
    )
    prep.add_argument("--limit", type=int, help="Max number of restaurants to include.")
    prep.add_argument(
        "--target-total-reviews",
        type=int,
        help="Approximate total review count when combined with --limit.",
    )
    prep.add_argument(
        "--order-by",
        default="user_rating_count",
        help="Column used for default sorting.",
    )
    prep.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending instead of descending.",
    )
    prep.add_argument(
        "--days-limit",
        type=int,
        default=DEFAULT_DAYS_LIMIT,
        help="Number of days worth of reviews to request per place.",
    )
    prep.add_argument(
        "--output",
        required=True,
        help="Where to write the Bright Data payload JSON.",
    )
    prep.add_argument(
        "--selection-output",
        help="Optional CSV recording which restaurants are included.",
    )
    prep.add_argument(
        "--use-place-id-url",
        action="store_true",
        help="Build https://www.google.com/maps/place/?q=place_id:<id> URLs instead of using google_maps_uri.",
    )
    prep.set_defaults(func=cmd_prepare)

    run = subparsers.add_parser("run", help="Trigger/poll/download a Bright Data run.")
    run.add_argument("--payload", required=True, help="Payload JSON file to submit.")
    run.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Bright Data dataset id to trigger.",
    )
    run.add_argument(
        "--api-key",
        help="Bright Data API key. Defaults to BRIGHTDATA_API_KEY env var.",
    )
    run.add_argument(
        "--api-key-env",
        default="BRIGHTDATA_API_KEY",
        help="Environment variable to read the API key from.",
    )
    run.add_argument(
        "--include-errors",
        action="store_true",
        help="Request Bright Data to include per-record errors in the snapshot.",
    )
    run.add_argument(
        "--run-name",
        required=True,
        help="Directory name under --output-dir to store artifacts.",
    )
    run.add_argument(
        "--output-dir",
        default="data/external/brightdata_snapshots",
        help="Folder for run outputs.",
    )
    run.add_argument(
        "--poll-interval",
        type=int,
        default=20,
        help="Seconds between Bright Data status checks.",
    )
    run.add_argument(
        "--max-wait",
        type=int,
        default=3600,
        help="Abort if the run takes longer than this many seconds.",
    )
    run.add_argument(
        "--output-json",
        default="reviews.json",
        help="Filename (under run directory) for the downloaded snapshot.",
    )
    run.add_argument(
        "--no-wait",
        action="store_true",
        help="Trigger only; skip polling and download.",
    )
    run.add_argument(
        "--skip-download",
        action="store_true",
        help="Poll until ready but do not download the snapshot.",
    )
    run.add_argument(
        "--days-limit",
        type=int,
        default=DEFAULT_DAYS_LIMIT,
        help="Recorded in metadata for traceability.",
    )
    run.set_defaults(func=cmd_run)

    post = subparsers.add_parser(
        "postprocess",
        help="Join Bright Data snapshot with master CSV to build a time series.",
    )
    post.add_argument(
        "--snapshot-json",
        required=True,
        help="reviews.json file downloaded from Bright Data.",
    )
    post.add_argument(
        "--restaurants-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to FINAL_MERGED_WITH_OSM_GOOGLE.csv",
    )
    post.add_argument(
        "--cities",
        nargs="+",
        help="Optional city filter when enriching reviews.",
    )
    post.add_argument(
        "--output-csv",
        default="data/processed/reviews_timeseries_zh_ge_nyon.csv",
        help="Where to write the final time-series CSV.",
    )
    post.set_defaults(func=cmd_postprocess)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
