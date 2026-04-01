#!/usr/bin/env python3
"""
Extract per-image feature counts from a COLMAP database.db file.

COLMAP stores feature data in a SQLite database. This script reads:
- images table
- keypoints table

and outputs a CSV with one row per image:
    image_id,image_name,num_keypoints,keypoint_cols

Usage:
    python extract_colmap_features.py /path/to/database.db
    python extract_colmap_features.py /path/to/database.db --out features.csv

Example:
    python extract_colmap_features.py fold_1/database.db --out fold_1_features.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-image feature counts from a COLMAP database."
    )
    parser.add_argument(
        "database",
        help="Path to COLMAP database.db",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output CSV path. Defaults to <database_dir>/feature_counts.csv",
    )
    return parser.parse_args()


def validate_db_path(db_path: str) -> None:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")


def fetch_feature_counts(db_path: str) -> List[Tuple[int, str, int, int]]:
    """
    Returns rows of:
        (image_id, image_name, num_keypoints, keypoint_cols)

    In COLMAP's keypoints table:
    - rows = number of keypoints
    - cols = keypoint dimensionality (commonly 2, 4, or 6 depending on settings)
    """
    query = """
        SELECT
            i.image_id,
            i.name,
            COALESCE(k.rows, 0) AS num_keypoints,
            COALESCE(k.cols, 0) AS keypoint_cols
        FROM images i
        LEFT JOIN keypoints k
            ON i.image_id = k.image_id
        ORDER BY i.name;
    """

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    finally:
        conn.close()

    return rows


def write_csv(rows: List[Tuple[int, str, int, int]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_name", "num_keypoints", "keypoint_cols"])
        writer.writerows(rows)


def print_summary(rows: List[Tuple[int, str, int, int]]) -> None:
    counts = [row[2] for row in rows]

    if not counts:
        print("No images found in database.")
        return

    total_images = len(counts)
    total_features = sum(counts)
    min_features = min(counts)
    max_features = max(counts)
    mean_features = total_features / total_images
    weak_under_2000 = sum(1 for c in counts if c < 2000)

    print(f"Images: {total_images}")
    print(f"Total keypoints: {total_features}")
    print(f"Mean keypoints/image: {mean_features:.2f}")
    print(f"Min keypoints/image: {min_features}")
    print(f"Max keypoints/image: {max_features}")
    print(
        f"Images with <2000 keypoints: {weak_under_2000} "
        f"({100 * weak_under_2000 / total_images:.1f}%)"
    )

    print("\nTop 10 strongest images:")
    for image_id, image_name, num_keypoints, keypoint_cols in sorted(
        rows, key=lambda r: r[2], reverse=True
    )[:10]:
        print(
            f"  {image_name:<24}  {num_keypoints:>5} keypoints "
            f"(image_id={image_id}, cols={keypoint_cols})"
        )

    print("\nTop 10 weakest images:")
    for image_id, image_name, num_keypoints, keypoint_cols in sorted(
        rows, key=lambda r: r[2]
    )[:10]:
        print(
            f"  {image_name:<24}  {num_keypoints:>5} keypoints "
            f"(image_id={image_id}, cols={keypoint_cols})"
        )


def main() -> int:
    args = parse_args()

    try:
        validate_db_path(args.database)
        rows = fetch_feature_counts(args.database)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(args.database)),
            "feature_counts.csv",
        )

    try:
        write_csv(rows, out_path)
    except Exception as e:
        print(f"Failed to write CSV: {e}", file=sys.stderr)
        return 1

    print_summary(rows)
    print(f"\nWrote CSV to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
