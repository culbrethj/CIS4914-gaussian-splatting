#!/usr/bin/env python3
"""
Analyze COLMAP feature data across one or more database.db files.

What it does:
- Reads per-image keypoint counts from each COLMAP SQLite database
- Extracts frame index from image names like frame_000123.jpg
- Computes per-fold summary statistics
- Writes per-image and summary CSVs
- Generates charts:
    1) overlaid histogram of keypoints/image
    2) boxplot by fold
    3) keypoints vs frame index
    4) empirical CDF by fold
    5) weak-frame percentage bar chart for thresholds 1500, 2000, 2500

Example:
    python analyze_colmap_folds.py \
        --db fold_1/database.db \
        --db fold_2/database.db \
        --db fold_3/database.db \
        --labels fold_1 fold_2 fold_3 \
        --outdir analysis_output
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze COLMAP keypoint counts across folds."
    )
    parser.add_argument(
        "--db",
        action="append",
        required=True,
        help="Path to a COLMAP database.db file. Pass once per fold.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each database, same order as --db.",
    )
    parser.add_argument(
        "--outdir",
        default="colmap_feature_analysis",
        help="Output directory for CSVs and charts.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.labels is not None and len(args.labels) != len(args.db):
        raise ValueError("Number of --labels entries must match number of --db entries.")

    for db_path in args.db:
        if not os.path.isfile(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")


def extract_frame_index(image_name: str) -> int | None:
    match = FRAME_RE.search(image_name)
    return int(match.group(1)) if match else None


def load_fold_from_db(db_path: str, fold_label: str) -> pd.DataFrame:
    query = """
        SELECT
            i.image_id,
            i.name AS image_name,
            COALESCE(k.rows, 0) AS num_keypoints,
            COALESCE(k.cols, 0) AS keypoint_cols
        FROM images i
        LEFT JOIN keypoints k
            ON i.image_id = k.image_id
        ORDER BY i.name;
    """

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    df["fold"] = fold_label
    df["database_path"] = str(Path(db_path).resolve())
    df["frame_index"] = df["image_name"].apply(extract_frame_index)

    return df


def summarize_fold(df: pd.DataFrame) -> dict:
    counts = df["num_keypoints"]

    summary = {
        "fold": df["fold"].iloc[0],
        "num_images": int(len(df)),
        "total_keypoints": int(counts.sum()),
        "mean_keypoints": float(counts.mean()),
        "median_keypoints": float(counts.median()),
        "std_keypoints": float(counts.std(ddof=1)) if len(df) > 1 else 0.0,
        "min_keypoints": int(counts.min()),
        "max_keypoints": int(counts.max()),
        "pct_below_1500": float((counts < 1500).mean() * 100),
        "pct_below_2000": float((counts < 2000).mean() * 100),
        "pct_below_2500": float((counts < 2500).mean() * 100),
    }
    return summary


def save_csvs(all_df: pd.DataFrame, summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(outdir / "per_image_keypoints.csv", index=False)
    summary_df.to_csv(outdir / "fold_summary.csv", index=False)


def make_histogram(all_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 6))
    bins = 25

    for fold_name, group in all_df.groupby("fold"):
        plt.hist(group["num_keypoints"], bins=bins, alpha=0.45, label=fold_name)

    plt.xlabel("Keypoints per image")
    plt.ylabel("Number of images")
    plt.title("Histogram of keypoints per image by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "histogram_keypoints_by_fold.png", dpi=200)
    plt.close()


def make_boxplot(all_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    order = list(all_df["fold"].drop_duplicates())
    data = [all_df.loc[all_df["fold"] == fold, "num_keypoints"] for fold in order]

    plt.boxplot(data, tick_labels=order)
    plt.xlabel("Fold")
    plt.ylabel("Keypoints per image")
    plt.title("Boxplot of keypoints per image by fold")
    plt.tight_layout()
    plt.savefig(outdir / "boxplot_keypoints_by_fold.png", dpi=200)
    plt.close()


def make_frame_index_plot(all_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(11, 6))

    for fold_name, group in all_df.groupby("fold"):
        group = group.sort_values("frame_index")
        plt.plot(group["frame_index"], group["num_keypoints"], marker="o", linewidth=1, markersize=3, label=fold_name)

    plt.xlabel("Frame index")
    plt.ylabel("Keypoints per image")
    plt.title("Keypoints vs frame index")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "keypoints_vs_frame_index.png", dpi=200)
    plt.close()


def make_ecdf(all_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 6))

    for fold_name, group in all_df.groupby("fold"):
        x = group["num_keypoints"].sort_values().to_numpy()
        y = (pd.Series(range(1, len(x) + 1)) / len(x)).to_numpy()
        plt.plot(x, y, label=fold_name)

    plt.xlabel("Keypoints per image")
    plt.ylabel("Empirical cumulative probability")
    plt.title("Empirical CDF of keypoints per image by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "ecdf_keypoints_by_fold.png", dpi=200)
    plt.close()


def make_weak_frame_bars(all_df: pd.DataFrame, outdir: Path) -> None:
    thresholds = [1500, 2000, 2500]
    rows = []

    for fold_name, group in all_df.groupby("fold"):
        for threshold in thresholds:
            pct = float((group["num_keypoints"] < threshold).mean() * 100)
            rows.append(
                {
                    "fold": fold_name,
                    "threshold": f"<{threshold}",
                    "percent": pct,
                }
            )

    weak_df = pd.DataFrame(rows)
    weak_df.to_csv(outdir / "weak_frame_percentages.csv", index=False)

    pivot = weak_df.pivot(index="threshold", columns="fold", values="percent")
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_xlabel("Weak-frame threshold")
    ax.set_ylabel("Percent of images")
    ax.set_title("Weak-frame percentage by fold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "weak_frame_percentages.png", dpi=200)
    plt.close()


def print_console_summary(summary_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nFold summary:\n")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main() -> int:
    args = parse_args()
    validate_args(args)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    labels = args.labels if args.labels is not None else [
        f"fold_{i+1}" for i in range(len(args.db))
    ]

    fold_dfs = []
    for db_path, label in zip(args.db, labels):
        df = load_fold_from_db(db_path, label)
        fold_dfs.append(df)

    all_df = pd.concat(fold_dfs, ignore_index=True)
    summary_df = pd.DataFrame([summarize_fold(df) for df in fold_dfs])

    save_csvs(all_df, summary_df, outdir)
    make_histogram(all_df, outdir)
    make_boxplot(all_df, outdir)
    make_frame_index_plot(all_df, outdir)
    make_ecdf(all_df, outdir)
    make_weak_frame_bars(all_df, outdir)

    print_console_summary(summary_df)
    print(f"\nSaved outputs to: {outdir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""
python feature_analysis.py \
  --db datasets/speaker_fold1/sparse/database.db \
  --db datasets/speaker_fold2/sparse/database.db \
  --db datasets/speaker_fold3/sparse/database.db \
  --labels fold_1 fold_2 fold_3 \
  --outdir analysis_output

python analyze_sparse.py \
    --model datasets/speaker_fold1/sparse/0 \
    --model datasets/speaker_fold2/sparse/0 \
    --model datasets/speaker_fold3/sparse/0 \
    --labels fold_1 fold_2 fold_3 \
    --outdir sparse_analysis
"""
