#!/usr/bin/env python3
"""
Analyze COLMAP sparse reconstructions across multiple folds using pycolmap.

This script:
- loads each sparse model with pycolmap.Reconstruction(model_dir)
- extracts per-point statistics from reconstruction.points3D
- computes useful per-fold summary statistics
- writes CSVs
- generates charts

Expected model layout examples:
    fold_1/sparse/0/
    fold_2/sparse/0/
    fold_3/sparse/0/

Usage:
    python analyze_sparse.py \
        --model datasets/speaker_fold1/sparse/0 \
        --model datasets/speaker_fold2/sparse/0 \
        --model datasets/speaker_fold3/sparse/0 \
        --labels fold_1 fold_2 fold_3 \
        --outdir sparse_analysis

Install:
    pip install pycolmap pandas matplotlib numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze COLMAP sparse reconstructions with pycolmap."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Path to sparse model directory. Pass once per fold.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching the order of --model arguments.",
    )
    parser.add_argument(
        "--outdir",
        default="sparse_analysis",
        help="Directory where CSVs and charts will be saved.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.labels is not None and len(args.labels) != len(args.model):
        raise ValueError("Number of labels must match number of model paths.")

    for model_dir in args.model:
        p = Path(model_dir)
        if not p.exists():
            raise FileNotFoundError(f"Model directory not found: {p}")


def load_reconstruction(model_dir: str) -> pycolmap.Reconstruction:
    # pycolmap.Reconstruction(path) reads a COLMAP-format reconstruction.
    rec = pycolmap.Reconstruction(model_dir)
    # if not rec.is_valid():
    #     raise ValueError(f"Invalid reconstruction: {model_dir}")
    return rec


def point_track_length(point3d: Any) -> int:
    """
    Tries a few common pycolmap Track interfaces safely.
    """
    track = point3d.track

    # Most likely options first
    if hasattr(track, "length"):
        attr = track.length
        return int(attr() if callable(attr) else attr)

    if hasattr(track, "__len__"):
        return int(len(track))

    if hasattr(track, "elements"):
        elems = track.elements
        return int(len(elems() if callable(elems) else elems))

    raise AttributeError("Could not determine track length from pycolmap Point3D.track")


def iter_points3d(rec: pycolmap.Reconstruction):
    """
    Yields (point3D_id, point3D) robustly whether rec.points3D behaves like a dict
    or another mapping-like object.
    """
    pts = rec.points3D

    if hasattr(pts, "items"):
        yield from pts.items()
        return

    # Fallback through point IDs
    for pid in rec.point3D_ids():
        yield pid, rec.point3D(pid)


def extract_per_point_df(rec: pycolmap.Reconstruction, fold_label: str, model_dir: str) -> pd.DataFrame:
    rows = []

    for point_id, point in iter_points3d(rec):
        xyz = np.asarray(point.xyz, dtype=float).reshape(-1)
        rgb = np.asarray(point.color if hasattr(point, "color") else point.rgb, dtype=float).reshape(-1)

        rows.append(
            {
                "fold": fold_label,
                "model_dir": str(Path(model_dir).resolve()),
                "point3D_id": int(point_id),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "r": int(rgb[0]),
                "g": int(rgb[1]),
                "b": int(rgb[2]),
                "error": float(point.error),
                "track_length": point_track_length(point),
            }
        )

    return pd.DataFrame(rows)


def summarize_fold(rec: pycolmap.Reconstruction, df: pd.DataFrame, fold_label: str) -> dict[str, Any]:
    total_points = int(rec.num_points3D())
    reg_images = int(rec.num_reg_images())

    xyz = df[["x", "y", "z"]] if not df.empty else pd.DataFrame(columns=["x", "y", "z"])

    return {
        "fold": fold_label,
        "registered_images": reg_images,
        "triangulated_points": total_points,
        "points_per_registered_image": (
            float(total_points / reg_images) if reg_images > 0 else np.nan
        ),
        "mean_track_length_builtin": float(rec.compute_mean_track_length()) if total_points > 0 else 0.0,
        "mean_reproj_error_builtin": float(rec.compute_mean_reprojection_error()) if total_points > 0 else 0.0,
        "median_track_length": float(df["track_length"].median()) if total_points > 0 else 0.0,
        "std_track_length": float(df["track_length"].std(ddof=1)) if total_points > 1 else 0.0,
        "min_track_length": int(df["track_length"].min()) if total_points > 0 else 0,
        "max_track_length": int(df["track_length"].max()) if total_points > 0 else 0,
        "median_reproj_error": float(df["error"].median()) if total_points > 0 else 0.0,
        "std_reproj_error": float(df["error"].std(ddof=1)) if total_points > 1 else 0.0,
        "min_reproj_error": float(df["error"].min()) if total_points > 0 else 0.0,
        "max_reproj_error": float(df["error"].max()) if total_points > 0 else 0.0,
        "pct_track_ge_3": float((df["track_length"] >= 3).mean() * 100) if total_points > 0 else 0.0,
        "pct_track_ge_4": float((df["track_length"] >= 4).mean() * 100) if total_points > 0 else 0.0,
        "pct_track_ge_5": float((df["track_length"] >= 5).mean() * 100) if total_points > 0 else 0.0,
        "x_min": float(xyz["x"].min()) if total_points > 0 else np.nan,
        "x_max": float(xyz["x"].max()) if total_points > 0 else np.nan,
        "y_min": float(xyz["y"].min()) if total_points > 0 else np.nan,
        "y_max": float(xyz["y"].max()) if total_points > 0 else np.nan,
        "z_min": float(xyz["z"].min()) if total_points > 0 else np.nan,
        "z_max": float(xyz["z"].max()) if total_points > 0 else np.nan,
    }


def save_csvs(all_points_df: pd.DataFrame, summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    all_points_df.to_csv(outdir / "per_point_stats.csv", index=False)
    summary_df.to_csv(outdir / "fold_sparse_summary.csv", index=False)


def plot_total_points(summary_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["fold"], summary_df["triangulated_points"])
    plt.xlabel("Fold")
    plt.ylabel("Triangulated 3D points")
    plt.title("Triangulated sparse points by fold")
    plt.tight_layout()
    plt.savefig(outdir / "triangulated_points_by_fold.png", dpi=200)
    plt.close()


def plot_points_per_registered_image(summary_df: pd.DataFrame, outdir: Path) -> None:
    df = summary_df.dropna(subset=["points_per_registered_image"])
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(df["fold"], df["points_per_registered_image"])
    plt.xlabel("Fold")
    plt.ylabel("Points per registered image")
    plt.title("Triangulated points per registered image")
    plt.tight_layout()
    plt.savefig(outdir / "points_per_registered_image.png", dpi=200)
    plt.close()


def plot_track_length_boxplot(all_points_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    order = list(all_points_df["fold"].drop_duplicates())
    data = [all_points_df.loc[all_points_df["fold"] == fold, "track_length"] for fold in order]
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.xlabel("Fold")
    plt.ylabel("Track length")
    plt.title("Track length by fold")
    plt.tight_layout()
    plt.savefig(outdir / "track_length_boxplot.png", dpi=200)
    plt.close()


def plot_track_length_histogram(all_points_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 6))
    max_track = max(2, int(all_points_df["track_length"].max()))
    bins = np.arange(1, max_track + 2) - 0.5

    for fold_name, group in all_points_df.groupby("fold"):
        plt.hist(group["track_length"], bins=bins, alpha=0.45, label=fold_name)

    plt.xlabel("Track length")
    plt.ylabel("Number of 3D points")
    plt.title("Track length histogram by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "track_length_histogram.png", dpi=200)
    plt.close()


def plot_reproj_error_histogram(all_points_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for fold_name, group in all_points_df.groupby("fold"):
        plt.hist(group["error"], bins=40, alpha=0.45, label=fold_name)

    plt.xlabel("Reprojection error")
    plt.ylabel("Number of 3D points")
    plt.title("Reprojection error histogram by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "reprojection_error_histogram.png", dpi=200)
    plt.close()


def plot_reproj_error_boxplot(all_points_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(9, 6))
    order = list(all_points_df["fold"].drop_duplicates())
    data = [all_points_df.loc[all_points_df["fold"] == fold, "error"] for fold in order]
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.xlabel("Fold")
    plt.ylabel("Reprojection error")
    plt.title("Reprojection error by fold")
    plt.tight_layout()
    plt.savefig(outdir / "reprojection_error_boxplot.png", dpi=200)
    plt.close()


def plot_reproj_error_ecdf(all_points_df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(10, 6))
    for fold_name, group in all_points_df.groupby("fold"):
        x = np.sort(group["error"].to_numpy())
        y = np.arange(1, len(x) + 1) / len(x)
        plt.plot(x, y, label=fold_name)

    plt.xlabel("Reprojection error")
    plt.ylabel("Empirical cumulative probability")
    plt.title("ECDF of reprojection error by fold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "reprojection_error_ecdf.png", dpi=200)
    plt.close()


def plot_track_support_thresholds(all_points_df: pd.DataFrame, outdir: Path) -> None:
    thresholds = [3, 4, 5]
    rows = []

    for fold_name, group in all_points_df.groupby("fold"):
        for t in thresholds:
            rows.append(
                {
                    "fold": fold_name,
                    "threshold": f">={t}",
                    "percent": float((group["track_length"] >= t).mean() * 100),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "track_support_thresholds.csv", index=False)

    pivot = df.pivot(index="threshold", columns="fold", values="percent")
    ax = pivot.plot(kind="bar", figsize=(9, 6))
    ax.set_xlabel("Track length threshold")
    ax.set_ylabel("Percent of 3D points")
    ax.set_title("Percent of 3D points with stronger multi-view support")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(outdir / "track_support_thresholds.png", dpi=200)
    plt.close()


def print_summary(summary_df: pd.DataFrame) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\nSparse reconstruction summary:\n")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> int:
    args = parse_args()
    validate_args(args)

    labels = args.labels if args.labels is not None else [
        f"fold_{i+1}" for i in range(len(args.model))
    ]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_fold_points = []
    summaries = []

    for model_dir, label in zip(args.model, labels):
        rec = load_reconstruction(model_dir)
        point_df = extract_per_point_df(rec, label, model_dir)
        all_fold_points.append(point_df)
        summaries.append(summarize_fold(rec, point_df, label))

    all_points_df = pd.concat(all_fold_points, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    save_csvs(all_points_df, summary_df, outdir)

    plot_total_points(summary_df, outdir)
    plot_points_per_registered_image(summary_df, outdir)
    plot_track_length_boxplot(all_points_df, outdir)
    plot_track_length_histogram(all_points_df, outdir)
    plot_reproj_error_histogram(all_points_df, outdir)
    plot_reproj_error_boxplot(all_points_df, outdir)
    plot_reproj_error_ecdf(all_points_df, outdir)
    plot_track_support_thresholds(all_points_df, outdir)

    print_summary(summary_df)
    print(f"\nSaved outputs to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())