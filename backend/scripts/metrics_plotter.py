from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reads metrics.jsonl and writes a set of PNGs next to it. These PNGs are
# "paper-ready" snapshots: they don't move once generated, so you can drop
# them into a report or email.
# The interactive charts in the frontend come from the same jsonl; we render
# PNGs here just so there's something to archive.


PLOTS = [
    ("psnr", "PSNR (dB)"),
    ("ssim", "SSIM"),
    ("lpips", "LPIPS"),
    ("loss", "Loss"),
    ("num_gaussians", "Number of Gaussians"),
    ("splats_per_frame", "Splats / Frame"),
    ("wall_seconds", "Wall Time (s)"),
]


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open() as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def plot_metric(records: list[dict], metric: str, ylabel: str, out_path: Path, title_prefix: str):
    # matplotlib is an optional dep (added to requirements). If it's missing
    # we still want the pipeline to succeed - print and skip.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless so this runs on HPG compute nodes
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[metrics_plotter] matplotlib not available, skipping {metric}: {exc}", file=sys.stderr)
        return

    xs: list[int] = []
    ys: list[float] = []
    for rec in records:
        x = rec.get("iteration")
        y = rec.get(metric)
        if x is None or y is None:
            continue
        try:
            xs.append(int(x))
            ys.append(float(y))
        except Exception:
            continue

    if not xs:
        # Nothing to plot. Write a tiny placeholder so the frontend can still
        # fetch a valid PNG rather than 404ing.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.text(0.5, 0.5, f"no {metric} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}{ylabel}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render PNGs from metrics.jsonl.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing metrics.jsonl")
    parser.add_argument("--title-prefix", default="", help="Optional prefix to prepend to every plot title")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    jsonl_path = metrics_dir / "metrics.jsonl"
    if not jsonl_path.is_file():
        print(f"[metrics_plotter] no metrics.jsonl at {jsonl_path}, nothing to plot", file=sys.stderr)
        return

    records = read_jsonl(jsonl_path)
    prefix = args.title_prefix.rstrip() + (" - " if args.title_prefix.strip() else "")
    for metric, ylabel in PLOTS:
        out_path = metrics_dir / f"{metric}.png"
        plot_metric(records, metric, ylabel, out_path, prefix)
        print(f"[metrics_plotter] wrote {out_path}")


if __name__ == "__main__":
    main()
