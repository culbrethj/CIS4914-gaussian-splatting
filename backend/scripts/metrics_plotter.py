"""
Renders matplotlib PNGs from the metrics jsonl + summary produced by
``metrics_collector.py``. Produces one line chart per captured metric
(PSNR, SSIM, LPIPS, loss, #gaussians, splats/frame, wall time), plus a
PSNR-vs-wall-time chart and histograms for gaussian scale + opacity.

Uses the ``Agg`` backend so it runs fine on HPG compute nodes without a
display. If matplotlib isn't installed, each plot call logs a warning and
no-ops - the frontend's interactive charts (Recharts) still work from the
same jsonl, so missing PNGs just mean the archivable snapshot is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reads metrics.jsonl + metrics_summary.json and writes PNGs next to them.
# These are snapshot images you can drop into a report or email; the
# frontend Reports page renders its own interactive versions from the same
# data. We still write PNGs here so there's something to archive even if
# the frontend goes away.


# Line plots: one PNG per metric, x-axis = iteration (default) or one of
# the other fields listed below (y label, alternative x field, alt x label).
LINE_PLOTS = [
    # (metric_key, y_label, x_key, x_label, title_suffix)
    ("psnr", "PSNR (dB)", "iteration", "Iteration", ""),
    ("ssim", "SSIM", "iteration", "Iteration", ""),
    ("lpips", "LPIPS", "iteration", "Iteration", ""),
    ("loss", "Loss", "iteration", "Iteration", ""),
    ("num_gaussians", "Number of Gaussians", "iteration", "Iteration", ""),
    ("splats_per_frame", "Splats / Frame", "iteration", "Iteration", ""),
    ("wall_seconds", "Wall Time (s)", "iteration", "Iteration", ""),
    # Shorter-Splatting validation: PSNR vs. wall time shows the actual
    # speedup from the paper's techniques (x axis is seconds, not iters).
    ("psnr", "PSNR (dB)", "wall_seconds", "Wall Time (s)", " vs time"),
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


def _placeholder(out_path: Path, text: str):
    # Write a tiny "no data" PNG so the frontend never 404s when it asks
    # for a plot that we had no data to render.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_line(records: list[dict], *, y_key: str, y_label: str,
              x_key: str, x_label: str, out_path: Path, title: str):
    # Generic line plot. Accepts any two fields from the record - we use
    # this for both "metric vs iteration" and "PSNR vs wall time".
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[metrics_plotter] matplotlib not available, skipping {y_key}: {exc}", file=sys.stderr)
        return

    xs: list[float] = []
    ys: list[float] = []
    for rec in records:
        x = rec.get(x_key)
        y = rec.get(y_key)
        if x is None or y is None:
            continue
        try:
            xs.append(float(x))
            ys.append(float(y))
        except Exception:
            continue

    if not xs:
        _placeholder(out_path, f"no {y_key} vs {x_key} data")
        return

    # Sort by x so the line renders in order even if records arrive
    # out-of-order in the jsonl.
    paired = sorted(zip(xs, ys), key=lambda p: p[0])
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # Start x axis at 0 so the reader sees the full training range, not
    # just the slice where we happen to have data points.
    if xs and xs[0] > 0:
        ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_histogram(hist: dict | None, *, out_path: Path, title: str, x_label: str):
    # Renders a precomputed histogram (bin edges + counts). We don't
    # recompute the histogram here; metrics_collector already did that from
    # the saved PLY and wrote it into metrics_summary.json.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not hist or not hist.get("bins") or not hist.get("counts"):
        _placeholder(out_path, "no histogram data")
        return

    bins = hist["bins"]
    counts = hist["counts"]
    if len(bins) != len(counts) + 1:
        _placeholder(out_path, "malformed histogram")
        return

    # Center each bar between its bin edges; width is the bin span.
    centers = [(bins[i] + bins[i + 1]) / 2.0 for i in range(len(counts))]
    widths = [bins[i + 1] - bins[i] for i in range(len(counts))]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(centers, counts, width=widths, align="center",
           edgecolor="#0048a4", color="#0f6bd8", alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render PNGs from metrics.jsonl + metrics_summary.json.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing metrics.jsonl + summary")
    parser.add_argument("--title-prefix", default="", help="Optional prefix prepended to every plot title")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    jsonl_path = metrics_dir / "metrics.jsonl"
    summary_path = metrics_dir / "metrics_summary.json"
    if not jsonl_path.is_file():
        print(f"[metrics_plotter] no metrics.jsonl at {jsonl_path}, nothing to plot", file=sys.stderr)
        return

    records = read_jsonl(jsonl_path)
    prefix = args.title_prefix.rstrip() + (" - " if args.title_prefix.strip() else "")

    # Line plots (one per iteration-indexed metric, plus PSNR-vs-time).
    for y_key, y_label, x_key, x_label, title_suffix in LINE_PLOTS:
        out_name = f"{y_key}.png" if x_key == "iteration" else f"{y_key}_vs_{x_key}.png"
        out_path = metrics_dir / out_name
        title = f"{prefix}{y_label}{title_suffix}"
        plot_line(records, y_key=y_key, y_label=y_label, x_key=x_key, x_label=x_label,
                  out_path=out_path, title=title)
        print(f"[metrics_plotter] wrote {out_path}")

    # Histograms (from summary, not jsonl). Used to validate the
    # Shorter-Splatting paper's claims about gaussian scale + opacity shifts.
    hists = {}
    if summary_path.is_file():
        try:
            hists = (json.loads(summary_path.read_text()).get("histograms") or {})
        except Exception:
            hists = {}

    plot_histogram(
        hists.get("scale"),
        out_path=metrics_dir / "scale_hist.png",
        title=f"{prefix}Scale Distribution",
        x_label="Mean gaussian scale (exp of mean log-scale)",
    )
    print(f"[metrics_plotter] wrote {metrics_dir / 'scale_hist.png'}")

    plot_histogram(
        hists.get("opacity"),
        out_path=metrics_dir / "opacity_hist.png",
        title=f"{prefix}Opacity Distribution",
        x_label="Sigmoid(opacity) in [0, 1]",
    )
    print(f"[metrics_plotter] wrote {metrics_dir / 'opacity_hist.png'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[metrics_plotter] error: {exc}", file=sys.stderr)
        sys.exit(1)
