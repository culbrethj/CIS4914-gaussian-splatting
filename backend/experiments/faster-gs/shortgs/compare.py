"""
Aggregates Shorter-Splatting matrix runs and produces comparison plots.

Reads every ``summary.json`` under ``--results-dir``, groups them by
dataset + config, and renders three bar plots (PSNR, wall time, gaussian
count) plus a combined CSV.

Works on partial results - you can aggregate mid-matrix if a few runs are
still queued.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Reads every summary.json run_matrix.py produced under --results-dir,
# groups them by experiment_config (baseline, scale_reset, entropy, ...),
# and renders bar plots comparing final_psnr, total_wall_seconds, and
# final_num_gaussians across configurations.
#
# Each bar shows the mean across seeds for that config on that dataset;
# error bars show seed-level stdev. Datasets are plotted as grouped bars.


def load_summaries(results_dir: Path) -> list[dict]:
    out = []
    for entry in results_dir.iterdir():
        if not entry.is_dir():
            continue
        summary = entry / "summary.json"
        if not summary.is_file():
            continue
        try:
            out.append(json.loads(summary.read_text()))
        except Exception as exc:
            print(f"[compare] bad summary {summary}: {exc}", file=sys.stderr)
    return out


def group_stats(summaries: list[dict], metric: str) -> dict:
    # Returns { dataset: { config_name: (mean, stdev) } }
    bucket: dict[str, dict[str, list[float]]] = {}
    for s in summaries:
        dataset = s.get("experiment_dataset") or s.get("dataset") or "unknown"
        config = s.get("experiment_config") or "unknown"
        val = s.get(metric)
        if val is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        bucket.setdefault(dataset, {}).setdefault(config, []).append(v)

    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for dataset, configs in bucket.items():
        for config, values in configs.items():
            mean = statistics.fmean(values) if values else 0.0
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            stats.setdefault(dataset, {})[config] = (mean, stdev)
    return stats


def plot_bars(stats: dict, ylabel: str, title: str, out_path: Path, config_order: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[compare] matplotlib unavailable: {exc}", file=sys.stderr)
        return

    datasets = sorted(stats.keys())
    if not datasets:
        print(f"[compare] no data for {title}, skipping plot")
        return

    # Filter config_order to ones actually present across all datasets.
    configs_seen = set()
    for d in datasets:
        configs_seen.update(stats[d].keys())
    configs = [c for c in config_order if c in configs_seen]
    for c in sorted(configs_seen):
        if c not in configs:
            configs.append(c)

    n_configs = len(configs)
    if n_configs == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_configs * len(datasets)), 4))
    bar_width = 0.8 / n_configs
    x_indices = range(len(datasets))

    for c_idx, config in enumerate(configs):
        means = []
        errs = []
        for d in datasets:
            entry = stats[d].get(config)
            if entry is None:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(entry[0])
                errs.append(entry[1])
        positions = [x + c_idx * bar_width - (n_configs - 1) * bar_width / 2 for x in x_indices]
        ax.bar(positions, means, width=bar_width, yerr=errs, label=config, capsize=3)

    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def write_combined_table(summaries: list[dict], out_path: Path):
    # CSV-style table so you can import into a paper or sheet.
    rows = [("dataset", "config", "seed", "final_psnr", "final_ssim", "final_lpips",
             "final_num_gaussians", "total_wall_seconds")]
    for s in summaries:
        rows.append((
            s.get("experiment_dataset") or s.get("dataset") or "",
            s.get("experiment_config") or "",
            str(s.get("experiment_seed") or s.get("seed") or ""),
            str(s.get("final_psnr") or ""),
            str(s.get("final_ssim") or ""),
            str(s.get("final_lpips") or ""),
            str(s.get("final_num_gaussians") or ""),
            str(s.get("total_wall_seconds") or ""),
        ))
    with out_path.open("w") as f:
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"[compare] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Shorter-Splatting matrix results.")
    parser.add_argument("--results-dir", required=True, help="Directory containing one subdir per run with summary.json")
    parser.add_argument("--output", required=True, help="Output directory for plots and combined CSV")
    parser.add_argument("--config-order", default="baseline,scale_reset,entropy,progressive,combined",
                        help="Comma-separated config names; controls bar order left-to-right")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.is_dir():
        sys.exit(f"results-dir not found: {results_dir}")

    summaries = load_summaries(results_dir)
    if not summaries:
        sys.exit(f"no summary.json files under {results_dir}")

    config_order = [c.strip() for c in args.config_order.split(",") if c.strip()]

    plot_bars(
        group_stats(summaries, "final_psnr"),
        ylabel="PSNR (dB)", title="Final PSNR by configuration",
        out_path=output_dir / "final_psnr.png",
        config_order=config_order,
    )
    plot_bars(
        group_stats(summaries, "total_wall_seconds"),
        ylabel="Wall time (seconds)", title="Total wall time by configuration",
        out_path=output_dir / "wall_time.png",
        config_order=config_order,
    )
    plot_bars(
        group_stats(summaries, "final_num_gaussians"),
        ylabel="Number of gaussians", title="Final gaussian count by configuration",
        out_path=output_dir / "final_num_gaussians.png",
        config_order=config_order,
    )

    write_combined_table(summaries, output_dir / "combined.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[compare] error: {exc}", file=sys.stderr)
        sys.exit(1)
