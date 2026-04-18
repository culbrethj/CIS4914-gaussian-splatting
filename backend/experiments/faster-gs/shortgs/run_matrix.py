"""
Shorter-Splatting paper experiment matrix runner.

Reads a YAML matrix (``datasets * seeds * configs``) and kicks off
``fastergs_pipeline.py`` once per combination. Each combo's
``metrics_summary.json`` is copied into ``--results-dir/<run_name>/summary.json``
so ``compare.py`` can aggregate without scanning every dataset dir.

Config YAML shape (see ``experiment_matrix.yaml``)::

  datasets: [can, garden]
  seeds:    [0, 1, 2]
  base_iterations: 10000
  partition: hpg-turin
  configs:
    - name: stock_baseline
      flags: {}
    - name: fastergs_adam           # stock rasterizer + FasterGS fused Adam
      flags: {use_fastergs_adam: true}
    - name: scale_reset             # shortgs paper technique
      flags: {shortgs_scale_reset_every: 1000, shortgs_scale_reset_factor: 0.9}
    - name: opensplat               # separate backend; shortgs flags ignored
      backend: opensplat
      flags: {}

Serial execution; each pipeline call blocks until HPG completes. Use
``--dry-run`` to print the invocation for every combo without running.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Reads experiment_matrix.yaml and runs fastergs_pipeline.py once per
# (dataset x config x seed) combination. After each run, copies
# metrics_summary.json into results_dir/<run_name>/summary.json so
# compare.py can aggregate them without scanning every dataset dir.
#
# The fastergs pipeline still publishes its normal outputs to
# backend/datasets/<dataset>/...; this runner is just a bookkeeping layer.


def parse_matrix(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


# --- Run execution --------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent


def build_env(config_flags: dict, seed: int) -> dict:
    # Map config.flags from the yaml (snake_case) into the SHORTGS_* env
    # vars fastergs_pipeline.py understands.
    env = os.environ.copy()
    mapping = {
        "shortgs_scale_reset_every": "SHORTGS_SCALE_RESET_EVERY",
        "shortgs_scale_reset_factor": "SHORTGS_SCALE_RESET_FACTOR",
        "shortgs_entropy_weight": "SHORTGS_ENTROPY_WEIGHT",
        "shortgs_progressive_resolution": "SHORTGS_PROGRESSIVE_RESOLUTION",
    }
    for k, v in config_flags.items():
        if k in mapping and v is not None:
            env[mapping[k]] = str(v)
    env["SHORTGS_SEED"] = str(seed)
    return env


def invoke_pipeline(*, dataset: str, video: Path, iterations: int, partition: str,
                    backend: str, config_flags: dict, seed: int, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(BACKEND_DIR / "scripts" / "fastergs_pipeline.py"),
        dataset,
        "--video", str(video),
        "--iters", str(iterations),
        "--train-partition", partition,
        "--seed", str(seed),
        "--backend", backend,
    ]
    # shortgs + fastergs-adam flags only apply to the fastergs backend.
    # OpenSplat is a separate C++ trainer that doesn't know about them, so
    # drop them here rather than errorring — lets the same yaml row carry
    # flags for mixed backends without surprising the user.
    if backend == "fastergs":
        if config_flags.get("shortgs_scale_reset_every"):
            cmd += ["--shortgs-scale-reset-every", str(config_flags["shortgs_scale_reset_every"])]
        if config_flags.get("shortgs_scale_reset_factor") is not None:
            cmd += ["--shortgs-scale-reset-factor", str(config_flags["shortgs_scale_reset_factor"])]
        if config_flags.get("shortgs_entropy_weight"):
            cmd += ["--shortgs-entropy-weight", str(config_flags["shortgs_entropy_weight"])]
        if config_flags.get("shortgs_progressive_resolution"):
            cmd += ["--shortgs-progressive-resolution", str(config_flags["shortgs_progressive_resolution"])]
        if config_flags.get("use_fastergs_adam"):
            cmd += ["--use-fastergs-adam"]

    env = build_env(config_flags, seed)
    print(f"[run_matrix] $ {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, env=env, cwd=str(BACKEND_DIR))


def latest_run_summary_for(dataset: str) -> Path | None:
    # Finds the most recently written metrics_summary.json under
    # backend/datasets/<dataset>/metrics/<run_tag>/. This is what we just
    # produced if the pipeline succeeded.
    root = BACKEND_DIR / "datasets" / dataset / "metrics"
    if not root.is_dir():
        return None
    candidates = []
    for run_dir in root.iterdir():
        summary = run_dir / "metrics_summary.json"
        if summary.is_file():
            candidates.append(summary)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Run the Shorter-Splatting experiment matrix.")
    parser.add_argument("--matrix", required=True, help="Path to experiment_matrix.yaml")
    parser.add_argument("--results-dir", required=True, help="Where to copy metrics_summary.json files")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing <dataset>.mp4 videos for each dataset in the matrix. "
                             "Falls back to the dataset's existing video/ dir if omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without invoking anything")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip dataset+config+seed combos that already have a summary under results-dir")
    args = parser.parse_args()

    matrix_path = Path(args.matrix).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    matrix = parse_matrix(matrix_path)
    datasets = matrix.get("datasets") or []
    seeds = matrix.get("seeds") or [0]
    iterations = matrix.get("base_iterations") or 10000
    partition = matrix.get("partition") or "hpg-turin"
    configs = matrix.get("configs") or []
    if not datasets:
        sys.exit("matrix has no datasets")
    if not configs:
        sys.exit("matrix has no configs")

    total = len(datasets) * len(seeds) * len(configs)
    print(f"[run_matrix] queued {total} runs: "
          f"{len(datasets)} datasets x {len(seeds)} seeds x {len(configs)} configs")

    run_idx = 0
    for dataset in datasets:
        for config in configs:
            cname = config.get("name", "unnamed")
            cflags = config.get("flags") or {}
            # Per-variant backend selector. Defaults to fastergs so every
            # pre-existing variant in the yaml keeps its current behavior.
            cbackend = config.get("backend") or "fastergs"
            if cbackend not in ("fastergs", "opensplat"):
                print(f"[run_matrix] unknown backend {cbackend!r} on config {cname}, skipping")
                continue
            for seed in seeds:
                run_idx += 1
                run_name = f"{dataset}__{cname}__seed{seed}"
                run_results_dir = results_dir / run_name
                if args.skip_existing and (run_results_dir / "summary.json").is_file():
                    print(f"[run_matrix] ({run_idx}/{total}) skip (existing): {run_name}")
                    continue

                print(f"[run_matrix] ({run_idx}/{total}) starting: {run_name}")

                # Pick the video: <video-dir>/<dataset>.mp4 if provided, else
                # the newest file under backend/datasets/<dataset>/video/.
                video_path: Path | None = None
                if args.video_dir:
                    vd = Path(args.video_dir).expanduser().resolve()
                    for ext in (".mp4", ".mov", ".mkv", ".webm"):
                        candidate = vd / f"{dataset}{ext}"
                        if candidate.is_file():
                            video_path = candidate
                            break
                if video_path is None:
                    vid_root = BACKEND_DIR / "datasets" / dataset / "video"
                    if vid_root.is_dir():
                        vids = sorted(vid_root.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                        video_path = next((v for v in vids if v.is_file()), None)
                if video_path is None:
                    print(f"[run_matrix] no video found for dataset {dataset}, skipping")
                    continue

                rc = invoke_pipeline(
                    dataset=dataset,
                    video=video_path,
                    iterations=iterations,
                    partition=partition,
                    backend=cbackend,
                    config_flags=cflags,
                    seed=seed,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    print(f"[run_matrix] pipeline exit code {rc} for {run_name}")
                    continue

                summary_src = latest_run_summary_for(dataset)
                if summary_src is None:
                    print(f"[run_matrix] no metrics summary found after {run_name}")
                    continue

                run_results_dir.mkdir(parents=True, exist_ok=True)
                dst = run_results_dir / "summary.json"
                try:
                    shutil.copy2(summary_src, dst)
                    # Also annotate with the config name so compare.py doesn't
                    # have to reverse-engineer it from the shortgs fields.
                    with dst.open() as f:
                        d = json.load(f)
                    d["experiment_config"] = cname
                    d["experiment_dataset"] = dataset
                    d["experiment_seed"] = seed
                    with dst.open("w") as f:
                        json.dump(d, f, indent=2)
                except Exception as exc:
                    print(f"[run_matrix] failed to copy summary: {exc}")

    print("[run_matrix] done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[run_matrix] interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"[run_matrix] error: {exc}", file=sys.stderr)
        sys.exit(1)
