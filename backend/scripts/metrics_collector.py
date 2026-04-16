from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Collects training metrics into a structured format the frontend can plot.
# Runs once per training job, after the trainer has finished.
#
# Inputs:
#   --run-dir    trainer's output dir (has point_cloud/iteration_*/*.ply, test/, etc.)
#   --log-file   tee'd stdout from the training script (iteration/loss/PSNR lines)
#   --out-dir    where to write metrics.jsonl + metrics_summary.json
#   --backend    "fastergs" or "opensplat" - used for the summary's backend field
#   plus metadata: --dataset, --run-tag, --iterations, --partition, --seed
#
# Outputs:
#   <out-dir>/metrics.jsonl          one JSON object per checkpoint record
#   <out-dir>/metrics_summary.json   single JSON object with final values + metadata
#
# Design rules:
# - Never crash. If SSIM/LPIPS can't be computed (no test renders on disk, or
#   the lpips package isn't available), record them as null and keep going.
#   The frontend consumer hides null metrics gracefully.
# - Run on HPG, inside the same conda env the trainer used.


# Inria train.py + Faster-GS trainer progress lines we can parse, e.g.:
#   "Training progress:  10%|...| 1000/10000 [00:20<03:30, ...Loss=0.045, PSNR=25.32]"
# We try a couple of patterns because the exact format varies by version.
PROGRESS_PATTERNS = [
    re.compile(r"(?P<iter>\d+)\s*/\s*(?P<total>\d+).*?Loss\s*[:=]\s*(?P<loss>[0-9.eE+-]+).*?PSNR\s*[:=]\s*(?P<psnr>[0-9.eE+-]+)"),
    re.compile(r"iter\w*\s*[:=]\s*(?P<iter>\d+).*?loss\s*[:=]\s*(?P<loss>[0-9.eE+-]+).*?psnr\s*[:=]\s*(?P<psnr>[0-9.eE+-]+)", re.IGNORECASE),
]
# Evaluation summary lines the trainer sometimes prints at save iterations, e.g.:
#   "[ITER 7000] Evaluating test: PSNR 27.41 SSIM 0.887 LPIPS 0.152"
EVAL_PATTERN = re.compile(
    r"ITER\s*(?P<iter>\d+).*?PSNR\s*[:=]?\s*(?P<psnr>[0-9.eE+-]+)"
    r"(?:.*?SSIM\s*[:=]?\s*(?P<ssim>[0-9.eE+-]+))?"
    r"(?:.*?LPIPS\s*[:=]?\s*(?P<lpips>[0-9.eE+-]+))?",
    re.IGNORECASE,
)

# OpenSplat binary prints less structured output, but we try to catch any PSNR
# line and the final iteration count if present.
OPENSPLAT_PSNR_PATTERN = re.compile(r"PSNR\s*[:=]?\s*(?P<psnr>[0-9.eE+-]+)", re.IGNORECASE)
OPENSPLAT_ITER_PATTERN = re.compile(r"iter(?:ation)?s?\s*[:=]?\s*(?P<iter>\d+)", re.IGNORECASE)


def count_ply_gaussians(ply_path: Path) -> int | None:
    # Gaussian-splat .ply files have one vertex per gaussian, so vertex count
    # equals the number of gaussians. We try plyfile first (cheap and correct
    # for both ascii and binary); fall back to parsing the header manually so
    # this still works in environments without plyfile installed.
    try:
        from plyfile import PlyData
        data = PlyData.read(str(ply_path))
        return int(data.elements[0].count)
    except Exception:
        pass
    try:
        with ply_path.open("rb") as fid:
            in_header = True
            while in_header:
                line = fid.readline().decode("ascii", errors="replace").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header" or not line:
                    break
    except Exception:
        return None
    return None


def parse_progress_log(log_text: str) -> list[dict]:
    # Walk the training log line-by-line and pull out the most recent
    # iter/loss/PSNR we can see. The trainer prints a lot of progress lines;
    # we deduplicate by keeping only the latest record per unique iteration.
    records: dict[int, dict] = {}
    for line in log_text.splitlines():
        matched = False
        for pattern in PROGRESS_PATTERNS:
            m = pattern.search(line)
            if m:
                try:
                    iteration = int(m.group("iter"))
                    loss = float(m.group("loss"))
                    psnr = float(m.group("psnr"))
                except Exception:
                    continue
                records[iteration] = {
                    "iteration": iteration,
                    "loss": loss,
                    "psnr": psnr,
                }
                matched = True
                break
        if matched:
            continue
        m = EVAL_PATTERN.search(line)
        if m:
            try:
                iteration = int(m.group("iter"))
                psnr = float(m.group("psnr"))
            except Exception:
                continue
            rec = records.get(iteration, {"iteration": iteration, "loss": None, "psnr": None})
            rec["psnr"] = psnr
            if m.group("ssim") is not None:
                try:
                    rec["ssim"] = float(m.group("ssim"))
                except Exception:
                    pass
            if m.group("lpips") is not None:
                try:
                    rec["lpips"] = float(m.group("lpips"))
                except Exception:
                    pass
            records[iteration] = rec

    return [records[k] for k in sorted(records.keys())]


def parse_opensplat_log(log_text: str) -> tuple[float | None, int | None]:
    # OpenSplat's output is freeform. Best we can do is grab the last PSNR and
    # last iteration-looking number we see. Both may be missing.
    last_psnr: float | None = None
    last_iter: int | None = None
    for line in log_text.splitlines():
        m = OPENSPLAT_PSNR_PATTERN.search(line)
        if m:
            try:
                last_psnr = float(m.group("psnr"))
            except Exception:
                pass
        m = OPENSPLAT_ITER_PATTERN.search(line)
        if m:
            try:
                last_iter = int(m.group("iter"))
            except Exception:
                pass
    return last_psnr, last_iter


def compute_image_metrics(renders_dir: Path, gt_dir: Path) -> tuple[float | None, float | None]:
    # Computes average SSIM + LPIPS over matching image filenames.
    # Returns (None, None) if either dir is missing, if there are no matching
    # files, or if required packages aren't installed.
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        return None, None

    exts = {".png", ".jpg", ".jpeg"}
    renders = {p.name: p for p in renders_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    gts = {p.name: p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    common = sorted(set(renders.keys()) & set(gts.keys()))
    if not common:
        return None, None

    try:
        import numpy as np
        from PIL import Image
    except Exception:
        return None, None

    # SSIM: scikit-image is the standard. If it isn't installed, skip.
    ssim_fn = None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_fn = _ssim
    except Exception:
        pass

    # LPIPS: torch + lpips package. Loading the AlexNet weights takes a few
    # seconds the first time. If either import fails, skip LPIPS cleanly.
    lpips_fn = None
    try:
        import torch
        import lpips as _lpips  # noqa: F401
        loss_fn = _lpips.LPIPS(net="alex", verbose=False)
        if torch.cuda.is_available():
            loss_fn = loss_fn.cuda()

        def _compute_lpips(a_np, b_np):
            # lpips wants [-1, 1] tensors shaped (1, 3, H, W)
            a_t = torch.from_numpy(a_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            b_t = torch.from_numpy(b_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
            if torch.cuda.is_available():
                a_t = a_t.cuda()
                b_t = b_t.cuda()
            with torch.no_grad():
                return float(loss_fn(a_t, b_t).item())

        lpips_fn = _compute_lpips
    except Exception:
        pass

    ssim_vals: list[float] = []
    lpips_vals: list[float] = []
    for name in common:
        try:
            render = np.array(Image.open(renders[name]).convert("RGB"))
            gt = np.array(Image.open(gts[name]).convert("RGB"))
        except Exception:
            continue
        if render.shape != gt.shape:
            continue
        if ssim_fn is not None:
            try:
                s = ssim_fn(render, gt, channel_axis=2, data_range=255)
                ssim_vals.append(float(s))
            except Exception:
                pass
        if lpips_fn is not None:
            try:
                lpips_vals.append(lpips_fn(render, gt))
            except Exception:
                pass

    avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else None
    avg_lpips = sum(lpips_vals) / len(lpips_vals) if lpips_vals else None
    return avg_ssim, avg_lpips


def find_checkpoint_iterations(run_dir: Path) -> list[int]:
    # The trainer saves gaussians every few thousand iterations at paths like
    # point_cloud/iteration_7000/point_cloud.ply. We scan for those and return
    # the sorted iteration numbers.
    pc_dir = run_dir / "point_cloud"
    if not pc_dir.is_dir():
        return []
    iterations = []
    for entry in pc_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("iteration_"):
            continue
        try:
            iterations.append(int(entry.name[len("iteration_"):]))
        except ValueError:
            continue
    return sorted(iterations)


def find_eval_dirs(run_dir: Path, iteration: int) -> tuple[Path, Path]:
    # Inria fork writes test renders to test/ours_{iter}/renders and ground
    # truth to test/ours_{iter}/gt. Both may or may not exist depending on
    # whether the trainer ran eval at this iteration.
    base = run_dir / "test" / f"ours_{iteration}"
    return base / "renders", base / "gt"


def collect_fastergs(
    run_dir: Path,
    log_text: str,
    start_wall: float | None,
) -> tuple[list[dict], dict]:
    # Returns (per-checkpoint records, per-checkpoint wall-time map keyed by
    # iteration). Wall time at a checkpoint is approximated from the ply mtime
    # (good enough for a rough "how long did we take to reach iter N" chart).
    progress = parse_progress_log(log_text)

    records: list[dict] = []
    checkpoints = find_checkpoint_iterations(run_dir)
    progress_by_iter = {r["iteration"]: r for r in progress}

    # A "checkpoint" is an iteration where the trainer saved a PLY. If we have
    # no checkpoints (dry run, training crashed early) fall back to the last
    # progress line so we still emit something.
    iter_list = checkpoints if checkpoints else ([progress[-1]["iteration"]] if progress else [])

    for iteration in iter_list:
        record: dict = {
            "iteration": iteration,
            "loss": None,
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "num_gaussians": None,
            "splats_per_frame": None,
            "wall_seconds": None,
        }

        # Pull loss/PSNR from the nearest earlier progress line if there's no
        # exact match (eval runs on multiples of 1000, progress prints on
        # multiples of 10, so exact match is common but not guaranteed).
        if iteration in progress_by_iter:
            src = progress_by_iter[iteration]
        else:
            earlier = [r for r in progress if r["iteration"] <= iteration]
            src = earlier[-1] if earlier else None
        if src:
            record["loss"] = src.get("loss")
            record["psnr"] = src.get("psnr")

        # Gaussian count: count vertices in the saved PLY.
        ply_path = run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if ply_path.is_file():
            g = count_ply_gaussians(ply_path)
            record["num_gaussians"] = g
            # During training every active gaussian gets rasterized per frame,
            # so splats_per_frame ~= num_gaussians. True per-frame counts would
            # need hooking into the rasterizer, which we skip for now.
            record["splats_per_frame"] = g

            if start_wall is not None:
                try:
                    record["wall_seconds"] = max(0.0, ply_path.stat().st_mtime - start_wall)
                except Exception:
                    pass

        # SSIM/LPIPS: only if the trainer emitted eval images for this iter.
        renders_dir, gt_dir = find_eval_dirs(run_dir, iteration)
        ssim_val, lpips_val = compute_image_metrics(renders_dir, gt_dir)
        record["ssim"] = ssim_val
        record["lpips"] = lpips_val

        records.append(record)

    summary_extras = {}
    if records:
        last = records[-1]
        summary_extras = {
            "final_psnr": last.get("psnr"),
            "final_ssim": last.get("ssim"),
            "final_lpips": last.get("lpips"),
            "final_num_gaussians": last.get("num_gaussians"),
            "final_splats_per_frame": last.get("splats_per_frame"),
            "final_loss": last.get("loss"),
            "total_wall_seconds": last.get("wall_seconds"),
        }
    return records, summary_extras


def collect_opensplat(
    run_dir: Path,
    log_text: str,
    start_wall: float | None,
) -> tuple[list[dict], dict]:
    # OpenSplat doesn't emit checkpoint-level metrics we can trust, so we just
    # write a single final record with whatever we can find. SSIM/LPIPS stay
    # null unless the user wired up an eval pass separately.
    psnr, iter_count = parse_opensplat_log(log_text)

    num_gaussians = None
    ply_path = run_dir / "splat.ply"
    if ply_path.is_file():
        num_gaussians = count_ply_gaussians(ply_path)

    wall_seconds = None
    if start_wall is not None and ply_path.is_file():
        try:
            wall_seconds = max(0.0, ply_path.stat().st_mtime - start_wall)
        except Exception:
            pass

    record = {
        "iteration": iter_count,
        "loss": None,
        "psnr": psnr,
        "ssim": None,
        "lpips": None,
        "num_gaussians": num_gaussians,
        "splats_per_frame": num_gaussians,
        "wall_seconds": wall_seconds,
    }
    summary_extras = {
        "final_psnr": psnr,
        "final_ssim": None,
        "final_lpips": None,
        "final_num_gaussians": num_gaussians,
        "final_splats_per_frame": num_gaussians,
        "final_loss": None,
        "total_wall_seconds": wall_seconds,
    }
    return [record], summary_extras


def main():
    parser = argparse.ArgumentParser(description="Collect training metrics into metrics.jsonl + metrics_summary.json")
    parser.add_argument("--run-dir", required=True, help="Trainer output dir (contains point_cloud/, test/, etc.)")
    parser.add_argument("--log-file", required=True, help="Path to tee'd training stdout log")
    parser.add_argument("--out-dir", required=True, help="Where to write metrics.jsonl + metrics_summary.json")
    parser.add_argument("--backend", required=True, choices=["fastergs", "opensplat"])
    parser.add_argument("--dataset", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--partition", default="")
    parser.add_argument("--seed", default=None)
    parser.add_argument("--start-epoch", type=float, default=None,
                        help="Training start time as unix epoch seconds; used to derive wall_seconds per checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = Path(args.log_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_text = ""
    if log_path.is_file():
        try:
            log_text = log_path.read_text(errors="replace")
        except Exception:
            log_text = ""
    else:
        print(f"[warn] log file not found: {log_path}", file=sys.stderr)

    if args.backend == "fastergs":
        records, summary_extras = collect_fastergs(run_dir, log_text, args.start_epoch)
    else:
        records, summary_extras = collect_opensplat(run_dir, log_text, args.start_epoch)

    # Write jsonl (one record per line). Empty file is acceptable - the
    # frontend just shows "no series data" in that case.
    jsonl_path = out_dir / "metrics.jsonl"
    with jsonl_path.open("w") as fid:
        for rec in records:
            fid.write(json.dumps(rec) + "\n")

    summary = {
        "backend": args.backend,
        "dataset": args.dataset,
        "run_tag": args.run_tag,
        "iterations": args.iterations,
        "partition": args.partition,
        "seed": args.seed,
        "created_at_epoch": time.time(),
    }
    summary.update(summary_extras)

    with (out_dir / "metrics_summary.json").open("w") as fid:
        json.dump(summary, fid, indent=2)

    print(f"[metrics_collector] wrote {jsonl_path} ({len(records)} records)")
    print(f"[metrics_collector] wrote {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
