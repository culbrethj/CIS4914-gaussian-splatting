"""
Post-training metrics collector for both Faster-GS and OpenSplat runs.

Parses the trainer's stdout log for iteration/loss/PSNR lines, counts
gaussians from each saved PLY, optionally computes SSIM + LPIPS on eval
renders if they exist on disk, and writes two files into an output dir:

  metrics.jsonl         one JSON record per iteration / checkpoint
  metrics_summary.json  dataset + shortgs metadata + final values +
                        scale/opacity histograms from the last PLY

Designed to never crash the training pipeline: when scikit-image, lpips,
or plyfile aren't installed the relevant fields just come back null.
"""

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


# The Inria trainer prints two kinds of lines we care about:
#
#   1) tqdm progress lines with iter + loss, e.g.:
#      "Training progress:   1%| | 100/10000 [00:03<05:04, 32.51it/s, Loss=0.1450, Depth Loss=0.0000]"
#      These arrive every few iterations and give us a dense loss curve.
#
#   2) Eval lines at checkpoint iterations, e.g.:
#      "[ITER 7000] Evaluating train: L1 0.017 PSNR 32.26 ..."
#      These are sparse (default is iters 7000 and 30000) and carry PSNR +
#      optionally SSIM/LPIPS if the trainer was configured to compute them.
#
# We parse both. Progress lines set loss/psnr where present, eval lines
# overwrite the same iteration's record with the authoritative PSNR + any
# SSIM/LPIPS values. The final jsonl is a merged view over every iteration
# that either kind of line mentions.
PROGRESS_PATTERNS = [
    re.compile(r"(?P<iter>\d+)\s*/\s*(?P<total>\d+).*?Loss\s*[:=]\s*(?P<loss>[0-9.eE+-]+)"),
    re.compile(r"iter\w*\s*[:=]\s*(?P<iter>\d+).*?loss\s*[:=]\s*(?P<loss>[0-9.eE+-]+)", re.IGNORECASE),
]
EVAL_PATTERN = re.compile(
    r"ITER\s*(?P<iter>\d+).*?PSNR\s*[:=]?\s*(?P<psnr>[0-9.eE+-]+)"
    r"(?:.*?SSIM\s*[:=]?\s*(?P<ssim>[0-9.eE+-]+))?"
    r"(?:.*?LPIPS\s*[:=]?\s*(?P<lpips>[0-9.eE+-]+))?",
    re.IGNORECASE,
)

# We throttle progress records: keep one record per this many iterations.
# Trainers print progress every 10 iters; over a 10k run that's 1000 points.
# Downsampling to every 100 iters keeps the chart readable without losing
# the shape of the loss curve.
PROGRESS_STRIDE = 100

# OpenSplat (pierotofy/OpenSplat v1.1.x) prints one "Step N: <loss> (pct%)"
# line per iteration during training. Example: "Step 1500: 0.0406808 (75%)".
# We parse those into a dense loss curve (downsampled on the same stride we
# use for fastergs progress lines). PSNR is not emitted during training, so
# we leave that null unless a separate eval pass writes something parsable.
OPENSPLAT_STEP_PATTERN = re.compile(
    r"Step\s+(?P<iter>\d+)\s*:\s*(?P<loss>[0-9.eE+-]+)"
)
OPENSPLAT_PSNR_PATTERN = re.compile(r"PSNR\s*[:=]?\s*(?P<psnr>[0-9.eE+-]+)", re.IGNORECASE)


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
    # tqdm writes progress with carriage returns, not newlines, so a bash
    # redirect glues thousands of progress updates onto a single line. We
    # split on BOTH \r and \n to separate them.
    segments = re.split(r"[\r\n]+", log_text)

    records: dict[int, dict] = {}

    for line in segments:
        # Progress line: iter/total ... Loss=X
        for pattern in PROGRESS_PATTERNS:
            m = pattern.search(line)
            if m:
                try:
                    iteration = int(m.group("iter"))
                    loss = float(m.group("loss"))
                except Exception:
                    break
                # downsample: only keep iterations on a stride, plus iter 0/last
                if iteration == 0 or iteration % PROGRESS_STRIDE == 0:
                    rec = records.get(iteration) or {"iteration": iteration}
                    rec["loss"] = loss
                    records[iteration] = rec
                break

        # Eval line: [ITER N] ... PSNR X (optionally SSIM Y LPIPS Z)
        m = EVAL_PATTERN.search(line)
        if m:
            try:
                iteration = int(m.group("iter"))
                psnr = float(m.group("psnr"))
            except Exception:
                continue
            rec = records.get(iteration) or {"iteration": iteration}
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

    # Ensure every record has all fields so downstream code can assume keys
    # exist (values may be None).
    for iteration, rec in records.items():
        rec.setdefault("loss", None)
        rec.setdefault("psnr", None)
        rec.setdefault("ssim", None)
        rec.setdefault("lpips", None)

    return [records[k] for k in sorted(records.keys())]


def parse_opensplat_log(log_text: str) -> tuple[list[dict], float | None, int | None]:
    # Parse "Step N: loss (pct%)" lines into an iteration-indexed record list
    # matching the fastergs schema. We downsample to PROGRESS_STRIDE so a
    # 10k-iter run produces ~100 datapoints. PSNR is rarely present during
    # training; we still scan for it so a post-hoc eval pass would surface.
    records: dict[int, dict] = {}
    last_psnr: float | None = None
    last_iter: int | None = None

    # OpenSplat uses newlines (not carriage returns) between steps, so we
    # don't need the \r split that the fastergs tqdm parser does.
    for line in log_text.splitlines():
        m = OPENSPLAT_STEP_PATTERN.search(line)
        if m:
            try:
                iteration = int(m.group("iter"))
                loss = float(m.group("loss"))
            except Exception:
                continue
            last_iter = iteration
            if iteration == 0 or iteration % PROGRESS_STRIDE == 0:
                rec = records.get(iteration) or {"iteration": iteration}
                rec["loss"] = loss
                records[iteration] = rec
            continue

        pm = OPENSPLAT_PSNR_PATTERN.search(line)
        if pm:
            try:
                last_psnr = float(pm.group("psnr"))
            except Exception:
                pass

    # Always keep the final iteration as a datapoint even when it doesn't
    # land on the stride boundary. Matches the fastergs behavior.
    if last_iter is not None and last_iter not in records:
        # Find the last loss we saw for any nearby iteration; fall back to
        # scanning the tail of the log for the literal final step.
        for line in reversed(log_text.splitlines()):
            m = OPENSPLAT_STEP_PATTERN.search(line)
            if m and int(m.group("iter")) == last_iter:
                try:
                    records[last_iter] = {"iteration": last_iter, "loss": float(m.group("loss"))}
                except Exception:
                    pass
                break

    for rec in records.values():
        rec.setdefault("loss", None)

    return [records[k] for k in sorted(records.keys())], last_psnr, last_iter


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


def compute_ply_histograms(ply_path: Path, bins: int = 32) -> dict | None:
    # Read scale_0/1/2 and opacity from the final PLY and build histograms
    # for the Reports page. This is how we validate the Shorter-Splatting
    # paper's claims:
    # - Scale reset should shift the scale distribution towards smaller
    #   values, so the histogram should skew left vs. baseline.
    # - Entropy constraint should push opacities towards 0 or 1 (polarized),
    #   so the histogram should have spikes at the ends and a sparse middle.
    # Returns a dict with bin edges + counts, or None if plyfile isn't
    # available (histograms just don't show up in that case).
    try:
        from plyfile import PlyData
        data = PlyData.read(str(ply_path))
    except Exception:
        return None
    try:
        import numpy as np
    except Exception:
        return None

    verts = data.elements[0]
    prop_names = {p.name for p in verts.properties}

    # Scale: PLY stores scale_0/1/2 in log-space. Paper reasons about actual
    # scale, so exp() the mean log-scale per gaussian.
    # Some aggressive shortgs regimes produce NaN scales for a fraction of
    # gaussians; we drop those before histogramming so the chart doesn't
    # just come back empty.
    scale_hist = None
    if {"scale_0", "scale_1", "scale_2"}.issubset(prop_names):
        try:
            s0 = np.asarray(verts["scale_0"])
            s1 = np.asarray(verts["scale_1"])
            s2 = np.asarray(verts["scale_2"])
            mean_log_scale = (s0 + s1 + s2) / 3.0
            actual_scale = np.exp(mean_log_scale)
            finite = actual_scale[np.isfinite(actual_scale)]
            nan_count = int(len(actual_scale) - len(finite))
            if len(finite) > 0:
                # Clip the top tail so the histogram isn't dominated by outliers
                cap = float(np.percentile(finite, 99.5))
                clipped = np.clip(finite, 0.0, cap if cap > 0 else 1.0)
                counts, edges = np.histogram(clipped, bins=bins)
                scale_hist = {
                    "bins": edges.tolist(),
                    "counts": counts.tolist(),
                    "unit": "mean_scale_exp",
                    "clipped_at_percentile": 99.5,
                    "nan_count": nan_count,
                }
        except Exception:
            pass

    # Opacity: PLY stores raw logits. Apply sigmoid to get [0, 1] values,
    # which is what the entropy constraint polarizes. Drop non-finite values
    # the same way as for scales.
    opacity_hist = None
    if "opacity" in prop_names:
        try:
            raw = np.asarray(verts["opacity"])
            finite_mask = np.isfinite(raw)
            nan_count = int((~finite_mask).sum())
            raw_finite = raw[finite_mask]
            if len(raw_finite) > 0:
                sig = 1.0 / (1.0 + np.exp(-raw_finite))
                counts, edges = np.histogram(sig, bins=bins, range=(0.0, 1.0))
                opacity_hist = {
                    "bins": edges.tolist(),
                    "counts": counts.tolist(),
                    "unit": "sigmoid_opacity",
                    "nan_count": nan_count,
                }
        except Exception:
            pass

    return {
        "scale": scale_hist,
        "opacity": opacity_hist,
    }


def collect_fastergs(
    run_dir: Path,
    log_text: str,
    start_wall: float | None,
) -> tuple[list[dict], dict]:
    # Parse every progress + eval line into a single iteration-indexed map,
    # then merge in per-checkpoint gaussian counts + wall times + SSIM/LPIPS.
    # This gives us a dense loss curve starting near iter 0, with sparse
    # PSNR/SSIM/LPIPS spikes at eval iterations, and gaussian-count samples
    # at save-checkpoint iterations.
    progress_records = parse_progress_log(log_text)

    # Start from the progress-log records (dense iteration grid with loss
    # + any PSNR from eval lines) and layer checkpoint data on top.
    by_iter: dict[int, dict] = {
        r["iteration"]: {
            "iteration": r["iteration"],
            "loss": r.get("loss"),
            "psnr": r.get("psnr"),
            "ssim": r.get("ssim"),
            "lpips": r.get("lpips"),
            "num_gaussians": None,
            "splats_per_frame": None,
            "wall_seconds": None,
        }
        for r in progress_records
    }

    checkpoints = find_checkpoint_iterations(run_dir)
    latest_ply_path: Path | None = None

    for iteration in checkpoints:
        rec = by_iter.get(iteration) or {
            "iteration": iteration,
            "loss": None,
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "num_gaussians": None,
            "splats_per_frame": None,
            "wall_seconds": None,
        }

        ply_path = run_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        if ply_path.is_file():
            latest_ply_path = ply_path
            g = count_ply_gaussians(ply_path)
            rec["num_gaussians"] = g
            # True per-frame splat counts would need a rasterizer hook we
            # don't have yet; every active gaussian is processed per frame
            # during training, so this is a reasonable proxy.
            rec["splats_per_frame"] = g

            if start_wall is not None:
                try:
                    rec["wall_seconds"] = max(0.0, ply_path.stat().st_mtime - start_wall)
                except Exception:
                    pass

        # SSIM/LPIPS: only computable if eval images exist on disk.
        renders_dir, gt_dir = find_eval_dirs(run_dir, iteration)
        ssim_val, lpips_val = compute_image_metrics(renders_dir, gt_dir)
        if ssim_val is not None:
            rec["ssim"] = ssim_val
        if lpips_val is not None:
            rec["lpips"] = lpips_val

        by_iter[iteration] = rec

    records = [by_iter[k] for k in sorted(by_iter.keys())]

    summary_extras = {}
    if records:
        # "Final" means the last record that actually carries each metric -
        # PSNR usually comes from the final eval line, gaussians from the
        # last checkpoint. Neither is guaranteed to be the literal last row.
        def _last_non_null(key):
            for r in reversed(records):
                if r.get(key) is not None:
                    return r[key]
            return None

        summary_extras = {
            "final_psnr": _last_non_null("psnr"),
            "final_ssim": _last_non_null("ssim"),
            "final_lpips": _last_non_null("lpips"),
            "final_num_gaussians": _last_non_null("num_gaussians"),
            "final_splats_per_frame": _last_non_null("splats_per_frame"),
            "final_loss": _last_non_null("loss"),
            "total_wall_seconds": _last_non_null("wall_seconds"),
        }

    # Histograms from the newest checkpoint PLY. Used by the Reports page to
    # validate Shorter-Splatting's claims about scale + opacity distributions.
    if latest_ply_path is not None:
        hists = compute_ply_histograms(latest_ply_path)
        if hists:
            summary_extras["histograms"] = hists

    return records, summary_extras


def collect_opensplat(
    run_dir: Path,
    log_text: str,
    start_wall: float | None,
) -> tuple[list[dict], dict]:
    # Parse the "Step N: loss (pct%)" trajectory and layer per-checkpoint
    # gaussian counts on top so the frontend gets a dense loss curve plus
    # the same fastergs-style summary fields. PSNR/SSIM/LPIPS stay null for
    # now; OpenSplat doesn't evaluate against held-out views during training.
    step_records, last_psnr, last_iter = parse_opensplat_log(log_text)

    # OpenSplat writes the PLY to one of two places depending on how it was
    # invoked: the opensplat sbatch we generate puts it under
    # point_cloud/iteration_<N>/point_cloud.ply to mirror fastergs; the
    # standalone wrapper puts it directly in the run dir. Both are handled.
    checkpoints = find_checkpoint_iterations(run_dir)
    latest_ply_path: Path | None = None
    if checkpoints:
        latest_ply_path = run_dir / "point_cloud" / f"iteration_{checkpoints[-1]}" / "point_cloud.ply"
    if latest_ply_path is None or not latest_ply_path.is_file():
        for candidate in (run_dir / "point_cloud.ply", run_dir / "splat.ply"):
            if candidate.is_file():
                latest_ply_path = candidate
                break

    num_gaussians = count_ply_gaussians(latest_ply_path) if latest_ply_path and latest_ply_path.is_file() else None
    wall_seconds = None
    if start_wall is not None and latest_ply_path is not None and latest_ply_path.is_file():
        try:
            wall_seconds = max(0.0, latest_ply_path.stat().st_mtime - start_wall)
        except Exception:
            pass

    by_iter: dict[int, dict] = {
        r["iteration"]: {
            "iteration": r["iteration"],
            "loss": r.get("loss"),
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "num_gaussians": None,
            "splats_per_frame": None,
            "wall_seconds": None,
        }
        for r in step_records
    }

    # Attach gaussian count + wall_seconds to the final-iteration record.
    final_iter = last_iter if last_iter is not None else (checkpoints[-1] if checkpoints else None)
    if final_iter is not None:
        rec = by_iter.get(final_iter) or {
            "iteration": final_iter,
            "loss": None,
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "num_gaussians": None,
            "splats_per_frame": None,
            "wall_seconds": None,
        }
        if num_gaussians is not None:
            rec["num_gaussians"] = num_gaussians
            rec["splats_per_frame"] = num_gaussians
        if wall_seconds is not None:
            rec["wall_seconds"] = wall_seconds
        if last_psnr is not None:
            rec["psnr"] = last_psnr
        by_iter[final_iter] = rec

    records = [by_iter[k] for k in sorted(by_iter.keys())]

    def _last_non_null(key):
        for r in reversed(records):
            if r.get(key) is not None:
                return r[key]
        return None

    summary_extras = {
        "final_psnr": _last_non_null("psnr"),
        "final_ssim": None,
        "final_lpips": None,
        "final_num_gaussians": num_gaussians,
        "final_splats_per_frame": num_gaussians,
        "final_loss": _last_non_null("loss"),
        "total_wall_seconds": wall_seconds,
    }

    # Histograms from the final PLY. Reports page uses these to compare
    # scale/opacity distributions across backends just like it does for the
    # fastergs variants.
    if latest_ply_path is not None and latest_ply_path.is_file():
        hists = compute_ply_histograms(latest_ply_path)
        if hists:
            summary_extras["histograms"] = hists

    return records, summary_extras


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
    # Single-line error so the SLURM log doesn't drown in tracebacks if
    # some unexpected trainer output trips the parser. Callers treat
    # non-zero exit as "no metrics for this run" and keep going.
    try:
        main()
    except Exception as exc:
        print(f"[metrics_collector] error: {exc}", file=sys.stderr)
        sys.exit(1)
