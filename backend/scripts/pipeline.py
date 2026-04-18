# Third-party code and techniques used in this file:
#  - OpenSplat (Pierotofy) for the local C++/libtorch training backend
#    https://github.com/pierotofy/OpenSplat
#  - COLMAP Structure-from-Motion pipeline (https://colmap.github.io)
"""Local pipeline: preprocess frames, run COLMAP SfM, train with OpenSplat.

Primary entry point for the OpenSplat backend. See
``fastergs_pipeline.py`` for the HPG/Faster-GS remote variant.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .converter import ply_to_splat
    from .frame_slicer import video_slicer
    from .preprocessor import preprocessor
    from .prep_fingerprint import (
        build_fingerprint,
        diff_fingerprints,
        fingerprints_match,
        load_fingerprint,
        save_fingerprint,
    )
except ImportError:
    from converter import ply_to_splat
    from frame_slicer import video_slicer
    from preprocessor import preprocessor
    from prep_fingerprint import (
        build_fingerprint,
        diff_fingerprints,
        fingerprints_match,
        load_fingerprint,
        save_fingerprint,
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("gaussian.pipeline")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

MIN_RAW_FRAMES = 10
MIN_PROCESSED_FRAMES = 8

# Safer demo defaults to reduce preprocessing + SfM runtime while preserving enough viewpoints.
DEFAULT_DUPLICATE_THRESHOLD = 1.5
DEFAULT_BLUR_THRESHOLD = 20.0
DEFAULT_FPS = 12.0
DEFAULT_DOWNSCALE = 0.75
DEFAULT_MAX_WIDTH = 1280


def run_command(cmd, cwd=HERE):
    cmd_str = " ".join(str(part) for part in cmd)
    logger.info("Running: %s (cwd=%s)", cmd_str, cwd)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def clear_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return len([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS])


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _suspicious_warnings(raw_count: int, prep_stats: dict) -> list[str]:
    warnings: list[str] = []

    total = int(prep_stats.get("total", 0))
    kept = int(prep_stats.get("kept", 0))
    skipped_blur = int(prep_stats.get("skipped_blur", 0))
    skipped_duplicate = int(prep_stats.get("skipped_duplicate", 0))

    keep_ratio = (kept / total) if total else 0.0
    blur_drop_ratio = (skipped_blur / total) if total else 0.0
    dup_drop_ratio = (skipped_duplicate / total) if total else 0.0

    if total == 0:
        warnings.append("No frames were available for preprocessing.")
        return warnings

    if keep_ratio > 0.98 and total >= 120:
        warnings.append(
            "Very high keep ratio (>98%). Filtering may be too weak; SfM/runtime may be slower than necessary."
        )

    if keep_ratio < 0.20 and total >= 40:
        warnings.append(
            "Very low keep ratio (<20%). Filtering may be too aggressive; reconstruction quality may drop."
        )

    if raw_count > 300:
        warnings.append("High raw frame count (>300). Consider lower FPS/downscale for faster runs.")

    if skipped_duplicate == 0 and total >= 120:
        warnings.append("No frames were flagged as duplicates. Duplicate threshold may be too low for this video.")

    if blur_drop_ratio > 0.70:
        warnings.append("More than 70% of frames were dropped as blurry. Blur threshold may be too strict.")

    if dup_drop_ratio > 0.80:
        warnings.append("More than 80% of frames were dropped as duplicates. Duplicate threshold may be too strict.")

    return warnings


def _write_preprocess_report(dataset_path: Path, report: dict):
    report_path = dataset_path / "preprocess_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Preprocess report written: %s", report_path)


def validate_run_args(args):
    if not DATASET_RE.fullmatch(args.dataset):
        raise ValueError("dataset name is invalid (allowed: letters, numbers, _ . -)")
    if not re.fullmatch(r"[A-Za-z0-9]+", args.img_format):
        raise ValueError("img_format must be alphanumeric (for example: jpg or png)")
    if not (50 <= args.iters <= 100000):
        raise ValueError("iters must be between 50 and 100000")
    if not (0 <= args.duplicate_threshold <= 255):
        raise ValueError("duplicate_threshold must be between 0 and 255")
    if not (0 <= args.blur_threshold <= 5000):
        raise ValueError("blur_threshold must be between 0 and 5000")
    if not (0 <= args.fps <= 120):
        raise ValueError("fps must be between 0 and 120")
    if not (0.1 <= args.downscale <= 1):
        raise ValueError("downscale must be between 0.1 and 1")
    if not (320 <= args.max_width <= 4096):
        raise ValueError("max_width must be between 320 and 4096")


def run_sfm(dataset_path: Path):
    # Smart SfM Reuse: skip COLMAP if the existing sparse model was built
    # from the same images currently on disk. We key this off the prep
    # fingerprint (since those settings determine the input images) plus
    # the presence of the sparse output.
    image_dir = dataset_path / "images"
    image_count = count_images(image_dir)
    if image_count < MIN_PROCESSED_FRAMES:
        raise ValueError(
            f"Not enough processed images for SfM ({image_count} found, need at least {MIN_PROCESSED_FRAMES})"
        )

    prep_fp_path = dataset_path / "prep_fingerprint.json"
    sfm_fp_path = dataset_path / "sfm_fingerprint.json"
    sparse_dir = dataset_path / "sparse"
    prep_fp = load_fingerprint(prep_fp_path) or {}
    sfm_fp = load_fingerprint(sfm_fp_path)

    sparse_ok = sparse_dir.is_dir() and any(sparse_dir.iterdir())
    if sparse_ok and fingerprints_match(sfm_fp, prep_fp):
        logger.info("SfM cache hit: reusing existing sparse model (preprocess settings match)")
        return
    if sparse_ok and sfm_fp:
        reasons = diff_fingerprints(sfm_fp, prep_fp)
        for reason in reasons:
            logger.info("SfM cache invalidated: %s", reason)

    try:
        try:
            from .sfm import sfm
        except ImportError:
            from sfm import sfm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pycolmap is required for SfM step but is not installed in this environment"
        ) from exc

    sfm(str(image_dir), str(sparse_dir))

    # Cache the fingerprint so the next run can skip SfM when preprocessing
    # settings haven't changed. We copy from the prep fingerprint because
    # SfM's validity tracks whichever images were last produced.
    if prep_fp:
        save_fingerprint(sfm_fp_path, prep_fp)


def run_opensplat(dataset_path: Path, num_iters: int):
    exe_suffix = ".exe" if sys.platform == "win32" else ""
    opensplat_path = PARENT / f"binaries/opensplat{exe_suffix}"

    if not opensplat_path.exists() or not opensplat_path.is_file():
        raise FileNotFoundError(f"opensplat not found (expected {opensplat_path})")

    output_ply = dataset_path / "splat.ply"
    cmd = [
        str(opensplat_path),
        str(dataset_path),
        "-o",
        str(output_ply),
        "-n",
        str(num_iters),
    ]

    # Capture opensplat stdout so the metrics collector can parse it after.
    # We still forward every line to our own stdout so the user sees live
    # progress in the web UI log stream.
    run_tag_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_tag = f"{dataset_path.name}_opensplat_{run_tag_ts}"
    metrics_dir = dataset_path / "metrics" / run_tag
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_log = metrics_dir / "train_stdout.log"

    logger.info("Running: %s", " ".join(str(p) for p in cmd))
    train_start_epoch = time.time()
    captured_lines: list[str] = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(HERE), text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            captured_lines.append(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)
    except subprocess.CalledProcessError as exc:
        try:
            train_log.write_text("".join(captured_lines))
        except Exception:
            pass
        raise RuntimeError(
            "OpenSplat runtime failed. This local binary likely depends on missing local dylibs/libraries. "
            "Use a rebuilt local binary or run remote OpenSplat on HiPerGator."
        ) from exc

    try:
        train_log.write_text("".join(captured_lines))
    except Exception:
        pass

    if not output_ply.exists() or output_ply.stat().st_size == 0:
        raise RuntimeError("opensplat did not produce a valid splat.ply")

    output_splat = Path(ply_to_splat(str(output_ply)))
    if not output_splat.exists() or output_splat.stat().st_size == 0:
        raise RuntimeError("PLY to SPLAT conversion failed")

    logger.info("Produced output: %s", output_splat)

    # Best-effort metrics collection for the OpenSplat path. The collector
    # pulls whatever PSNR/iteration info it can scrape from the captured log
    # plus gaussian count from the PLY. SSIM/LPIPS stay null (OpenSplat
    # doesn't emit eval images). If the collector or plotter errors out,
    # training still counts as a success - we just log a warning.
    metrics_collector = HERE / "metrics_collector.py"
    metrics_plotter = HERE / "metrics_plotter.py"
    if metrics_collector.is_file():
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(metrics_collector),
                    "--run-dir", str(dataset_path),
                    "--log-file", str(train_log),
                    "--out-dir", str(metrics_dir),
                    "--backend", "opensplat",
                    "--dataset", dataset_path.name,
                    "--run-tag", run_tag,
                    "--iterations", str(num_iters),
                    "--start-epoch", str(train_start_epoch),
                ],
                check=True,
            )
        except Exception as exc:
            logger.warning("metrics_collector failed: %s", exc)
    if metrics_plotter.is_file():
        try:
            subprocess.run(
                [sys.executable, str(metrics_plotter), "--metrics-dir", str(metrics_dir),
                 "--title-prefix", f"{dataset_path.name} {run_tag}"],
                check=True,
            )
        except Exception as exc:
            logger.warning("metrics_plotter failed: %s", exc)


def run_prepare(
    dataset_path: Path,
    video_path: Path | None,
    img_format: str,
    duplicate_threshold: float,
    blur_threshold: float,
    fps: float,
    downscale: float,
    max_width: int,
    write_scores: bool,
    *,
    use_existing_frames: bool = False,
):
    raw_path = dataset_path / "raw"
    images_path = dataset_path / "images"
    fingerprint_path = dataset_path / "prep_fingerprint.json"

    # Existing-frames mode: skip video_slicer and re-run ONLY the
    # preprocessor on the existing raw/ dir with the current blur / dup /
    # downscale / max_width settings. FPS isn't meaningful here because
    # the raw frames are already extracted - we inherit it from the prior
    # fingerprint (if any) so the fingerprint comparison stays stable and
    # doesn't needlessly invalidate.
    prior_fp = load_fingerprint(fingerprint_path)
    if use_existing_frames:
        if not raw_path.is_dir() or count_images(raw_path) == 0:
            # Fallback: no raw/ saved (older dataset). Promote the
            # existing images/ dir to raw/ so we have a stable source
            # that survives the clear_dir(images_path) below. Future
            # existing-frames reruns on this dataset now pick up the
            # real raw path automatically. If the ultimate fallback
            # fails, surface a clear error.
            if not images_path.is_dir() or count_images(images_path) == 0:
                raise FileNotFoundError(
                    f"No raw/ or images/ found under {dataset_path}; cannot re-run preprocessing."
                )
            logger.warning(
                "No raw/ frames for %s; promoting existing images/ to raw/ before re-filtering",
                dataset_path.name,
            )
            raw_path.mkdir(parents=True, exist_ok=True)
            for img in images_path.iterdir():
                if img.is_file():
                    img.rename(raw_path / img.name)
        source_path = raw_path
        fps_for_fingerprint = (prior_fp or {}).get("fps", fps)
    else:
        fps_for_fingerprint = fps

    # Smart SfM Reuse: if the preprocessing settings match the last run AND
    # the output images dir already has content, skip the expensive slicing
    # + blur/duplicate filtering step entirely and go straight to SfM.
    new_fp = build_fingerprint(
        fps=fps_for_fingerprint, downscale=downscale, blur_threshold=blur_threshold,
        duplicate_threshold=duplicate_threshold, max_width=max_width,
    )
    if (
        fingerprints_match(prior_fp, new_fp)
        and images_path.is_dir()
        and count_images(images_path) >= MIN_PROCESSED_FRAMES
    ):
        logger.info(
            "Preprocessing cache hit: reusing %d images (fps=%s downscale=%s blur=%s duplicate=%s max_width=%s)",
            count_images(images_path), fps_for_fingerprint, downscale, blur_threshold, duplicate_threshold, max_width,
        )
        return
    # If we have a prior fingerprint but it doesn't match, tell the user
    # which settings changed so they understand why preprocessing reran.
    if prior_fp:
        reasons = diff_fingerprints(prior_fp, new_fp)
        for reason in reasons:
            logger.info("Preprocessing cache invalidated: %s", reason)

    prepare_start = time.perf_counter()

    if use_existing_frames:
        # Preserve raw/, only rebuild images/.
        raw_count = count_images(source_path)
        logger.info(
            "Re-running preprocessing on %d existing frames (blur=%s dup=%s downscale=%s max_width=%s)",
            raw_count, blur_threshold, duplicate_threshold, downscale, max_width,
        )
        clear_dir(images_path)
        slicing_elapsed = 0.0
        slice_stats = {
            "saved": raw_count,
            "source_fps": None,
            "target_fps": None,
            "frame_step": None,
            "downscale": downscale,
        }
    else:
        clear_dir(raw_path)
        clear_dir(images_path)

        logger.info("Starting video slicing")
        slice_start = time.perf_counter()
        slice_stats = video_slicer(
            video_path,
            raw_path,
            img_format,
            fps=fps,
            downscale=downscale,
            return_metadata=True,
        )
        slicing_elapsed = time.perf_counter() - slice_start
        source_path = raw_path

        raw_count = int(slice_stats["saved"])
        logger.info("Video slicing finished (%s frames in %.2fs)", raw_count, slicing_elapsed)

        if raw_count < MIN_RAW_FRAMES:
            raise ValueError(
                f"Too few extracted frames ({raw_count}). Provide a longer/steadier video with more viewpoints."
            )

    logger.info("Starting preprocessing")
    prep_start = time.perf_counter()
    prep_stats = preprocessor(
        source_path,
        images_path,
        duplicate_threshold=duplicate_threshold,
        blur_threshold=blur_threshold,
        max_output_width=max_width,
        # Only apply downscale in existing-frames mode. In the normal
        # video path, video_slicer has already downscaled during
        # extraction, so passing it again here would compound.
        downscale=downscale if use_existing_frames else 1.0,
        write_scores_csv=write_scores,
        scores_csv_path=dataset_path / "preprocess_scores.csv",
    )
    preprocess_elapsed = time.perf_counter() - prep_start

    kept_count = prep_stats["kept"]
    logger.info(
        "Preprocessing finished (%s/%s frames kept in %.2fs)",
        kept_count,
        prep_stats["total"],
        preprocess_elapsed,
    )

    warnings = _suspicious_warnings(raw_count=raw_count, prep_stats=prep_stats)
    for warning in warnings:
        logger.warning("PREPROCESS WARNING: %s", warning)

    report = {
        "created_at": _utcnow_iso(),
        "dataset": dataset_path.name,
        "video": {
            "input_path": str(video_path) if video_path is not None else None,
            "reused_existing_frames": use_existing_frames,
            "source_fps": slice_stats.get("source_fps"),
            "target_fps": slice_stats.get("target_fps"),
            "frame_step": slice_stats.get("frame_step"),
            "downscale": slice_stats.get("downscale"),
            "raw_frames_saved": raw_count,
        },
        "preprocess": {
            "thresholds": {
                "duplicate_threshold": duplicate_threshold,
                "blur_threshold": blur_threshold,
                "max_output_width": max_width,
            },
            "counts": {
                "total_frames": prep_stats["total"],
                "kept_frames": prep_stats["kept"],
                "dropped_blurry_frames": prep_stats["skipped_blur"],
                "dropped_duplicate_frames": prep_stats["skipped_duplicate"],
                "invalid_frames": prep_stats["skipped_invalid"],
                "resized_frames": prep_stats["resized_count"],
            },
            "ratios": {
                "keep_ratio": prep_stats["keep_ratio"],
                "dropped_ratio": prep_stats["dropped_ratio"],
            },
            "scores": {
                "blur_mean": prep_stats["blur_score_mean"],
                "blur_median": prep_stats["blur_score_median"],
                "duplicate_mean": prep_stats["duplicate_score_mean"],
                "duplicate_median": prep_stats["duplicate_score_median"],
            },
            "warnings": warnings,
            "scores_csv_path": prep_stats["scores_csv_path"],
        },
        "timings_sec": {
            "video_slicing": slicing_elapsed,
            "preprocessing": preprocess_elapsed,
            "prepare_total": time.perf_counter() - prepare_start,
        },
    }
    _write_preprocess_report(dataset_path, report)

    if kept_count < MIN_PROCESSED_FRAMES:
        raise ValueError(
            f"Too few usable frames after preprocessing ({kept_count}). "
            "Lower thresholds or use a clearer video with more camera movement."
        )

    # Cache the successful fingerprint so future runs with the same
    # preprocessing settings can skip this whole function.
    save_fingerprint(fingerprint_path, new_fp)


def main():
    parser = argparse.ArgumentParser(description="Orchestrate SfM and Gaussian Splatting")
    parser.add_argument("dataset", help="Dataset name inside datasets/")
    parser.add_argument("--iters", type=int, default=1000, help="Number of OpenSplat iterations")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--img_format", default="jpg", help="Frame image format")
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument("--blur-threshold", type=float, default=DEFAULT_BLUR_THRESHOLD)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Target extraction FPS (0 means original video FPS)")
    parser.add_argument("--downscale", type=float, default=DEFAULT_DOWNSCALE, help="Frame downscale factor (0.1 to 1)")
    parser.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="Maximum output frame width after preprocessing")
    parser.add_argument("--no-scores", action="store_true", help="Skip writing preprocess_scores.csv")
    parser.add_argument(
        "--only",
        choices=["prepare", "sfm", "opensplat", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    # When set, skip the prepare stage even under --only all/prepare.
    # Pairs with the LiveDemos "Existing dataset" flow where we want to
    # rerun SfM / training with different settings but keep the already-
    # extracted frames as-is.
    parser.add_argument(
        "--use-existing-frames",
        action="store_true",
        help="Skip frame extraction and reuse the existing dataset images/ dir.",
    )
    args = parser.parse_args()

    validate_run_args(args)

    dataset_path = PARENT / "datasets" / args.dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        if args.only in ("prepare", "all"):
            if args.use_existing_frames:
                # Existing-dataset flow. Skip video_slicer entirely but
                # still re-run the preprocessor on raw/ so the user can
                # change blur / dup / downscale / max_width without
                # re-uploading a video.
                run_prepare(
                    dataset_path,
                    None,
                    args.img_format,
                    duplicate_threshold=args.duplicate_threshold,
                    blur_threshold=args.blur_threshold,
                    fps=args.fps,
                    downscale=args.downscale,
                    max_width=args.max_width,
                    write_scores=not args.no_scores,
                    use_existing_frames=True,
                )
            else:
                if not args.video:
                    raise ValueError("--video is required when running prepare/all")
                video_path = Path(args.video).resolve()
                if not video_path.exists():
                    raise FileNotFoundError(f"Video not found: {video_path}")

                run_prepare(
                    dataset_path,
                    video_path,
                    args.img_format,
                    duplicate_threshold=args.duplicate_threshold,
                    blur_threshold=args.blur_threshold,
                    fps=args.fps,
                    downscale=args.downscale,
                    max_width=args.max_width,
                    write_scores=not args.no_scores,
                )

        if args.only in ("sfm", "all"):
            logger.info("Starting SfM step")
            run_sfm(dataset_path)
            logger.info("SfM step finished successfully")

        if args.only in ("opensplat", "all"):
            logger.info("Starting Gaussian Splatting (opensplat)")
            run_opensplat(dataset_path, args.iters)
            logger.info("Gaussian Splatting finished successfully")

    except subprocess.CalledProcessError as exc:
        logger.error("Command failed with exit code %s", exc.returncode)
        sys.exit(exc.returncode)
    except Exception as exc:
        logger.exception("Error during run: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
