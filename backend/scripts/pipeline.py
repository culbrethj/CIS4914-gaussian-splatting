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
except ImportError:
    from converter import ply_to_splat
    from frame_slicer import video_slicer
    from preprocessor import preprocessor

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
    try:
        try:
            from .sfm import sfm
        except ImportError:
            from sfm import sfm
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pycolmap is required for SfM step but is not installed in this environment"
        ) from exc

    image_dir = dataset_path / "images"
    image_count = count_images(image_dir)
    if image_count < MIN_PROCESSED_FRAMES:
        raise ValueError(
            f"Not enough processed images for SfM ({image_count} found, need at least {MIN_PROCESSED_FRAMES})"
        )

    sfm(str(image_dir), str(dataset_path / "sparse"))


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

    try:
        run_command(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "OpenSplat runtime failed. This local binary likely depends on missing local dylibs/libraries. "
            "Use a rebuilt local binary or run remote OpenSplat on HiPerGator."
        ) from exc

    if not output_ply.exists() or output_ply.stat().st_size == 0:
        raise RuntimeError("opensplat did not produce a valid splat.ply")

    output_splat = Path(ply_to_splat(str(output_ply)))
    if not output_splat.exists() or output_splat.stat().st_size == 0:
        raise RuntimeError("PLY to SPLAT conversion failed")

    logger.info("Produced output: %s", output_splat)


def run_prepare(
    dataset_path: Path,
    video_path: Path,
    img_format: str,
    duplicate_threshold: float,
    blur_threshold: float,
    fps: float,
    downscale: float,
    max_width: int,
    write_scores: bool,
):
    raw_path = dataset_path / "raw"
    images_path = dataset_path / "images"

    clear_dir(raw_path)
    clear_dir(images_path)

    prepare_start = time.perf_counter()

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

    raw_count = int(slice_stats["saved"])
    logger.info("Video slicing finished (%s frames in %.2fs)", raw_count, slicing_elapsed)

    if raw_count < MIN_RAW_FRAMES:
        raise ValueError(
            f"Too few extracted frames ({raw_count}). Provide a longer/steadier video with more viewpoints."
        )

    logger.info("Starting preprocessing")
    prep_start = time.perf_counter()
    prep_stats = preprocessor(
        raw_path,
        images_path,
        duplicate_threshold=duplicate_threshold,
        blur_threshold=blur_threshold,
        max_output_width=max_width,
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
            "input_path": str(video_path),
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
    args = parser.parse_args()

    validate_run_args(args)

    dataset_path = PARENT / "datasets" / args.dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        if args.only in ("prepare", "all"):
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
