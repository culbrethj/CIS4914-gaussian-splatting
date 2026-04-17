from __future__ import annotations

import csv
import logging
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("gaussian.preprocessor")


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _safe_median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def preprocessor(
    raw_images,
    output_path,
    duplicate_threshold=3.0,
    blur_threshold=50,
    *,
    max_output_width=1280,
    downscale=1.0,
    write_scores_csv=True,
    scores_csv_path=None,
):
    """
    Preprocess extracted frames and return a stats dict.

    Steps per frame:
    - optional downscale by `downscale` factor
    - optional resize capped at max_output_width
    - blur filtering using Laplacian variance
    - duplicate filtering using MAD to previous kept frame

    `downscale` lets callers re-run preprocessing on already-extracted
    raw frames (existing-dataset mode in LiveDemos). In the normal
    video path, the frames arrive pre-scaled by video_slicer and this
    preprocessor is called with downscale=1.0 so the factor isn't
    applied twice.
    """
    if duplicate_threshold < 0:
        raise ValueError("duplicate_threshold must be >= 0")
    if blur_threshold < 0:
        raise ValueError("blur_threshold must be >= 0")
    if max_output_width < 320:
        raise ValueError("max_output_width must be >= 320")
    if downscale <= 0 or downscale > 1:
        raise ValueError("downscale must be in (0, 1]")

    path = Path(raw_images)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"raw image path not found: {path}")

    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS])

    score_writer = None
    score_file = None
    if write_scores_csv:
        csv_path = Path(scores_csv_path) if scores_csv_path else out_path.parent / "preprocess_scores.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        score_file = csv_path.open("w", newline="", encoding="utf-8")
        score_writer = csv.DictWriter(
            score_file,
            fieldnames=[
                "frame_index",
                "filename",
                "width",
                "height",
                "resized_width",
                "resized_height",
                "blur_score",
                "duplicate_score",
                "decision",
                "reason",
            ],
        )
        score_writer.writeheader()

    start = time.perf_counter()

    prev = None
    total = len(imgs)
    kept = 0
    skipped_blur = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    resized_count = 0

    blur_scores: list[float] = []
    duplicate_scores: list[float] = []

    try:
        for idx, img_path in enumerate(imgs):
            image = cv2.imread(str(img_path))
            if image is None:
                skipped_invalid += 1
                if score_writer:
                    score_writer.writerow(
                        {
                            "frame_index": idx,
                            "filename": img_path.name,
                            "width": 0,
                            "height": 0,
                            "resized_width": 0,
                            "resized_height": 0,
                            "blur_score": "",
                            "duplicate_score": "",
                            "decision": "drop",
                            "reason": "invalid_image",
                        }
                    )
                continue

            height, width = image.shape[:2]
            resized_image = image
            resized_width = width
            resized_height = height

            # Combined scale: shrink by `downscale` factor first (used in
            # existing-frames mode to apply a user-selected downscale on
            # already-extracted raw frames), then cap the result at
            # max_output_width. Both steps are resolution-reducing, so
            # we only resize if the final target is smaller than the
            # source.
            target_width = width
            if downscale < 1.0:
                target_width = max(1, int(round(width * downscale)))
            if target_width > max_output_width:
                target_width = max_output_width
            if target_width < width:
                scale = target_width / width
                resized_height = max(1, int(round(height * scale)))
                resized_width = target_width
                resized_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
                resized_count += 1

            gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

            # Blur score = variance of the Laplacian. Low variance means the
            # image has few sharp edges, so it's probably motion-blurred or
            # out of focus. Anything under blur_threshold gets dropped.
            laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
            blur_score = float(laplacian.var())
            blur_scores.append(blur_score)

            duplicate_score = None
            reason = "kept"
            decision = "keep"

            if blur_score < blur_threshold:
                skipped_blur += 1
                decision = "drop"
                reason = "blur"
            else:
                # Duplicate detection: shrink to 64x36 and compare to the
                # previous kept frame with mean absolute difference. Tiny
                # numbers mean "almost identical to the previous frame" so we
                # skip it - no new information for SfM.
                downscaled_img = cv2.resize(gray_img, (64, 36), interpolation=cv2.INTER_AREA)

                keep = True
                if prev is not None:
                    diff = np.abs(downscaled_img.astype(np.float32) - prev.astype(np.float32))
                    duplicate_score = float(diff.mean())
                    duplicate_scores.append(duplicate_score)
                    if duplicate_score < duplicate_threshold:
                        keep = False

                if keep:
                    out_file = out_path / img_path.name
                    ok = cv2.imwrite(str(out_file), resized_image)
                    if not ok:
                        skipped_invalid += 1
                        decision = "drop"
                        reason = "write_failed"
                    else:
                        prev = downscaled_img
                        kept += 1
                else:
                    skipped_duplicate += 1
                    decision = "drop"
                    reason = "duplicate"

            if score_writer:
                score_writer.writerow(
                    {
                        "frame_index": idx,
                        "filename": img_path.name,
                        "width": width,
                        "height": height,
                        "resized_width": resized_width,
                        "resized_height": resized_height,
                        "blur_score": f"{blur_score:.4f}",
                        "duplicate_score": "" if duplicate_score is None else f"{duplicate_score:.4f}",
                        "decision": decision,
                        "reason": reason,
                    }
                )
    finally:
        if score_file:
            score_file.close()

    elapsed_sec = time.perf_counter() - start
    keep_ratio = (kept / total) if total else 0.0

    stats = {
        "total": total,
        "kept": kept,
        "skipped_blur": skipped_blur,
        "skipped_duplicate": skipped_duplicate,
        "skipped_invalid": skipped_invalid,
        "resized_count": resized_count,
        "keep_ratio": keep_ratio,
        "dropped_ratio": 1.0 - keep_ratio if total else 0.0,
        "blur_score_mean": _safe_mean(blur_scores),
        "blur_score_median": _safe_median(blur_scores),
        "duplicate_score_mean": _safe_mean(duplicate_scores),
        "duplicate_score_median": _safe_median(duplicate_scores),
        "elapsed_sec": elapsed_sec,
        "max_output_width": max_output_width,
        "scores_csv_path": str(csv_path) if write_scores_csv else None,
    }
    logger.info("Preprocessor stats: %s", stats)
    return stats
