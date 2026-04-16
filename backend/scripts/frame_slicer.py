from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import cv2

logger = logging.getLogger("gaussian.frame_slicer")


def video_slicer(
    video_path,
    output_dir,
    img_format,
    fps=None,
    downscale: float = 1.0,
    *,
    return_metadata: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # keep upstream's range checks; add one for downscale too
    if fps is not None and (fps < 0 or fps > 120):
        raise ValueError("fps must be between 0 and 120")
    if downscale < 0.1 or downscale > 1:
        raise ValueError("downscale must be between 0.1 and 1")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    start = time.perf_counter()

    native_fps = video.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        step = max(1, round(native_fps / fps))
    else:
        step = 1

    saved = 0
    i = 0

    while True:
        cont, frame = video.read()
        if not cont:
            break

        if i % step == 0:
            # my addition: optional downscale before writing, saves disk + later CPU
            if downscale < 1:
                h, w = frame.shape[:2]
                new_w = max(1, int(w * downscale))
                new_h = max(1, int(h * downscale))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            filename = f"frame_{i:06d}.{img_format}"
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), frame)
            saved += 1
        i += 1

    video.release()

    elapsed_sec = time.perf_counter() - start
    logger.info("Video slicing wrote %s frames to %s", saved, output_dir)

    # my addition: return a metadata dict so pipeline.py can log stats.
    # default behavior (return saved count) is unchanged.
    if return_metadata:
        return {
            "saved": saved,
            "source_fps": float(native_fps or 0.0),
            "target_fps": float(fps or 0.0),
            "frame_step": step,
            "downscale": downscale,
            "elapsed_sec": elapsed_sec,
        }
    return saved
