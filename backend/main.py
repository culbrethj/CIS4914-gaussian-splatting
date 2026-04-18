"""FastAPI entry point for the Gaussian Splatting Studio backend.

Routes are grouped by concern:
    - Upload: /api/upload accepts a video and writes it into a dataset dir.
    - Jobs: /api/run starts a pipeline run; /api/jobs/* polls status and
      drains per-job log buffers. Live logs stream over a WebSocket at
      /ws/jobs/{job_id}.
    - Datasets: /api/datasets, /api/datasets/{name}/runs expose the
      prepared datasets and the training runs attached to each.
    - Health: /api/health is a cheap liveness probe used by the frontend
      banner to detect a missing backend.

Pipeline orchestration lives in two scripts this file shells out to:
    - backend/scripts/pipeline.py for the local OpenSplat backend
    - backend/scripts/fastergs_pipeline.py for the remote Faster-GS backend
      on HiPerGator
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_PIPELINE_STEPS = {"prepare", "sfm", "opensplat", "all"}
# Two training backends can be selected from the UI:
#   opensplat - runs locally against the bundled binary (CPU or CUDA)
#   fastergs  - sends the job to HiPerGator and trains there
ALLOWED_BACKENDS = {"opensplat", "fastergs"}
# SfM methods. colmap is the default (traditional + reliable); vggt is the
# feed-forward neural network alternative that runs on a GPU in under a
# minute. Keep the enum small so the frontend can map directly.
ALLOWED_SFM_METHODS = {"colmap", "vggt"}
DEFAULT_SFM_METHOD = "colmap"
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_DUPLICATE_THRESHOLD = 1.5
DEFAULT_BLUR_THRESHOLD = 20.0
DEFAULT_PREP_FPS = 12.0
DEFAULT_PREP_DOWNSCALE = 0.75
DEFAULT_PREP_MAX_WIDTH = 1280
DEFAULT_PIPELINE_BACKEND = "fastergs"

logger = logging.getLogger("gaussian.backend")

app = FastAPI()

# ensure an hpg dir exists (contains splat files on Hipergator)
hpg_dir = HERE / "hipergator"
hpg_dir.mkdir(parents=True, exist_ok=True)

# mount datasets directory so frontend can fetch files directly
datasets_dir = HERE / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")

# in-memory maps of job queues and process tasks
JOB_QUEUES: dict[str, asyncio.Queue[str]] = {}
JOB_TASKS: dict[str, asyncio.Task] = {}
JOB_META: dict[str, dict] = {}
# Local subprocess handles, keyed by job_id. Used by /api/jobs/{id}/cancel
# to terminate the running pipeline. Populated when a job starts, removed
# when it exits.
JOB_PROCS: dict[str, asyncio.subprocess.Process] = {}

SLURM_ID_RE = re.compile(r"Submitted SLURM job id:\s*(\d+)")

# Rolling buffer of the last N log lines per job. The WebSocket stream is
# ephemeral - once a line is consumed it's gone - so if the user
# navigates to another page and back, the LiveDemos component remounts
# with an empty logs array and the progress / stage cards go blank until
# new lines trickle in. The buffer lets the frontend replay recent
# history via GET /api/jobs/{id}/logs before the WS reconnects, so
# progress state survives navigation without extra server load.
from collections import deque  # noqa: E402

JOB_LOG_BUFFERS: dict[str, deque[str]] = {}
JOB_LOG_BUFFER_MAX = 1500


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_int(name: str, raw, *, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be an integer") from exc
    if not (minimum <= value <= maximum):
        raise HTTPException(status_code=400, detail=f"{name} must be in range [{minimum}, {maximum}]")
    return value


def _coerce_float(name: str, raw, *, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{name} must be a number") from exc
    if not (minimum <= value <= maximum):
        raise HTTPException(status_code=400, detail=f"{name} must be in range [{minimum}, {maximum}]")
    return value


def _validate_dataset_name(name: str) -> str:
    if not isinstance(name, str):
        raise HTTPException(status_code=400, detail="dataset must be a string")
    cleaned = name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="dataset is required")
    if Path(cleaned).name != cleaned:
        raise HTTPException(status_code=400, detail="invalid dataset name")
    if not DATASET_RE.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="invalid dataset name (allowed: letters, numbers, _ . -)")
    return cleaned


def _dataset_dir(dataset_name: str) -> Path:
    safe_name = _validate_dataset_name(dataset_name)
    ds_root = datasets_dir.resolve()
    ds_dir = (ds_root / safe_name).resolve()
    if ds_dir.parent != ds_root:
        raise HTTPException(status_code=400, detail="invalid dataset path")
    return ds_dir


def _pick_dataset_video_file(dataset_name: str) -> Path:
    video_dir = _dataset_dir(dataset_name) / "video"
    if not video_dir.exists() or not video_dir.is_dir():
        raise HTTPException(status_code=400, detail="no uploaded video found for dataset")

    video_files = [
        p
        for p in sorted(video_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
        if p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTS
    ]
    if not video_files:
        raise HTTPException(status_code=400, detail="no uploaded video file found for dataset")
    return video_files[0]


def _sanitize_filename(filename: str | None) -> str:
    base = Path(filename or "upload.mp4").name
    if not base:
        raise HTTPException(status_code=400, detail="invalid filename")
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(status_code=400, detail=f"unsupported video file extension: {ext or 'none'}")
    return base


def _parse_run_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid JSON payload")

    dataset_name = _validate_dataset_name(payload.get("dataset"))
    only = payload.get("only", "all")
    if only not in ALLOWED_PIPELINE_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"only must be one of {sorted(ALLOWED_PIPELINE_STEPS)}",
        )
    backend = payload.get("backend", DEFAULT_PIPELINE_BACKEND)
    if backend not in ALLOWED_BACKENDS:
        raise HTTPException(status_code=400, detail=f"backend must be one of {sorted(ALLOWED_BACKENDS)}")
    # sfm_method is a separate axis from the training backend. Most users
    # will leave this at "colmap"; opt into "vggt" for near-instant SfM.
    sfm_method = payload.get("sfm_method", DEFAULT_SFM_METHOD)
    if sfm_method not in ALLOWED_SFM_METHODS:
        raise HTTPException(status_code=400, detail=f"sfm_method must be one of {sorted(ALLOWED_SFM_METHODS)}")

    # "Use existing dataset" flag. When set, the pipeline skips video
    # extraction and runs straight against backend/datasets/<name>/images.
    # Lets the frontend iterate on training / SfM settings without re-
    # uploading the source video every time.
    use_existing_frames = bool(payload.get("use_existing_frames", False))

    return {
        "dataset": dataset_name,
        "backend": backend,
        "sfm_method": sfm_method,
        "use_existing_frames": use_existing_frames,
        "iters": _coerce_int("iters", payload.get("iters", 1000), minimum=50, maximum=100000),
        "only": only,
        "duplicate_threshold": _coerce_float(
            "duplicate_threshold",
            payload.get("duplicate_threshold", DEFAULT_DUPLICATE_THRESHOLD),
            minimum=0,
            maximum=255,
        ),
        "blur_threshold": _coerce_float(
            "blur_threshold",
            payload.get("blur_threshold", DEFAULT_BLUR_THRESHOLD),
            minimum=0,
            maximum=5000,
        ),
        "fps": _coerce_float("fps", payload.get("fps", DEFAULT_PREP_FPS), minimum=0, maximum=120),
        "downscale": _coerce_float(
            "downscale",
            payload.get("downscale", DEFAULT_PREP_DOWNSCALE),
            minimum=0.1,
            maximum=1,
        ),
        "max_width": _coerce_int(
            "max_width",
            payload.get("max_width", DEFAULT_PREP_MAX_WIDTH),
            minimum=320,
            maximum=4096,
        ),
        # Experiment / Shorter-Splatting fields. All optional - absent keys
        # just mean the technique is off. Validated by _coerce_int/_float so
        # bad input gets rejected with a 400 before we launch a subprocess.
        "seed": (
            _coerce_int("seed", payload["seed"], minimum=0, maximum=2**31 - 1)
            if "seed" in payload and payload["seed"] is not None else None
        ),
        "shortgs_scale_reset_every": (
            _coerce_int(
                "shortgs_scale_reset_every",
                payload["shortgs_scale_reset_every"],
                minimum=1, maximum=1_000_000,
            )
            if "shortgs_scale_reset_every" in payload else 0
        ),
        "shortgs_scale_reset_factor": (
            _coerce_float(
                "shortgs_scale_reset_factor",
                payload["shortgs_scale_reset_factor"],
                minimum=0.01, maximum=1.0,
            )
            if "shortgs_scale_reset_factor" in payload else 1.0
        ),
        "shortgs_entropy_weight": (
            _coerce_float(
                "shortgs_entropy_weight",
                payload["shortgs_entropy_weight"],
                minimum=0.0, maximum=10.0,
            )
            if "shortgs_entropy_weight" in payload else 0.0
        ),
        "shortgs_progressive_resolution": _validate_progressive_schedule(
            payload.get("shortgs_progressive_resolution", "")
        ),
    }


def _validate_progressive_schedule(value) -> str:
    # Expects a string like "0:0.25,5000:0.5,10000:1.0" or empty. We don't
    # trust the client format - reparse and re-stringify so garbage strings
    # (e.g. shell injection, extra whitespace) can't slip through to SLURM.
    if value is None or value == "":
        return ""
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="shortgs_progressive_resolution must be a string")
    value = value.strip()
    if not value:
        return ""
    parts = []
    for pair in value.split(","):
        if ":" not in pair:
            raise HTTPException(status_code=400, detail="progressive schedule entries must be 'iter:scale'")
        i_str, s_str = pair.split(":", 1)
        try:
            iteration = int(i_str.strip())
            scale = float(s_str.strip())
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"invalid progressive schedule entry {pair!r}") from exc
        if iteration < 0 or scale <= 0 or scale > 1:
            raise HTTPException(status_code=400, detail="progressive schedule iter must be >=0 and scale in (0, 1]")
        parts.append(f"{iteration}:{scale}")
    return ",".join(parts)


def _to_datasets_url(path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(datasets_dir.resolve())
    except ValueError:
        return None
    return f"/datasets/{rel.as_posix()}"


def _find_root_splat(dataset_dir: Path) -> Path | None:
    preferred = dataset_dir / "splat.splat"
    if preferred.exists() and preferred.is_file():
        return preferred
    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".splat":
            return p
    return None


def _find_latest_preview_image(dataset_dir: Path) -> Path | None:
    candidates: list[Path] = []
    for sub in ("images", "raw"):
        folder = dataset_dir / sub
        if not folder.exists() or not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS:
                candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _refresh_job_preview(job_id: str):
    meta = JOB_META.get(job_id)
    if not meta:
        return

    dataset = meta.get("dataset")
    if not dataset:
        return

    try:
        dataset_dir = _dataset_dir(dataset)
    except HTTPException:
        return

    if not dataset_dir.exists():
        return

    preview_img = _find_latest_preview_image(dataset_dir)
    preview_cloud = dataset_dir / "sparse" / "output_cloud.ply"
    final_splat = _find_root_splat(dataset_dir)

    meta["preview_image_path"] = _to_datasets_url(preview_img) if preview_img else None
    meta["preview_cloud_path"] = _to_datasets_url(preview_cloud) if preview_cloud.exists() else None
    meta["final_splat_path"] = _to_datasets_url(final_splat) if final_splat else None

    image_dir = dataset_dir / "images"
    if image_dir.exists() and image_dir.is_dir():
        meta["processed_image_count"] = len(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_EXTS]
        )
    else:
        meta["processed_image_count"] = 0


def _detect_stage_from_line(line: str) -> str | None:
    normalized = line.lower()
    if (
        "starting video slicing" in normalized
        or "starting preprocessing" in normalized
        or "preprocessing finished" in normalized
    ):
        return "prepare"
    if (
        "starting sfm step" in normalized
        or "sfm extracting features" in normalized
        or "sfm matching features" in normalized
        or "sfm incremental mapping" in normalized
    ):
        return "sfm"
    if (
        "starting gaussian splatting" in normalized
        and "fastergs" in normalized
    ):
        return "fastergs"
    if (
        "starting gaussian splatting" in normalized
        or "produced output:" in normalized
        or ("running:" in normalized and "opensplat" in normalized)
    ):
        return "opensplat"
    if "starting faster-gs sync" in normalized:
        return "prepare"
    return None


def _update_job_stage_from_line(job_id: str, line: str) -> bool:
    stage = _detect_stage_from_line(line)
    if not stage:
        return False
    meta = JOB_META.get(job_id)
    if not meta:
        return False
    if meta.get("stage") == stage:
        return False
    meta["stage"] = stage
    meta["stage_updated_at"] = _utcnow_iso()
    return True


def _cleanup_job_state_if_safe(job_id: str):
    task = JOB_TASKS.get(job_id)
    queue = JOB_QUEUES.get(job_id)
    if task is not None and task.done() and (queue is None or queue.empty()):
        JOB_TASKS.pop(job_id, None)
        JOB_QUEUES.pop(job_id, None)
        logger.info("Cleaned up completed job state job_id=%s", job_id)


def _append_to_job_log_buffer(job_id: str, text: str) -> None:
    buf = JOB_LOG_BUFFERS.get(job_id)
    if buf is None:
        buf = deque(maxlen=JOB_LOG_BUFFER_MAX)
        JOB_LOG_BUFFERS[job_id] = buf
    buf.append(text)


async def _read_process_and_stream(proc: asyncio.subprocess.Process, queue: asyncio.Queue, job_id: str):
    try:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").rstrip("\n")
            meta = JOB_META.get(job_id)
            if meta is not None:
                meta["last_log_line"] = text
                meta["last_log_at"] = _utcnow_iso()
                meta["_line_count"] = int(meta.get("_line_count", 0)) + 1
                if "slurm_job_id" not in meta:
                    match = SLURM_ID_RE.search(text)
                    if match:
                        meta["slurm_job_id"] = match.group(1)
                stage_changed = _update_job_stage_from_line(job_id, text)
                if stage_changed or meta["_line_count"] % 25 == 0:
                    _refresh_job_preview(job_id)
            _append_to_job_log_buffer(job_id, text)
            await queue.put(text)

        await proc.wait()
        meta = JOB_META[job_id]
        if meta.get("status") == "cancelled":
            # Cancel endpoint already stamped status + emitted markers.
            exit_status = "cancelled"
        else:
            exit_status = "completed" if proc.returncode == 0 else "failed"
            meta["status"] = exit_status
            meta["stage"] = exit_status
        meta["exit_code"] = proc.returncode
        meta["finished_at"] = _utcnow_iso()
        _refresh_job_preview(job_id)
        if exit_status != "cancelled":
            done_line = f"<<DONE:{proc.returncode}>>"
            _append_to_job_log_buffer(job_id, done_line)
            await queue.put(done_line)
        JOB_PROCS.pop(job_id, None)
        _cleanup_job_state_if_safe(job_id)
    except Exception as exc:
        JOB_META[job_id]["status"] = "failed"
        JOB_META[job_id]["stage"] = "failed"
        JOB_META[job_id]["finished_at"] = _utcnow_iso()
        JOB_META[job_id]["error"] = str(exc)
        _refresh_job_preview(job_id)
        error_line = f"<<ERROR:{str(exc)}>>"
        _append_to_job_log_buffer(job_id, error_line)
        await queue.put(error_line)
        JOB_PROCS.pop(job_id, None)
        _cleanup_job_state_if_safe(job_id)


@app.get("/api/health")
async def health_check():
    return JSONResponse({"ok": True, "time": _utcnow_iso()})


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...), dataset: str | None = Form(None)):
    """
    Save uploaded video only. Accepts optional form field 'dataset' to set the dataset name.
    Returns dataset name and saved filename.
    """
    try:
        if video.content_type and not video.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="uploaded file must be a video")

        safe_filename = _sanitize_filename(video.filename)
        if dataset:
            dataset_name = _validate_dataset_name(dataset)
        else:
            guessed = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(safe_filename).stem).strip("._")
            dataset_name = _validate_dataset_name(guessed or f"dataset_{uuid.uuid4().hex[:8]}")

        ds_dir = _dataset_dir(dataset_name)
        save_path = ds_dir / "video" / safe_filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = save_path.with_suffix(save_path.suffix + ".part")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        temp_path.replace(save_path)

        logger.info("Uploaded video dataset=%s file=%s", dataset_name, safe_filename)
        return JSONResponse({"dataset": dataset_name, "filename": safe_filename})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"upload failed: {exc}") from exc


@app.post("/api/run")
async def run_pipeline(payload: dict = Body(...)):
    """
    Start the pipeline for an already-uploaded dataset.
    Expected JSON body includes at minimum { "dataset": "<name>" }.
    Returns: { "job_id": "<id>" }
    """
    try:
        params = _parse_run_payload(payload)
        # "Existing dataset" runs reuse backend/datasets/<name>/images/ and
        # skip the frame-extraction step. The dataset must already be
        # preprocessed (enforced here so we fail fast instead of crashing
        # inside the subprocess). Regular runs need a stored video file.
        if params["use_existing_frames"]:
            ds_images = _dataset_dir(params["dataset"]) / "images"
            if not ds_images.is_dir() or not any(ds_images.iterdir()):
                raise HTTPException(
                    status_code=400,
                    detail="use_existing_frames was set but dataset has no preprocessed images on disk",
                )
            video_path = None
        else:
            video_path = _pick_dataset_video_file(params["dataset"])

        # See backend/scripts/pipeline.py for preprocessing + local OpenSplat
        # orchestration. See backend/scripts/fastergs_pipeline.py for HPG
        # dispatch (remote SfM + SLURM training).
        pipeline_path = HERE / "scripts/pipeline.py"
        if not pipeline_path.exists():
            raise HTTPException(status_code=500, detail="pipeline.py not found on server")

        if params["backend"] == "opensplat":
            cmd = [
                sys.executable,
                str(pipeline_path),
                params["dataset"],
                "--iters",
                str(params["iters"]),
                "--only",
                params["only"],
                "--duplicate-threshold",
                str(params["duplicate_threshold"]),
                "--blur-threshold",
                str(params["blur_threshold"]),
                "--fps",
                str(params["fps"]),
                "--downscale",
                str(params["downscale"]),
                "--max-width",
                str(params["max_width"]),
            ]
            if video_path is not None:
                cmd += ["--video", str(video_path)]
            if params["use_existing_frames"]:
                cmd.append("--use-existing-frames")
        else:
            # Faster-GS path. The orchestrator script handles the local
            # preprocess + remote SfM/undistort + remote training + fetch.
            # HPG knobs come from env vars so deployment can override without
            # changing code (e.g. switching train partition from b200 to a100).
            fastergs_pipeline_path = HERE / "scripts/fastergs_pipeline.py"
            if not fastergs_pipeline_path.exists():
                raise HTTPException(status_code=500, detail="fastergs_pipeline.py not found on server")
            cmd = [
                sys.executable,
                str(fastergs_pipeline_path),
                params["dataset"],
                "--iters",
                str(params["iters"]),
                "--duplicate-threshold",
                str(params["duplicate_threshold"]),
                "--blur-threshold",
                str(params["blur_threshold"]),
                "--fps",
                str(params["fps"]),
                "--downscale",
                str(params["downscale"]),
                "--max-width",
                str(params["max_width"]),
                "--remote",
                os.getenv("FASTERGS_REMOTE", "hpg"),
                "--remote-root",
                os.getenv("FASTERGS_REMOTE_ROOT", "/blue/cis4914/joshuabowman/gs_final"),
                "--slurm-account",
                os.getenv("FASTERGS_SLURM_ACCOUNT", "cis4914"),
                "--prepare-partition",
                os.getenv("FASTERGS_PREP_PARTITION", "hpg-default"),
                "--train-partition",
                os.getenv("FASTERGS_TRAIN_PARTITION", "hpg-turin"),
                "--sfm-method",
                params.get("sfm_method", DEFAULT_SFM_METHOD),
            ]
            if video_path is not None:
                cmd += ["--video", str(video_path)]
            if params["use_existing_frames"]:
                cmd.append("--use-existing-frames")
            # Optional experiment flags. Only appended when set so absent
            # flags match the CLI "technique off" default. Seed is always
            # appended when present so run_tag + metrics_summary record it.
            if params.get("seed") is not None:
                cmd += ["--seed", str(params["seed"])]
            if params.get("shortgs_scale_reset_every"):
                cmd += [
                    "--shortgs-scale-reset-every", str(params["shortgs_scale_reset_every"]),
                    "--shortgs-scale-reset-factor", str(params["shortgs_scale_reset_factor"]),
                ]
            if params.get("shortgs_entropy_weight"):
                cmd += ["--shortgs-entropy-weight", str(params["shortgs_entropy_weight"])]
            if params.get("shortgs_progressive_resolution"):
                cmd += ["--shortgs-progressive-resolution", params["shortgs_progressive_resolution"]]

        job_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        JOB_QUEUES[job_id] = queue
        JOB_META[job_id] = {
            "job_id": job_id,
            "dataset": params["dataset"],
            "backend": params["backend"],
            "status": "running",
            "stage": "queued",
            "created_at": _utcnow_iso(),
            "command": cmd,
            "preview_image_path": None,
            "preview_cloud_path": None,
            "final_splat_path": None,
            "processed_image_count": 0,
        }

        logger.info(
            "Starting pipeline job_id=%s dataset=%s backend=%s only=%s iters=%s",
            job_id,
            params["dataset"],
            params["backend"],
            params["only"],
            params["iters"],
        )

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(HERE),
        )
        task = asyncio.create_task(_read_process_and_stream(proc, queue, job_id))
        JOB_TASKS[job_id] = task
        JOB_PROCS[job_id] = proc

        return JSONResponse({"job_id": job_id})
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start pipeline")
        raise HTTPException(status_code=500, detail=f"failed to start pipeline: {exc}") from exc


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    meta = JOB_META.get(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="unknown job")
    public_meta = dict(meta)
    public_meta.pop("_line_count", None)
    return JSONResponse(public_meta)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    meta = JOB_META.get(job_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if meta.get("status") in {"completed", "failed", "cancelled"}:
        return JSONResponse({"job_id": job_id, "status": meta.get("status")})

    proc = JOB_PROCS.get(job_id)
    slurm_job_id = meta.get("slurm_job_id")
    queue = JOB_QUEUES.get(job_id)

    # Stamp cancelled first so the reader task doesn't later overwrite status.
    meta["status"] = "cancelled"
    meta["stage"] = "cancelled"
    meta["finished_at"] = _utcnow_iso()

    marker = "<<CANCEL: job cancelled by user>>"
    _append_to_job_log_buffer(job_id, marker)
    if queue is not None:
        await queue.put(marker)

    # Best-effort remote scancel. The orchestrator is still alive at this
    # point so it cleans up once it sees the SLURM job exit.
    if slurm_job_id:
        try:
            await asyncio.create_subprocess_exec(
                "ssh", "hpg", f"scancel {slurm_job_id}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            note = f"<<CANCEL: requested scancel {slurm_job_id} on HPG>>"
            _append_to_job_log_buffer(job_id, note)
            if queue is not None:
                await queue.put(note)
        except Exception as exc:
            logger.warning("scancel failed job_id=%s slurm=%s: %s", job_id, slurm_job_id, exc)

    # Terminate the local orchestrator subprocess. SIGTERM first, then
    # kill if it doesn't exit in a couple of seconds.
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
        except Exception as exc:
            logger.warning("terminate failed job_id=%s: %s", job_id, exc)

    done_line = "<<DONE:-1>>"
    _append_to_job_log_buffer(job_id, done_line)
    if queue is not None:
        await queue.put(done_line)

    return JSONResponse({"job_id": job_id, "status": "cancelled", "slurm_job_id": slurm_job_id})


@app.get("/api/jobs-active")
async def get_latest_active_job():
    """
    Return the most recently created active job (status running or queued), if any.
    """
    active = [m for m in JOB_META.values() if m.get("status") in {"running", "queued"}]
    if not active:
        raise HTTPException(status_code=404, detail="no active job")

    latest = max(active, key=lambda m: m.get("created_at", ""))
    public_meta = dict(latest)
    public_meta.pop("_line_count", None)
    return JSONResponse(public_meta)


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(job_id: str):
    """
    Replay the rolling log buffer for a job. Used by the frontend on
    mount to rehydrate progress / stage state after the user navigates
    away and back. Returns [] if the job never had any output or was
    already cleaned up.
    """
    meta = JOB_META.get(job_id)
    if meta is None and job_id not in JOB_LOG_BUFFERS:
        raise HTTPException(status_code=404, detail="unknown job")
    buf = JOB_LOG_BUFFERS.get(job_id)
    lines = list(buf) if buf is not None else []
    return JSONResponse({"job_id": job_id, "lines": lines, "truncated": len(lines) >= JOB_LOG_BUFFER_MAX})


@app.websocket("/api/ws/{job_id}")
async def websocket_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint the frontend connects to for realtime logs.
    Sends each stdout line as a text message. When the process finishes, sends <<DONE:exitcode>>.
    """
    await websocket.accept()
    queue = JOB_QUEUES.get(job_id)
    if queue is None:
        await websocket.send_text(f"<<ERROR:unknown job {job_id}>>")
        await websocket.close()
        return

    try:
        while True:
            line = await queue.get()
            try:
                await websocket.send_text(line)
            except Exception:
                break
            if line.startswith("<<DONE:") or line.startswith("<<ERROR:"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        _cleanup_job_state_if_safe(job_id)


@app.get("/api/datasets")
async def list_datasets():
    """
    Return JSON list of datasets found under backend/datasets. Each
    entry carries enough status for LiveDemos to decide whether the
    dataset is usable in "existing dataset" mode without re-uploading
    a video:

      name            folder name under backend/datasets/
      has_splat       final .splat file sitting at the dataset root
      splat_path      URL path to that splat (when present)
      has_images      preprocessed images/ exists with >= 8 files
                      (same floor the trainer enforces)
      has_raw_video   a video file was previously uploaded (lets the
                      frontend know a re-extract is still an option)
      has_sfm         a SfM fingerprint (local or remote) is on disk,
                      so a rerun with matching settings will skip SfM
      run_count       number of training runs we've fetched back for
                      this dataset (counts metrics/ subdirs)
    """
    out = []
    ds_root = datasets_dir.resolve()
    for d in sorted(ds_root.iterdir()):
        if not d.is_dir() or d.parent != ds_root:
            continue

        has_images = False
        try:
            images_dir = d / "images"
            if images_dir.is_dir():
                # Match the preflight / trainer threshold so the UI
                # doesn't advertise datasets the training step would
                # immediately reject.
                image_count = sum(1 for p in images_dir.iterdir() if p.is_file())
                has_images = image_count >= 8
        except OSError:
            has_images = False

        has_raw_video = False
        try:
            video_dir = d / "video"
            if video_dir.is_dir():
                has_raw_video = any(
                    p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTS
                    for p in video_dir.iterdir()
                )
        except OSError:
            has_raw_video = False

        has_sfm = any(
            (d / name).is_file()
            for name in ("remote_sfm_fingerprint.json", "sfm_fingerprint.json", "prep_fingerprint.json")
        )

        run_count = 0
        metrics_dir = d / "metrics"
        if metrics_dir.is_dir():
            try:
                run_count = sum(1 for p in metrics_dir.iterdir() if p.is_dir())
            except OSError:
                run_count = 0

        entry = {
            "name": d.name,
            "has_splat": False,
            "splat_path": None,
            "has_images": has_images,
            "has_raw_video": has_raw_video,
            "has_sfm": has_sfm,
            "run_count": run_count,
        }
        preferred = _find_root_splat(d)
        if preferred:
            entry["has_splat"] = True
            entry["splat_path"] = f"/datasets/{d.name}/{preferred.name}"
        out.append(entry)
    return JSONResponse(out)


@app.get("/api/datasets/{name}/prep-status")
async def get_dataset_prep_status(name: str):
    """
    Tell the frontend whether this dataset has cached preprocess + SfM
    output that a new run with matching settings would reuse. Used to
    show a "cache hit = faster rerun" hint in LiveDemos.

    Returns:
      {
        has_prep_fingerprint: bool,
        has_sfm_fingerprint: bool,
        has_remote_sfm_fingerprint: bool,
        fingerprint: <fields> | null,
      }
    """
    d = _dataset_dir(name)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")

    def _read(path: Path):
        try:
            if path.is_file():
                return json.loads(path.read_text())
        except Exception:
            return None
        return None

    prep = _read(d / "prep_fingerprint.json")
    sfm = _read(d / "sfm_fingerprint.json")
    remote = _read(d / "remote_sfm_fingerprint.json")
    # Expose the "prep" fingerprint if present; remote takes precedence
    # since that's what matters for the Faster-GS path most users hit.
    primary = remote or prep or sfm
    preview = None
    if primary:
        preview = {
            k: primary.get(k)
            for k in ("fps", "downscale", "blur_threshold", "duplicate_threshold", "max_width")
        }
    return JSONResponse({
        "has_prep_fingerprint": prep is not None,
        "has_sfm_fingerprint": sfm is not None,
        "has_remote_sfm_fingerprint": remote is not None,
        "fingerprint": preview,
    })


@app.get("/api/datasets/{name}/runs")
async def list_dataset_runs(name: str):
    """
    Enumerate every training run for a dataset so the Gallery + viewer
    dropdowns can group runs under their parent dataset. A "run" here is
    one of:
      - A metrics/<run_tag>/ subdir with a matching .splat fetched back
        to backend/hipergator/gs_final/<run_tag>.splat
      - A dataset-root splat (e.g. banana/banana.splat) that isn't a
        copy of any HPG run (treated as a "showcase" single-run entry
        so pre-committed demo scenes still appear in the picker).
    Runs are sorted oldest-first and assigned monotonic run_number
    values (1, 2, 3...) within their dataset so the UI can display
    "Run N" labels consistently across Gallery, Reports, and LiveDemos.
    """
    d = _dataset_dir(name)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")

    runs = []
    metrics_root = d / "metrics"
    hpg_splat_tags = set()

    if metrics_root.is_dir():
        for entry in metrics_root.iterdir():
            if not entry.is_dir():
                continue
            run_tag = entry.name
            summary = None
            summary_path = entry / "metrics_summary.json"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text())
                except Exception:
                    summary = None
            # Splat is expected under backend/hipergator/gs_final/<tag>.splat.
            # If it's missing (fetch failed or got cleaned up) we still list
            # the run so the user can see its metrics, just with no splat to
            # view.
            splat_filename = f"{run_tag}.splat"
            splat_path = None
            # Real training runs land under backend/hipergator/gs_final/
            # per fastergs_pipeline.fetch_dir. The legacy showcase splats
            # sit at backend/hipergator/ root. Check the gs_final subdir
            # first so we route to the per-run file, then fall back to
            # the root only if someone manually placed it there.
            gs_final_splat = hpg_dir / "gs_final" / splat_filename
            root_splat = hpg_dir / splat_filename
            if gs_final_splat.is_file():
                splat_path = f"/api/hpg/gs_final/{splat_filename}/splat"
                hpg_splat_tags.add(splat_filename)
            elif root_splat.is_file():
                splat_path = f"/api/hpg/{splat_filename}/splat"
                hpg_splat_tags.add(splat_filename)
            created_at_epoch = None
            if isinstance(summary, dict) and summary.get("created_at_epoch") is not None:
                try:
                    created_at_epoch = float(summary["created_at_epoch"])
                except (TypeError, ValueError):
                    created_at_epoch = None
            if created_at_epoch is None:
                # Fallback to the metrics dir's own mtime if the summary
                # didn't carry a timestamp. Keeps run ordering stable.
                try:
                    created_at_epoch = entry.stat().st_mtime
                except OSError:
                    created_at_epoch = 0.0
            runs.append({
                "run_tag": run_tag,
                "splat_path": splat_path,
                "splat_filename": splat_filename if splat_path else None,
                "metrics_summary": summary,
                "created_at_epoch": created_at_epoch,
                "is_showcase": False,
            })

    # Sort oldest-first so run_number increments chronologically.
    runs.sort(key=lambda r: r["created_at_epoch"] or 0.0)
    for idx, r in enumerate(runs, start=1):
        r["run_number"] = idx
        r["display_label"] = f"Run {idx}"

    # Dataset-root splat (e.g. banana/banana.splat) that doesn't already
    # correspond to one of the HPG runs above. Shown as a single
    # "showcase" entry so pre-built demo scenes still appear in the
    # picker. If the root splat IS a copy of a training run, we don't
    # double-list it.
    preferred = _find_root_splat(d)
    if preferred and preferred.name not in hpg_splat_tags:
        runs.append({
            "run_tag": None,
            "run_number": None,
            "display_label": "Showcase",
            "splat_path": f"/api/datasets/{name}/splat",
            "splat_filename": preferred.name,
            "metrics_summary": None,
            "created_at_epoch": None,
            "is_showcase": True,
        })

    return JSONResponse(runs)


@app.get("/api/datasets/{name}/splat")
async def get_dataset_splat(name: str):
    """
    Return the first .splat file found at the dataset root as a FileResponse.
    """
    d = _dataset_dir(name)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")
    preferred = _find_root_splat(d)
    if preferred:
        return FileResponse(path=str(preferred), media_type="application/octet-stream", filename=preferred.name)
    raise HTTPException(status_code=404, detail="splat not found")


@app.get("/api/hpg/splats")
async def list_hpg_splats():
    """
    Return JSON list of .splat files found under backend/hpg.
    Each entry: { name, filename, api_path } where api_path is /api/hpg/<filename>/splat
    """
    # This endpoint powers the Gallery page. It lists any .splat in
    # backend/hipergator/ (where fastergs_pipeline.py publishes its outputs,
    # plus any showcase files teammates have committed). We keep it flat (not
    # nested by dataset) so the gallery picker is just a flat list.
    out = []
    if not hpg_dir.exists() or not hpg_dir.is_dir():
        return JSONResponse(out)

    for p in sorted(hpg_dir.iterdir()):
        if p.is_file() and p.suffix.lower() == ".splat":
            out.append({"name": p.stem, "filename": p.name, "api_path": f"/api/hpg/{p.name}/splat"})
    return JSONResponse(out)


@app.get("/api/hpg/{filename}/splat")
async def get_hpg_splat(filename: str):
    """
    Return the requested .splat file from backend/hpg as a FileResponse.
    Filename is sanitized to avoid path traversal.
    """
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".splat"):
        raise HTTPException(status_code=400, detail="invalid splat filename")

    p = (hpg_dir / safe_name).resolve()
    if p.parent != hpg_dir.resolve() or not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="splat not found")
    return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)


@app.get("/api/hpg/gs_final/{filename}/splat")
async def get_hpg_gs_final_splat(filename: str):
    """
    Serve a per-run splat fetched back from HPG. Runs land under
    backend/hipergator/gs_final/<run_tag>.splat via fastergs_pipeline.
    Kept as its own route (rather than folded into /api/hpg/{name}/splat)
    so the path prefix reflects the directory and matches the layout
    used by the /runs endpoint.
    """
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.lower().endswith(".splat"):
        raise HTTPException(status_code=400, detail="invalid splat filename")
    gs_final_root = (hpg_dir / "gs_final").resolve()
    p = (gs_final_root / safe_name).resolve()
    if p.parent != gs_final_root or not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="splat not found")
    return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)


# --- Metrics endpoints ---
# The frontend Reports page reads these. Every route tolerates missing files
# (returns 404) rather than crashing, so a dataset with no runs yet is fine.

ALLOWED_PLOT_NAMES = {
    "psnr", "ssim", "lpips", "loss", "num_gaussians", "splats_per_frame", "wall_seconds",
    # Shorter-Splatting validation plots - time-domain curves + PLY histograms
    "psnr_vs_wall_seconds", "scale_hist", "opacity_hist",
}


def _safe_run_tag(run_tag: str) -> str:
    # Run tags look like "can_train_20260414_1230". Accept the same charset
    # as dataset names so we can reuse the pattern and keep path traversal
    # closed off.
    if not DATASET_RE.fullmatch(run_tag):
        raise HTTPException(status_code=400, detail="invalid run_tag")
    if Path(run_tag).name != run_tag:
        raise HTTPException(status_code=400, detail="invalid run_tag")
    return run_tag


def _metrics_root_for(dataset_name: str) -> Path:
    ds_dir = _dataset_dir(dataset_name)
    return (ds_dir / "metrics").resolve()


def _run_metrics_dir(dataset_name: str, run_tag: str) -> Path:
    safe_tag = _safe_run_tag(run_tag)
    root = _metrics_root_for(dataset_name)
    p = (root / safe_tag).resolve()
    if p.parent != root:
        raise HTTPException(status_code=400, detail="invalid metrics path")
    if not p.is_dir():
        raise HTTPException(status_code=404, detail="metrics run not found")
    return p


@app.get("/api/datasets/{dataset_name}/metrics")
async def list_dataset_metrics(dataset_name: str):
    """List every run tag under this dataset that has a metrics_summary.json."""
    try:
        root = _metrics_root_for(dataset_name)
    except HTTPException:
        return JSONResponse([])

    out = []
    if not root.is_dir():
        return JSONResponse(out)
    for entry in sorted(root.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir():
            continue
        summary_path = entry / "metrics_summary.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                summary = None
        out.append({
            "run_tag": entry.name,
            "summary": summary,
            "has_series": (entry / "metrics.jsonl").is_file(),
        })
    return JSONResponse(out)


@app.get("/api/datasets/{dataset_name}/metrics/{run_tag}/summary")
async def get_metrics_summary(dataset_name: str, run_tag: str):
    d = _run_metrics_dir(dataset_name, run_tag)
    p = d / "metrics_summary.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="metrics_summary.json not found")
    try:
        return JSONResponse(json.loads(p.read_text()))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"corrupt summary: {exc}") from exc


@app.get("/api/datasets/{dataset_name}/metrics/{run_tag}/series")
async def get_metrics_series(dataset_name: str, run_tag: str):
    d = _run_metrics_dir(dataset_name, run_tag)
    p = d / "metrics.jsonl"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="metrics.jsonl not found")
    records = []
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"corrupt series: {exc}") from exc
    return JSONResponse(records)


@app.get("/api/datasets/{dataset_name}/metrics/{run_tag}/plot/{plot_name}.png")
async def get_metrics_plot(dataset_name: str, run_tag: str, plot_name: str):
    if plot_name not in ALLOWED_PLOT_NAMES:
        raise HTTPException(status_code=400, detail="unknown plot name")
    d = _run_metrics_dir(dataset_name, run_tag)
    p = d / f"{plot_name}.png"
    if not p.is_file():
        raise HTTPException(status_code=404, detail="plot not found")
    return FileResponse(path=str(p), media_type="image/png", filename=p.name)


@app.get("/api/datasets/{dataset_name}/metrics/{run_tag}/download")
async def download_metrics_bundle(dataset_name: str, run_tag: str):
    """Zip of summary + series + every PNG for this run, for archival."""
    d = _run_metrics_dir(dataset_name, run_tag)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in d.iterdir():
            if entry.is_file() and entry.suffix.lower() in {".json", ".jsonl", ".png", ".log"}:
                zf.write(entry, arcname=entry.name)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{run_tag}_metrics.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
