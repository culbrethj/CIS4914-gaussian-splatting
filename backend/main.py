from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
DATASET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_PIPELINE_STEPS = {"prepare", "sfm", "opensplat", "all"}
ALLOWED_BACKENDS = {"opensplat", "fastergs"}
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

    return {
        "dataset": dataset_name,
        "backend": backend,
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
    }


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
                stage_changed = _update_job_stage_from_line(job_id, text)
                if stage_changed or meta["_line_count"] % 25 == 0:
                    _refresh_job_preview(job_id)
            await queue.put(text)

        await proc.wait()
        JOB_META[job_id]["status"] = "completed" if proc.returncode == 0 else "failed"
        JOB_META[job_id]["exit_code"] = proc.returncode
        JOB_META[job_id]["finished_at"] = _utcnow_iso()
        JOB_META[job_id]["stage"] = "completed" if proc.returncode == 0 else "failed"
        _refresh_job_preview(job_id)
        await queue.put(f"<<DONE:{proc.returncode}>>")
        _cleanup_job_state_if_safe(job_id)
    except Exception as exc:
        JOB_META[job_id]["status"] = "failed"
        JOB_META[job_id]["stage"] = "failed"
        JOB_META[job_id]["finished_at"] = _utcnow_iso()
        JOB_META[job_id]["error"] = str(exc)
        _refresh_job_preview(job_id)
        await queue.put(f"<<ERROR:{str(exc)}>>")
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
        video_path = _pick_dataset_video_file(params["dataset"])

        pipeline_path = HERE / "scripts/pipeline.py"
        if not pipeline_path.exists():
            raise HTTPException(status_code=500, detail="pipeline.py not found on server")

        if params["backend"] == "opensplat":
            cmd = [
                sys.executable,
                str(pipeline_path),
                params["dataset"],
                "--video",
                str(video_path),
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
        else:
            fastergs_pipeline_path = HERE / "scripts/fastergs_pipeline.py"
            if not fastergs_pipeline_path.exists():
                raise HTTPException(status_code=500, detail="fastergs_pipeline.py not found on server")
            cmd = [
                sys.executable,
                str(fastergs_pipeline_path),
                params["dataset"],
                "--video",
                str(video_path),
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
                os.getenv("FASTERGS_TRAIN_PARTITION", "hpg-b200"),
            ]

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
    Return JSON list of datasets found under backend/datasets.
    For each dataset return its name and whether a .splat file exists at the dataset root.
    """
    out = []
    ds_root = datasets_dir.resolve()
    for d in sorted(ds_root.iterdir()):
        if not d.is_dir() or d.parent != ds_root:
            continue
        entry = {"name": d.name, "has_splat": False, "splat_path": None}
        preferred = _find_root_splat(d)
        if preferred:
            entry["has_splat"] = True
            entry["splat_path"] = f"/datasets/{d.name}/{preferred.name}"
        out.append(entry)
    return JSONResponse(out)


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
