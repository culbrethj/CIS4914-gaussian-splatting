from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import sys
import asyncio
import uuid
import json
import os

HERE = Path(__file__).resolve().parent

app = FastAPI()

# ensure an hpg dir exists (contains splat files on Hipergator)
hpg_dir = HERE / "hpg"
hpg_dir.mkdir(parents=True, exist_ok=True)

# mount datasets directory so frontend can fetch files directly:
datasets_dir = HERE / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)
app.mount("/datasets", StaticFiles(directory=str(datasets_dir)), name="datasets")

# in-memory maps of job queues and process tasks
JOB_QUEUES: dict[str, asyncio.Queue] = {}
JOB_TASKS: dict[str, asyncio.Task] = {}


async def _read_process_and_stream(proc: asyncio.subprocess.Process, queue: asyncio.Queue, job_id: str):
    try:
        # read stdout lines until EOF
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode(errors="ignore").rstrip("\n")
            await queue.put(text)
        await proc.wait()
        await queue.put(f"<<DONE:{proc.returncode}>>")
    except Exception as e:
        await queue.put(f"<<ERROR:{str(e)}>>")
    finally:
        # leave queue to be drained by client; do not delete immediately
        pass


@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    """
    Save uploaded video and start the pipeline asynchronously.
    Returns a job_id which the frontend can use to open a WebSocket to receive logs:
      ws://<host>/api/ws/{job_id}
    """
    try:
        dataset_name = Path(video.filename).stem
        save_path = datasets_dir / dataset_name / "video" / video.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # write file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        pipeline_path = HERE / "pipeline.py"
        if not pipeline_path.exists():
            raise HTTPException(status_code=500, detail="pipeline.py not found on server")

        # build command
        cmd = [sys.executable,
               str(pipeline_path),
               dataset_name,
               "--video", str(save_path),
               "--iters", "1000",
               "--only", "all"]

        # create job id and queue
        job_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        JOB_QUEUES[job_id] = queue

        # launch subprocess asynchronously and stream stdout into the queue
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                # client disconnected
                break
            if line.startswith("<<DONE:") or line.startswith("<<ERROR:"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        # cleanup: remove job queue and task if finished
        JOB_QUEUES.pop(job_id, None)
        task = JOB_TASKS.pop(job_id, None)
        if task and task.done():
            # nothing to do
            pass


@app.get("/api/datasets")
async def list_datasets():
    """
    Return JSON list of datasets found under backend/datasets.
    For each dataset return its name and whether a .splat file exists at the dataset root.
    If present, return the splat_path (served from /datasets/...), otherwise splat_path is null.
    """
    out = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        entry = {"name": d.name, "has_splat": False, "splat_path": None}
        # look for .splat files at dataset root only (do not search sparse/ or other subfolders)
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() == ".splat":
                entry["has_splat"] = True
                entry["splat_path"] = f"/datasets/{d.name}/{p.name}"
                break
        out.append(entry)
    return JSONResponse(out)


@app.get("/api/datasets/{name}/splat")
async def get_dataset_splat(name: str):
    """
    Return the first .splat file found at the dataset root as a FileResponse.
    This routes splat downloads through the backend so the frontend can fetch
    via the same /api host (avoids dev-server proxy/CORS problems).
    """
    d = datasets_dir / name
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() == ".splat":
            return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)
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
            out.append({
                "name": p.stem,
                "filename": p.name,
                "api_path": f"/api/hpg/{p.name}/splat"
            })
    return JSONResponse(out)


@app.get("/api/hpg/{filename}/splat")
async def get_hpg_splat(filename: str):
    """
    Return the requested .splat file from backend/hpg as a FileResponse.
    Filename is sanitized to avoid path traversal.
    """
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".splat"):
        raise HTTPException(status_code=400, detail="invalid splat filename")
    p = hpg_dir / safe_name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="splat not found")
    return FileResponse(path=str(p), media_type="application/octet-stream", filename=p.name)
