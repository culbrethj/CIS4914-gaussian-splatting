from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import sys
import subprocess

app = FastAPI()

@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        dataset_name = Path(video.filename).stem
        HERE = Path(__file__).resolve().parent
        save_path = HERE / "datasets" / dataset_name / "video" / video.filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        pipeline_path = HERE / "pipeline.py"

        cmd = [sys.executable,
            str(pipeline_path),
            dataset_name,
            "--video", str(save_path),
            "--iters", "1000",
            "--only", "all"]
        subprocess.run(cmd, check=True, cwd=str(HERE))

        return {"message": "Upload and pipeline successful", "filename": video.filename}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed w/ exit code {e.returncode}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
