from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

app = FastAPI()

@app.post("/api/upload")
async def upload_video(video: UploadFile = File(...)):
    save_path = Path("../data/videos") / video.filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return {"message": "Upload successful", "filename": video.filename}