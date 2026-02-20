from pathlib import Path
from preprocessor import preprocessor
import cv2

DUPLICATE_THRESHOLD = 3.0
BLUR_THRESHOLD = 50

def video_slicer(video_path, output_dir, img_format):
    output_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    
    i = 0
    while True:
        cont, frame = video.read()
        if not cont:
            break
        
        filename = f"frame_{i:06d}.{img_format}"
        output_path = output_dir / filename

        cv2.imwrite(output_path, frame)
        i += 1

    video.release()
    print(f"{i+1} frames saved to {output_dir}")
    return

if __name__ == "__main__":
    # get root directory (parent 1 ->backend parent 2 ->CIS4914-Gaussian-Splatting)
    root = Path(__file__).parent.parent.resolve()

    # run with test video sample-10s.mp4
    video_path = root / "data" / "videos" / "sample-10s.mp4"
    # create output folder w same name for output
    raw_path = root / "data" / "frames" / video_path.stem / "raw"

    video_slicer(video_path, raw_path, "jpg")

    # create dir of processed frames
    processed_path = root / "data" / "frames" / video_path.stem / "proc"
    preprocessor(raw_path, processed_path, DUPLICATE_THRESHOLD, BLUR_THRESHOLD)