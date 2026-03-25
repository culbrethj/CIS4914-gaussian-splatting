import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import logging
from sfm import sfm
from frame_slicer import video_slicer
from preprocessor import preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
HERE = Path(__file__).resolve().parent

DUPLICATE_THRESHOLD = 3.0
BLUR_THRESHOLD = 50

def run_command(cmd, cwd=HERE):
    logging.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    # subprocess.run is synchronous and will not return until the process finishes.
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_sfm(dataset_path):
    sfm(f"{dataset_path}/images", f"{dataset_path}/sparse")


def run_opensplat(dataset_path, num_iters):
    exe = ""
    if sys.platform == "win32":
        exe = ".exe"
    opensplat_path = HERE / f"opensplat{exe}"

    if opensplat_path.exists() and opensplat_path.is_file():
        exe = str(opensplat_path)

        output_ply = f"{dataset_path}/splat.ply"

        cmd = [
            exe,
            dataset_path,
            "-o", output_ply,
            "-n", str(num_iters)
        ]

        run_command(cmd)

    else:
        raise FileNotFoundError("opensplat not found (expected ./opensplat)")
    
def run_prepare(dataset_path, video_path, img_format):
    raw_path = dataset_path / "raw"
    images_path = dataset_path / "images"

    logging.info("Starting video slicing")
    video_slicer(video_path, raw_path, img_format)
    logging.info("Video slicing finished")

    logging.info("Starting preprocessing")
    preprocessor(raw_path, images_path, DUPLICATE_THRESHOLD, BLUR_THRESHOLD)
    logging.info("Preprocessing finished")

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SfM and Gaussian Splatting")
    parser.add_argument("dataset", help="Dataset name inside datasets/")
    parser.add_argument("--iters", type=int, default = 1000, help="Number of OpenSplat iterations")
    parser.add_argument("--video", help="Path to input video")
    parser.add_argument("--img_format", default="jpg", help="Frame image format")
    parser.add_argument(
        "--only",
        choices=["prepare", "sfm", "opensplat", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    dataset_path = HERE / "datasets" / args.dataset
    dataset_path.mkdir(parents=True, exist_ok=True)

    try:
        if args.only in ("prepare", "all"):
            if not args.video:
                raise ValueError("--video is a required parameter")
            video_path = Path(args.video).resolve()
            if not video_path.exists():
                raise FileNotFoundError(f"Video not found {video_path}")
            run_prepare(dataset_path, video_path, args.img_format)
        if args.only in ("sfm", "all"):
            logging.info("Starting SfM step")
            run_sfm(dataset_path)
            logging.info("SfM step finished successfully")

        if args.only in ("opensplat", "all"):
            logging.info("Starting Gaussian Splatting (opensplat)")
            run_opensplat(dataset_path, args.iters)
            logging.info("Gaussian Splatting finished successfully")

    except subprocess.CalledProcessError as e:
        logging.error("Command failed with exit code %s", e.returncode)
        sys.exit(e.returncode)
    except Exception as e:
        logging.exception("Error during run: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    # TODO verify dataset content before running
    main()
