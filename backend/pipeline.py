import sys
import subprocess
import shutil
from pathlib import Path
import argparse
import logging
from sfm import sfm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
HERE = Path(__file__).resolve().parent

def run_command(cmd, cwd=HERE):
    logging.info("Running: %s (cwd=%s)", " ".join(cmd), cwd)
    # subprocess.run is synchronous and will not return until the process finishes.
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_sfm(dataset):
    sfm(f"datasets/{dataset}/images", f"datasets/{dataset}/sparse")


def run_opensplat(dataset, num_iters):
    opensplat_path = HERE / "opensplat"

    if opensplat_path.exists() and opensplat_path.is_file():
        exe = str(opensplat_path)

        dataset_path = f"datasets/{dataset}"
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

def main(dataset, iters):
    parser = argparse.ArgumentParser(description="Orchestrate SfM and Gaussian Splatting")
    parser.add_argument("dataset", help="Dataset name inside datasets/")
    parser.add_argument("iters", type=int, help="Number of OpenSplat iterations")
    parser.add_argument(
        "--only",
        choices=["sfm", "opensplat", "all"],
        default="all",
        help="Which step to run (default: all)",
    )
    args = parser.parse_args()

    try:
        if args.only in ("sfm", "all"):
            logging.info("Starting SfM step")
            run_sfm(dataset)
            logging.info("SfM step finished successfully")

        if args.only in ("opensplat", "all"):
            logging.info("Starting Gaussian Splatting (opensplat)")
            run_opensplat(dataset, iters)
            logging.info("Gaussian Splatting finished successfully")

    except subprocess.CalledProcessError as e:
        logging.error("Command failed with exit code %s", e.returncode)
        sys.exit(e.returncode)
    except Exception as e:
        logging.exception("Error during run: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    # TODO verify argv[1] is a valid dataset
    # TODO verify argv[2] is a valid num of iters
    main(sys.argv[1], sys.argv[2])
