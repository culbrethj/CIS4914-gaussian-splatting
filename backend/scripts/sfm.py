from __future__ import annotations

import logging
import sys
from pathlib import Path

import pycolmap

logger = logging.getLogger("gaussian.sfm")


def sfm(in_path, out_path):
    dataset_path = Path(in_path)
    output_path = Path(out_path)

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Input image directory not found: {dataset_path}")

    image_files = [p for p in dataset_path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(image_files) < 8:
        raise ValueError(f"Need at least 8 images for SfM, found {len(image_files)}")

    output_path.mkdir(parents=True, exist_ok=True)
    database_path = output_path / "database.db"

    logger.info("SfM extracting features from %s", dataset_path)
    pycolmap.extract_features(database_path, dataset_path)

    logger.info("SfM matching features")
    pycolmap.match_sequential(database_path)

    logger.info("SfM incremental mapping")
    reconstructions = pycolmap.incremental_mapping(database_path, dataset_path, output_path)

    if reconstructions:
        reconstructions[0].write(output_path)
        reconstructions[0].export_PLY(output_path / "output_cloud.ply")
        logger.info("SfM reconstruction successful: %s", output_path)
        return

    raise RuntimeError("No reconstruction could be created")


if __name__ == "__main__":
    sfm(sys.argv[1], sys.argv[2])
