from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

# Minimal COLMAP camera-model table (matches COLMAP read_write_model and Faster-GS fork loader).
CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
SUPPORTED_FASTERGS_COLMAP_MODELS = {"SIMPLE_PINHOLE", "PINHOLE"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _read_next(fid, num_bytes: int, fmt: str):
    return struct.unpack("<" + fmt, fid.read(num_bytes))


def read_camera_models(cameras_bin: Path) -> list[str]:
    models: list[str] = []
    with cameras_bin.open("rb") as fid:
        num_cameras = _read_next(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next(fid, 24, "iiQQ")
            _ = camera_id, width, height
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            _read_next(fid, 8 * num_params, "d" * num_params)
            models.append(model_name)
    return models


def read_image_names(images_bin: Path) -> list[str]:
    names: list[str] = []
    with images_bin.open("rb") as fid:
        num_images = _read_next(fid, 8, "Q")[0]
        for _ in range(num_images):
            _read_next(fid, 4, "i")  # image_id
            _read_next(fid, 8 * 7, "ddddddd")  # qvec(4), tvec(3)
            _read_next(fid, 4, "i")  # camera_id

            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b"":
                    raise ValueError("Unexpected EOF while reading COLMAP image name from images.bin")
                if c == b"\x00":
                    break
                name_bytes.extend(c)
            names.append(name_bytes.decode("utf-8", errors="replace"))

            num_points2d = _read_next(fid, 8, "Q")[0]
            # points2D: x(double), y(double), point3D_id(long long) => 24 bytes per point
            fid.seek(24 * num_points2d, 1)
    return names


def inspect_dataset(dataset_dir: Path) -> dict:
    images_dir = dataset_dir / "images"
    sparse0_dir = dataset_dir / "sparse" / "0"
    cameras_bin = sparse0_dir / "cameras.bin"
    images_bin = sparse0_dir / "images.bin"
    points3d_bin = sparse0_dir / "points3D.bin"
    points3d_txt = sparse0_dir / "points3D.txt"

    image_count = 0
    if images_dir.exists() and images_dir.is_dir():
        image_count = len([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    camera_models: list[str] = []
    if cameras_bin.exists():
        camera_models = read_camera_models(cameras_bin)
    model_counts = dict(Counter(camera_models))
    unique_models = sorted(model_counts.keys())
    unsupported_models = sorted([m for m in unique_models if m not in SUPPORTED_FASTERGS_COLMAP_MODELS])

    referenced_image_names: list[str] = []
    missing_referenced_images: list[str] = []
    if images_bin.exists() and images_dir.exists() and images_dir.is_dir():
        referenced_image_names = read_image_names(images_bin)
        existing_names = {p.name for p in images_dir.iterdir() if p.is_file()}
        missing_referenced_images = sorted([n for n in referenced_image_names if n not in existing_names])

    checks = {
        "images_dir_exists": images_dir.exists() and images_dir.is_dir(),
        "sparse0_dir_exists": sparse0_dir.exists() and sparse0_dir.is_dir(),
        "cameras_bin_exists": cameras_bin.exists() and cameras_bin.is_file(),
        "images_bin_exists": images_bin.exists() and images_bin.is_file(),
        "points3D_bin_or_txt_exists": (
            (points3d_bin.exists() and points3d_bin.is_file())
            or (points3d_txt.exists() and points3d_txt.is_file())
        ),
        "enough_images_for_training": image_count >= 8,
        "camera_models_supported_by_fastergs_inria": len(unsupported_models) == 0,
        "all_referenced_images_exist": len(missing_referenced_images) == 0,
    }

    compatible = all(checks.values())

    warnings: list[str] = []
    if image_count < 8:
        warnings.append("Very low image count (<8). Training is unlikely to be stable.")
    if unsupported_models:
        warnings.append(
            "Unsupported COLMAP camera model(s) for Faster-GS Inria fork: "
            + ", ".join(unsupported_models)
            + ". Use undistorted SIMPLE_PINHOLE/PINHOLE cameras."
        )
    if missing_referenced_images:
        warnings.append(
            "Missing image files referenced by COLMAP model: "
            f"{len(missing_referenced_images)} missing (showing first 5): "
            + ", ".join(missing_referenced_images[:5])
        )

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "image_count": image_count,
        "referenced_image_count": len(referenced_image_names),
        "missing_referenced_image_count": len(missing_referenced_images),
        "missing_referenced_images_sample": missing_referenced_images[:20],
        "camera_model_counts": model_counts,
        "unsupported_camera_models": unsupported_models,
        "checks": checks,
        "compatible_with_fastergs_inria_colmap_loader": compatible,
        "warnings": warnings,
    }


def print_human_report(report: dict):
    print(f"Dataset: {report['dataset_dir']}")
    print(f"Images: {report['image_count']}")
    print(f"Referenced images in COLMAP model: {report.get('referenced_image_count', 0)}")
    print(f"Camera models: {report['camera_model_counts'] or '{}'}")
    print(f"Compatible (Faster-GS Inria COLMAP loader): {report['compatible_with_fastergs_inria_colmap_loader']}")

    for name, ok in report["checks"].items():
        marker = "OK" if ok else "FAIL"
        print(f" - {marker}: {name}")

    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f" - {w}")

    if report["unsupported_camera_models"]:
        print("Suggested next step:")
        print(
            " - Run a COLMAP undistortion/export step so sparse cameras become SIMPLE_PINHOLE or PINHOLE "
            "(or use the upstream convert.py flow in the Faster-GS Inria fork)."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Preflight checker for experimental Faster-GS (Inria fork) compatibility."
    )
    parser.add_argument("dataset", help="Dataset name under backend/datasets or absolute dataset path")
    parser.add_argument("--json", action="store_true", help="Print JSON only")
    args = parser.parse_args()

    candidate = Path(args.dataset)
    if not candidate.is_absolute():
        backend_root = Path(__file__).resolve().parent.parent
        candidate = backend_root / "datasets" / args.dataset
    dataset_dir = candidate.resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    report = inspect_dataset(dataset_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_human_report(report)

    raise SystemExit(0 if report["compatible_with_fastergs_inria_colmap_loader"] else 2)


if __name__ == "__main__":
    main()
