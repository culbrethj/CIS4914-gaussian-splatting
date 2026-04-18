"""
Local dataset prep helper for Faster-GS testing.

Runs ``colmap image_undistorter`` on an existing dataset locally (no SLURM,
no HPG) and checks the output with ``fastergs_preflight``. Handy when
iterating on camera-model / distortion fixes on a laptop without burning
HPG compute time. Not used by the web pipeline - the production path goes
through ``hpg_gs_final_prepare.py`` instead.

Requires local ``colmap`` on the PATH (or ``--colmap-executable``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Local version of the undistort step. Useful for testing the Faster-GS
# pipeline on a laptop without bouncing to HiPerGator (just run colmap
# image_undistorter here and then preflight to check the result).
# Not used in the web flow (that goes through hpg_gs_final_prepare.py instead).
try:
    from .fastergs_preflight import inspect_dataset, print_human_report
except ImportError:
    from fastergs_preflight import inspect_dataset, print_human_report


BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENT_OUTPUT_ROOT = BACKEND_DIR / "experiments" / "faster-gs" / "datasets"


def resolve_source_dataset(dataset_arg: str) -> Path:
    raw = Path(dataset_arg).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (BACKEND_DIR / "datasets" / dataset_arg).resolve()


def resolve_output_dir(source_dataset: Path, output: str | None, output_mode: str) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    if output_mode == "adjacent":
        return (source_dataset.parent / f"{source_dataset.name}_undistorted").resolve()
    return (DEFAULT_EXPERIMENT_OUTPUT_ROOT / source_dataset.name).resolve()


def resolve_colmap_executable(colmap_executable: str) -> str:
    candidate = Path(colmap_executable).expanduser()
    if candidate.parent != Path("."):
        if not candidate.exists():
            raise FileNotFoundError(f"COLMAP executable not found: {candidate}")
        return str(candidate)
    found = shutil.which(colmap_executable)
    if not found:
        raise FileNotFoundError(
            "COLMAP executable not found on PATH. Install COLMAP or pass --colmap-executable /path/to/colmap"
        )
    return found


def run_command(cmd: list[str], *, dry_run: bool):
    printable = " ".join(str(c) for c in cmd)
    print(f"[cmd] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_clean_output_dir(output_dir: Path, *, force: bool):
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --force to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_sparse_layout(output_dir: Path):
    sparse_dir = output_dir / "sparse"
    sparse0_dir = sparse_dir / "0"
    if sparse0_dir.exists() and sparse0_dir.is_dir():
        return

    if not sparse_dir.exists() or not sparse_dir.is_dir():
        raise FileNotFoundError(f"Expected sparse directory not found in output: {sparse_dir}")

    model_files = [
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "cameras.txt",
        "images.txt",
        "points3D.txt",
        "rigs.bin",
        "frames.bin",
        "rigs.txt",
        "frames.txt",
    ]
    has_root_model = any((sparse_dir / name).exists() for name in model_files)
    if not has_root_model:
        raise FileNotFoundError(
            "Could not find COLMAP model files in output sparse directory. "
            "Expected sparse/0/* or sparse/{cameras,images,points3D}.*"
        )

    sparse0_dir.mkdir(parents=True, exist_ok=True)
    for name in model_files:
        src = sparse_dir / name
        dst = sparse0_dir / name
        if src.exists() and src.is_file():
            shutil.move(str(src), str(dst))


def validate_source_dataset(source_dataset: Path):
    images_dir = source_dataset / "images"
    sparse0_dir = source_dataset / "sparse" / "0"
    if not source_dataset.exists() or not source_dataset.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Source images directory not found: {images_dir}")
    if not sparse0_dir.exists() or not sparse0_dir.is_dir():
        raise FileNotFoundError(f"Source sparse/0 directory not found: {sparse0_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental Faster-GS dataset prep: run COLMAP image_undistorter into a separate output "
            "dataset and verify compatibility with fastergs_preflight."
        )
    )
    parser.add_argument("dataset", help="Dataset name under backend/datasets or absolute dataset path")
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output dataset path (default: backend/experiments/faster-gs/datasets/<dataset>)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["experiments", "adjacent"],
        default="experiments",
        help="Default output style when --output is omitted",
    )
    parser.add_argument("--colmap-executable", default="colmap", help="COLMAP executable path or name")
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Optional max image size passed to colmap image_undistorter",
    )
    parser.add_argument("--force", action="store_true", help="Replace output dir if it already exists")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    source_dataset = resolve_source_dataset(args.dataset)
    validate_source_dataset(source_dataset)

    output_dir = resolve_output_dir(source_dataset, args.output, args.output_mode)
    if output_dir == source_dataset:
        raise ValueError("Output directory must be different from source dataset directory")

    if args.dry_run:
        try:
            colmap = resolve_colmap_executable(args.colmap_executable)
        except FileNotFoundError:
            colmap = args.colmap_executable
            print(f"[warn] COLMAP not found on PATH during dry-run; using placeholder executable: {colmap}")
    else:
        colmap = resolve_colmap_executable(args.colmap_executable)

    print(f"[info] Source dataset: {source_dataset}")
    print(f"[info] Output dataset: {output_dir}")
    print("[info] Original dataset will not be modified.")

    ensure_clean_output_dir(output_dir, force=args.force)

    cmd = [
        colmap,
        "image_undistorter",
        "--image_path",
        str(source_dataset / "images"),
        "--input_path",
        str(source_dataset / "sparse" / "0"),
        "--output_path",
        str(output_dir),
        "--output_type",
        "COLMAP",
    ]
    if args.max_image_size:
        cmd.extend(["--max_image_size", str(args.max_image_size)])

    run_command(cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("[dry-run] Skipping sparse layout normalization and preflight verification.")
        return

    normalize_sparse_layout(output_dir)
    report = inspect_dataset(output_dir)
    print_human_report(report)

    if not report["compatible_with_fastergs_inria_colmap_loader"]:
        raise SystemExit(2)

    print("[ok] Undistorted dataset is compatible with Faster-GS Inria loader.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)
