from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_cmd(cmd: list[str]):
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"[cmd] {printable}")
    subprocess.run(cmd, check=True)


def common_ssh_options(*, use_mux: bool, control_persist: str) -> list[str]:
    opts: list[str] = []
    if use_mux:
        opts.extend(
            [
                "-o",
                "ControlMaster=auto",
                "-o",
                f"ControlPersist={control_persist}",
                "-o",
                "ControlPath=/tmp/ssh_mux_%r_%h_%p",
            ]
        )
    return opts


def rsync_ssh_option(port: int, identity_file: str | None, ssh_opts: list[str]) -> str:
    parts = ["ssh", "-p", str(port)]
    if identity_file:
        parts.extend(["-i", identity_file])
    parts.extend(ssh_opts)
    return " ".join(shlex.quote(p) for p in parts)


def latest_matching_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate local prepare + HiPerGator Faster-GS prepare/train and publish .splat for frontend viewer."
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--video", required=True, help="Local dataset video file path")
    parser.add_argument("--iters", type=int, default=1000, help="Faster-GS train iterations")
    parser.add_argument("--duplicate-threshold", type=float, default=1.5)
    parser.add_argument("--blur-threshold", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--downscale", type=float, default=0.75)
    parser.add_argument("--max-width", type=int, default=1280)

    parser.add_argument("--remote", default=os.getenv("FASTERGS_REMOTE", "hpg"), help="SSH target/alias")
    parser.add_argument(
        "--remote-root",
        default=os.getenv("FASTERGS_REMOTE_ROOT", "/blue/cis4914/joshuabowman/gs_final"),
        help="Remote gs_final root",
    )
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", default=None)
    parser.add_argument("--no-ssh-mux", action="store_true")
    parser.add_argument("--ssh-control-persist", default="8h")

    parser.add_argument("--prepare-partition", default=os.getenv("FASTERGS_PREP_PARTITION", "hpg-default"))
    parser.add_argument("--prepare-cpus", type=int, default=int(os.getenv("FASTERGS_PREP_CPUS", "1")))
    parser.add_argument("--prepare-mem", default=os.getenv("FASTERGS_PREP_MEM", "8G"))
    parser.add_argument("--prepare-time", default=os.getenv("FASTERGS_PREP_TIME", "01:00:00"))

    parser.add_argument("--train-partition", default=os.getenv("FASTERGS_TRAIN_PARTITION", "hpg-b200"))
    parser.add_argument("--train-gpus", type=int, default=int(os.getenv("FASTERGS_TRAIN_GPUS", "1")))
    parser.add_argument("--train-cpus", type=int, default=int(os.getenv("FASTERGS_TRAIN_CPUS", "2")))
    parser.add_argument("--train-mem", default=os.getenv("FASTERGS_TRAIN_MEM", "24G"))
    parser.add_argument("--train-time", default=os.getenv("FASTERGS_TRAIN_TIME", "06:00:00"))
    parser.add_argument("--slurm-account", default=os.getenv("FASTERGS_SLURM_ACCOUNT", "cis4914"))
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent.parent
    scripts_dir = backend_dir / "scripts"
    dataset_dir = backend_dir / "datasets" / args.dataset
    images_dir = dataset_dir / "images"
    video_path = Path(args.video).resolve()

    fetch_dir = backend_dir / "hipergator" / "gs_final"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    log("INFO: Starting preprocessing")
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "pipeline.py"),
            args.dataset,
            "--video",
            str(video_path),
            "--only",
            "prepare",
            "--duplicate-threshold",
            str(args.duplicate_threshold),
            "--blur-threshold",
            str(args.blur_threshold),
            "--fps",
            str(args.fps),
            "--downscale",
            str(args.downscale),
            "--max-width",
            str(args.max_width),
        ]
    )
    if not images_dir.exists():
        raise RuntimeError(f"Preprocessing did not generate images: {images_dir}")
    log(f"INFO: Preprocessing finished ({len(list(images_dir.glob('*')))} files in images)")

    log("INFO: Starting Faster-GS sync (uploading preprocessed images to HiPerGator)")
    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh_base = ["ssh", "-p", str(args.port), *(["-i", args.identity_file] if args.identity_file else []), *ssh_opts]
    run_cmd(
        ssh_base
        + [
            args.remote,
            f"bash -lc {shlex.quote(f'mkdir -p {args.remote_root}/datasets/{args.dataset}/images {args.remote_root}/logs')}",
        ]
    )
    run_cmd(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            rsync_ssh_option(args.port, args.identity_file, ssh_opts),
            str(images_dir) + "/",
            f"{args.remote}:{args.remote_root}/datasets/{args.dataset}/images/",
        ]
    )
    log("INFO: Faster-GS sync finished")

    log("INFO: Starting SfM step on HiPerGator (queue + SfM + undistort; typically several minutes)")
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "hpg_gs_final_prepare.py"),
            args.dataset,
            "--remote",
            args.remote,
            "--remote-root",
            args.remote_root,
            "--slurm-account",
            args.slurm_account,
            "--slurm-partition",
            args.prepare_partition,
            "--preflight-script",
            f"{args.remote_root}/src/fastergs_preflight.py",
            "--slurm-cpus",
            str(args.prepare_cpus),
            "--slurm-mem",
            args.prepare_mem,
            "--slurm-time",
            args.prepare_time,
            "--port",
            str(args.port),
            "--ssh-control-persist",
            args.ssh_control_persist,
            *(["--identity-file", args.identity_file] if args.identity_file else []),
            *(["--no-ssh-mux"] if args.no_ssh_mux else []),
        ]
    )
    log("INFO: SfM step finished successfully")

    log("INFO: Starting Gaussian Splatting (fastergs) on HiPerGator GPU (may take 10-60+ minutes)")
    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "hpg_gs_final_train.py"),
            args.dataset,
            "--remote",
            args.remote,
            "--remote-root",
            args.remote_root,
            "--slurm-account",
            args.slurm_account,
            "--slurm-partition",
            args.train_partition,
            "--slurm-gpus",
            str(args.train_gpus),
            "--slurm-cpus",
            str(args.train_cpus),
            "--slurm-mem",
            args.train_mem,
            "--slurm-time",
            args.train_time,
            "--stage",
            "train",
            "--reuse-env",
            "--iterations",
            str(args.iters),
            "--fetch-local-dir",
            str(fetch_dir),
            "--port",
            str(args.port),
            "--ssh-control-persist",
            args.ssh_control_persist,
            *(["--identity-file", args.identity_file] if args.identity_file else []),
            *(["--no-ssh-mux"] if args.no_ssh_mux else []),
        ]
    )
    log("INFO: Gaussian Splatting finished successfully")

    latest_splat = latest_matching_file(fetch_dir, f"{args.dataset}_train_*.splat")
    latest_ply = latest_matching_file(fetch_dir, f"{args.dataset}_train_*.ply")
    if latest_splat is None:
        raise RuntimeError(f"No fetched Faster-GS .splat found in {fetch_dir} for dataset {args.dataset}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_dataset_splat = dataset_dir / "splat.splat"
    shutil.copy2(latest_splat, target_dataset_splat)

    target_gallery_splat = backend_dir / "hipergator" / f"{args.dataset}_fastergs_latest.splat"
    shutil.copy2(latest_splat, target_gallery_splat)

    if latest_ply is not None:
        shutil.copy2(latest_ply, dataset_dir / "splat.ply")

    log(f"INFO: Final .splat published to {target_dataset_splat}")
    log(f"INFO: Gallery .splat published to {target_gallery_splat}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
