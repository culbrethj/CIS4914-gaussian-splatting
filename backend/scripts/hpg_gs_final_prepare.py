"""
Submits the SfM + undistort SLURM job for the Faster-GS backend on HPG.

Workflow (inside the generated sbatch script, run on a CPU compute node):
  1. ``colmap feature_extractor`` on the rsynced images (SIMPLE_RADIAL).
  2. ``colmap sequential_matcher`` + ``colmap mapper`` -> sparse model.
  3. ``colmap image_undistorter`` rewrites images + sparse model into an
     undistorted PINHOLE space so the Inria Faster-GS loader can read it.
  4. Flattens the undistorter output into the standard ``sparse/0/`` layout.
  5. Runs ``fastergs_preflight.py`` to confirm the result is trainer-ready.

COLMAP runs inside an apptainer container (``/apps/colmap/3.11/container.sif``
by default) since HPG login/compute nodes don't have ``colmap`` on the PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    from .hpg_utils import common_ssh_options, format_cmd, log
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hpg_utils import common_ssh_options, format_cmd, log

# HPG workspace root. Required; no dev fallback is baked in. fastergs_pipeline.py
# (the usual entry point) loads the repo-root .env before invoking this script
# via subprocess, so the env var propagates through. Running this script
# standalone requires the caller to set FASTERGS_REMOTE_ROOT in their shell.
DEFAULT_REMOTE_ROOT = os.environ.get("FASTERGS_REMOTE_ROOT")
if not DEFAULT_REMOTE_ROOT:
    sys.exit(
        "FASTERGS_REMOTE_ROOT is not set. Copy .env.example to .env "
        "at the repo root and fill in your HPG workspace path "
        "(/blue/cis4914/<your-gatorlink>/gs_final)."
    )
DEFAULT_COLMAP_CONTAINER = os.environ.get(
    "FASTERGS_COLMAP_CONTAINER", "/apps/colmap/3.11/container.sif"
)
DEFAULT_PREFLIGHT_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/fastergs_preflight.py"


def run_cmd(cmd: list[str], *, dry_run: bool = False):
    log(f"[cmd] {format_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_cmd_capture(cmd: list[str], *, dry_run: bool = False, log_cmd: bool = True) -> str:
    if log_cmd:
        log(f"[cmd] {format_cmd(cmd)}")
    if dry_run:
        return ""
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def bash_lc(command: str) -> str:
    return f"bash -lc {shlex.quote(command)}"


def parse_job_id(sbatch_output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {sbatch_output!r}")
    return match.group(1)


def poll_slurm_job(*, ssh: list[str], job_id: str, poll_seconds: int, dry_run: bool) -> str:
    squeue_cmd = f"squeue -h -j {job_id} -o %T\\|%R"
    sacct_cmd = f"sacct -n -X -j {job_id} -o State | head -n 1 | awk '{{print $1}}'"
    if dry_run:
        run_cmd(ssh + [bash_lc(squeue_cmd)], dry_run=True)
        return "COMPLETED"

    last_state = None
    start = time.time()
    next_heartbeat = start + max(60, poll_seconds * 6)
    log(f"Waiting for SLURM job {job_id} (poll interval {poll_seconds}s)")
    while True:
        state_reason = run_cmd_capture(ssh + [bash_lc(squeue_cmd)], log_cmd=False).strip()
        if not state_reason:
            break
        if "|" in state_reason:
            state, reason = state_reason.split("|", 1)
        else:
            state, reason = state_reason, ""
        state = state.strip()
        reason = reason.strip()
        if state != last_state:
            if reason and reason not in {"None", "(None)", "null", "(null)"}:
                if reason.startswith("(") and reason.endswith(")"):
                    log(f"SLURM job {job_id} state: {state} {reason}")
                else:
                    log(f"SLURM job {job_id} state: {state} ({reason})")
            else:
                log(f"SLURM job {job_id} state: {state}")
            last_state = state
        now = time.time()
        if now >= next_heartbeat:
            elapsed_min = (now - start) / 60.0
            log(
                f"SLURM job {job_id} still {state} after {elapsed_min:.1f} min "
                "(SfM + undistort can take several minutes)"
            )
            next_heartbeat = now + max(60, poll_seconds * 6)
        time.sleep(poll_seconds)

    final_state = run_cmd_capture(ssh + [bash_lc(sacct_cmd)], log_cmd=False).strip()
    return final_state or "UNKNOWN"


def print_log_tail(path: Path, label: str, lines: int = 60):
    if not path.exists():
        return
    try:
        all_lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return
    if not all_lines:
        return
    tail = all_lines[-lines:]
    log(f"Last {len(tail)} lines from {label}:")
    for line in tail:
        print(line, flush=True)


def build_sbatch_script(
    *,
    remote_root: str,
    dataset: str,
    colmap_container: str,
    preflight_script: str,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_partition: str | None,
    slurm_account: str | None,
    out_path: str,
    err_path: str,
) -> str:
    # Builds the sbatch script that runs on HPG. The script does:
    #   1) SIFT feature extraction with SIMPLE_RADIAL camera model
    #   2) sequential matching (works well for video-extracted frames)
    #   3) incremental mapping -> sparse model
    #   4) image_undistorter  <-- this is the important step
    # Step 4 rewrites the images + sparse model into an undistorted PINHOLE
    # space. We need PINHOLE because the Faster-GS Inria fork's dataset
    # loader only supports PINHOLE/SIMPLE_PINHOLE. Then we run preflight to
    # make sure the output is actually what the trainer expects.
    #
    # COLMAP runs inside an apptainer (singularity) container at
    # /apps/colmap/3.11/container.sif. HPG login nodes don't have COLMAP on
    # the PATH directly. The --bind mounts /blue so the container sees our
    # dataset dirs.
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gsf_prepare",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
        f"#SBATCH --time={slurm_time}",
        f"#SBATCH --cpus-per-task={slurm_cpus}",
        f"#SBATCH --mem={slurm_mem}",
    ]
    if slurm_partition:
        lines.append(f"#SBATCH --partition={slurm_partition}")
    if slurm_account:
        lines.append(f"#SBATCH --account={slurm_account}")

    script = f"""#!/bin/bash
set -euo pipefail
echo "[prepare] host=$(hostname) started=$(date -Is)"

ROOT={shlex.quote(remote_root)}
DATASET={shlex.quote(dataset)}
COLMAP_CONTAINER={shlex.quote(colmap_container)}
PREFLIGHT={shlex.quote(preflight_script)}

SRC="$ROOT/datasets/$DATASET"
IMG="$SRC/images"
DB="$SRC/database.db"
SPARSE_REBUILD="$SRC/sparse_rebuild"
SPARSE_FINAL="$SRC/sparse/0"
UND="$ROOT/experiments/faster-gs/datasets/$DATASET"

mkdir -p "$ROOT/logs" "$ROOT/experiments/faster-gs/datasets"

if [ ! -d "$IMG" ]; then
  echo "[error] Missing images directory: $IMG" >&2
  exit 12
fi

echo "[prepare] Input image count: $(find "$IMG" -maxdepth 1 -type f | wc -l)"

rm -f "$DB"
rm -rf "$SPARSE_REBUILD" "$SRC/sparse" "$UND"
mkdir -p "$SPARSE_REBUILD" "$SPARSE_FINAL"

COLMAP="apptainer exec --bind /blue:/blue $COLMAP_CONTAINER colmap"
THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

$COLMAP feature_extractor \
  --database_path "$DB" \
  --image_path "$IMG" \
  --ImageReader.single_camera 0 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.max_num_features 4096 \
  --SiftExtraction.num_threads "$THREADS" \
  --SiftExtraction.use_gpu 0

$COLMAP sequential_matcher \
  --database_path "$DB" \
  --SequentialMatching.overlap 10 \
  --SequentialMatching.loop_detection 0 \
  --SiftMatching.num_threads "$THREADS" \
  --SiftMatching.use_gpu 0

$COLMAP mapper \
  --database_path "$DB" \
  --image_path "$IMG" \
  --output_path "$SPARSE_REBUILD"

MODEL_DIR="$(find "$SPARSE_REBUILD" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
if [ -z "$MODEL_DIR" ]; then
  echo "[error] COLMAP mapper did not produce a sparse model directory" >&2
  exit 13
fi

cp "$MODEL_DIR/cameras.bin" "$MODEL_DIR/images.bin" "$MODEL_DIR/points3D.bin" "$SPARSE_FINAL/"
echo "[prepare] Rebuilt sparse model: $SPARSE_FINAL"

$COLMAP image_undistorter \
  --image_path "$IMG" \
  --input_path "$SPARSE_FINAL" \
  --output_path "$UND" \
  --output_type COLMAP

# image_undistorter writes sparse/cameras.bin directly (no sparse/0/ subdir),
# but the Faster-GS loader expects the standard COLMAP sparse/0/ layout.
# Flatten/rename it here so the downstream trainer finds what it expects.
if [ -d "$UND/sparse" ] && [ ! -d "$UND/sparse/0" ]; then
  if [ -f "$UND/sparse/cameras.bin" ] && [ -f "$UND/sparse/images.bin" ] && [ -f "$UND/sparse/points3D.bin" ]; then
    mkdir -p "$UND/sparse/0"
    mv "$UND/sparse/cameras.bin" "$UND/sparse/0/cameras.bin"
    mv "$UND/sparse/images.bin" "$UND/sparse/0/images.bin"
    mv "$UND/sparse/points3D.bin" "$UND/sparse/0/points3D.bin"
  fi
fi

if [ -f "$PREFLIGHT" ]; then
  python3 "$PREFLIGHT" "$UND"
else
  echo "[warn] Preflight script not found: $PREFLIGHT"
fi

echo "[prepare] Undistorted image count: $(find "$UND/images" -maxdepth 1 -type f | wc -l)"
echo "[ok] gs_final prepare complete"
echo "[prepare] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + script + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="SLURM prepare stage for gs_final: rebuild sparse from cleaned images, undistort, and preflight."
    )
    parser.add_argument("dataset", help="Dataset name (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg)")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote gs_final root")
    parser.add_argument("--colmap-container", default=DEFAULT_COLMAP_CONTAINER, help="COLMAP container path")
    parser.add_argument("--preflight-script", default=DEFAULT_PREFLIGHT_SCRIPT, help="Remote preflight script path")
    parser.add_argument("--slurm-time", default="02:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-default", help="SLURM partition")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=4, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="16G", help="SLURM memory request")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return without waiting")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent.parent
    local_logs = backend_dir / "build_logs"
    local_logs.mkdir(parents=True, exist_ok=True)

    root = args.remote_root.rstrip("/")
    remote_logs_dir = f"{root}/logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"gsf_prepare_{args.dataset}_{stamp}"
    remote_sbatch = f"{remote_logs_dir}/{tag}.sbatch"
    remote_out = f"{remote_logs_dir}/{tag}.out"
    remote_err = f"{remote_logs_dir}/{tag}.err"
    local_out = local_logs / f"{tag}.out"
    local_err = local_logs / f"{tag}.err"

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ["ssh", "-p", str(args.port)]
    if args.identity_file:
        ssh.extend(["-i", args.identity_file])
    ssh.extend(ssh_opts)
    ssh.append(args.remote)

    log(
        f"Starting gs_final prepare dataset={args.dataset} remote_root={root} "
        "(rebuild sparse + undistort + preflight)"
    )
    run_cmd(ssh + [bash_lc(f"mkdir -p {shlex.quote(root)} {shlex.quote(remote_logs_dir)}")], dry_run=args.dry_run)

    sbatch_text = build_sbatch_script(
        remote_root=root,
        dataset=args.dataset,
        colmap_container=args.colmap_container,
        preflight_script=args.preflight_script,
        slurm_time=args.slurm_time,
        slurm_cpus=args.slurm_cpus,
        slurm_mem=args.slurm_mem,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        out_path=remote_out,
        err_path=remote_err,
    )

    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False, encoding="utf-8") as tf:
        tf.write(sbatch_text)
        temp_path = Path(tf.name)
    try:
        run_cmd(
            [
                "scp",
                "-P",
                str(args.port),
                *(["-i", args.identity_file] if args.identity_file else []),
                *ssh_opts,
                str(temp_path),
                f"{args.remote}:{remote_sbatch}",
            ],
            dry_run=args.dry_run,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if args.dry_run:
        run_cmd(ssh + [bash_lc(f"sbatch {shlex.quote(remote_sbatch)}")], dry_run=True)
        job_id = "DRYRUN_JOB"
    else:
        out = run_cmd_capture(ssh + [bash_lc(f"sbatch {shlex.quote(remote_sbatch)}")])
        log(f"sbatch output: {out}")
        job_id = parse_job_id(out)
        log(f"Submitted SLURM job id: {job_id}")

    if args.no_wait:
        log("[ok] Submission complete (--no-wait)")
        log(f"Track with: squeue -j {job_id}")
        log(f"Remote logs: {remote_out} {remote_err}")
        return

    final_state = poll_slurm_job(ssh=ssh, job_id=job_id, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    log(f"SLURM job {job_id} final state: {final_state}")

    run_cmd(
        [
            "scp",
            "-P",
            str(args.port),
            *(["-i", args.identity_file] if args.identity_file else []),
            *ssh_opts,
            f"{args.remote}:{remote_out}",
            str(local_out),
        ],
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            "scp",
            "-P",
            str(args.port),
            *(["-i", args.identity_file] if args.identity_file else []),
            *ssh_opts,
            f"{args.remote}:{remote_err}",
            str(local_err),
        ],
        dry_run=args.dry_run,
    )
    log(f"Local logs: {local_out} {local_err}")

    if not args.dry_run and not final_state.startswith("COMPLETED"):
        print_log_tail(local_err, "prepare stderr log")
        print_log_tail(local_out, "prepare stdout log")
        raise RuntimeError(f"Prepare job ended in state {final_state}. Check logs: {local_out} {local_err}")
    log("[ok] gs_final prepare stage completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
