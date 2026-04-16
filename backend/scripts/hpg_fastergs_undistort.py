from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

DEFAULT_REMOTE_BASE = "/blue/cis4914/joshuabowman/gaussian-splatting"
DEFAULT_SOURCE_ROOT = "/blue/cis4914/joshuabowman/datasets"
DEFAULT_COLMAP_CONTAINER = "/apps/colmap/3.11/container.sif"


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_cmd(cmd: list[str], *, dry_run: bool = False):
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"[cmd] {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_cmd_capture(cmd: list[str], *, dry_run: bool = False) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"[cmd] {printable}")
    if dry_run:
        return ""
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


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


def ssh_base(remote: str, port: int, identity_file: str | None, ssh_opts: list[str]) -> list[str]:
    base = ["ssh", "-p", str(port)]
    if identity_file:
        base.extend(["-i", identity_file])
    base.extend(ssh_opts)
    base.append(remote)
    return base


def ssh_cmd(ssh: list[str], remote_shell_cmd: str) -> list[str]:
    return ssh + [f"bash -lc {shlex.quote(remote_shell_cmd)}"]


def scp_upload_cmd(
    remote: str, port: int, identity_file: str | None, local_path: Path, remote_path: str, ssh_opts: list[str]
) -> list[str]:
    return [
        "scp",
        "-P",
        str(port),
        *(["-i", identity_file] if identity_file else []),
        *ssh_opts,
        str(local_path),
        f"{remote}:{remote_path}",
    ]


def scp_download_cmd(
    remote: str, port: int, identity_file: str | None, remote_path: str, local_path: Path, ssh_opts: list[str]
) -> list[str]:
    return [
        "scp",
        "-P",
        str(port),
        *(["-i", identity_file] if identity_file else []),
        *ssh_opts,
        f"{remote}:{remote_path}",
        str(local_path),
    ]


def parse_job_id(sbatch_output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {sbatch_output!r}")
    return match.group(1)


def poll_slurm_job(*, ssh: list[str], job_id: str, poll_seconds: int, dry_run: bool) -> str:
    if dry_run:
        run_cmd(ssh_cmd(ssh, f"squeue -h -j {job_id} -o %T"), dry_run=True)
        log("[dry-run] Skipping SLURM polling loop.")
        return "COMPLETED"

    last_state = None
    while True:
        state = run_cmd_capture(ssh_cmd(ssh, f"squeue -h -j {job_id} -o %T")).strip()
        if not state:
            break
        if state != last_state:
            log(f"SLURM job {job_id} state: {state}")
            last_state = state
        time.sleep(poll_seconds)

    final_state = run_cmd_capture(
        ssh_cmd(ssh, f"sacct -n -X -j {job_id} -o State | head -n 1 | awk '{{print $1}}'")
    ).strip()
    return final_state or "UNKNOWN"


def build_sbatch_script(
    *,
    source_root: str,
    dataset: str,
    output_root: str,
    colmap_container: str,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_partition: str | None,
    slurm_account: str | None,
    out_path: str,
    err_path: str,
    job_name: str,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
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

    src = f"{source_root.rstrip('/')}/{dataset}"
    out = f"{output_root.rstrip('/')}/{dataset}"
    src_q = shlex.quote(src)
    out_q = shlex.quote(out)
    container_q = shlex.quote(colmap_container)

    body = f"""#!/bin/bash
set -euo pipefail
echo "[undistort] host=$(hostname) started=$(date -Is)"

SRC={src_q}
OUT={out_q}
BASE_BIND="/blue:/blue"

if [ ! -d "$SRC/images" ]; then
  echo "[error] Missing source images dir: $SRC/images" >&2
  exit 11
fi

IN=""
for d in "$SRC/sparse/0" "$SRC/sparse"; do
  if [ -f "$d/cameras.bin" ] && [ -f "$d/images.bin" ] && [ -f "$d/points3D.bin" ]; then
    IN="$d"
    break
  fi
  if [ -f "$d/cameras.txt" ] && [ -f "$d/images.txt" ] && [ -f "$d/points3D.txt" ]; then
    IN="$d"
    break
  fi
done

if [ -z "$IN" ]; then
  echo "[error] Could not find COLMAP model triplet under $SRC/sparse" >&2
  find "$SRC/sparse" -maxdepth 2 -type f | sort || true
  exit 12
fi

echo "[undistort] source=$SRC"
echo "[undistort] model_input=$IN"
echo "[undistort] output=$OUT"

rm -rf "$OUT"
mkdir -p "$OUT"

apptainer exec --bind "$BASE_BIND" {container_q} colmap image_undistorter \\
  --image_path "$SRC/images" \\
  --input_path "$IN" \\
  --output_path "$OUT" \\
  --output_type COLMAP

if [ -d "$OUT/sparse" ] && [ ! -d "$OUT/sparse/0" ]; then
  if [ -f "$OUT/sparse/cameras.bin" ] && [ -f "$OUT/sparse/images.bin" ] && [ -f "$OUT/sparse/points3D.bin" ]; then
    mkdir -p "$OUT/sparse/0"
    mv "$OUT/sparse/cameras.bin" "$OUT/sparse/0/cameras.bin"
    mv "$OUT/sparse/images.bin" "$OUT/sparse/0/images.bin"
    mv "$OUT/sparse/points3D.bin" "$OUT/sparse/0/points3D.bin"
  elif [ -f "$OUT/sparse/cameras.txt" ] && [ -f "$OUT/sparse/images.txt" ] && [ -f "$OUT/sparse/points3D.txt" ]; then
    mkdir -p "$OUT/sparse/0"
    mv "$OUT/sparse/cameras.txt" "$OUT/sparse/0/cameras.txt"
    mv "$OUT/sparse/images.txt" "$OUT/sparse/0/images.txt"
    mv "$OUT/sparse/points3D.txt" "$OUT/sparse/0/points3D.txt"
  fi
fi

MODEL_IN="$OUT/sparse/0"
[ -d "$MODEL_IN" ] || MODEL_IN="$OUT/sparse"
mkdir -p "$OUT/sparse_txt"

apptainer exec --bind "$BASE_BIND" {container_q} colmap model_converter \\
  --input_path "$MODEL_IN" \\
  --output_path "$OUT/sparse_txt" \\
  --output_type TXT

echo "[undistort] first camera lines:"
head -n 20 "$OUT/sparse_txt/cameras.txt" || true

BAD_MODELS="$(awk 'NF && $1 !~ /^#/ {{print $2}}' "$OUT/sparse_txt/cameras.txt" | sort -u | grep -Ev '^(SIMPLE_PINHOLE|PINHOLE)$' || true)"
if [ -n "$BAD_MODELS" ]; then
  echo "[error] Unsupported camera models remain after undistortion: $BAD_MODELS" >&2
  exit 13
fi

echo "[ok] Undistorted dataset looks Faster-GS compatible"
echo "[undistort] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + body + "\n"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Submit experimental Faster-GS dataset undistortion via SLURM (COLMAP image_undistorter on compute node)."
        )
    )
    parser.add_argument("dataset", help="Dataset name under source root (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg or user@hpg.rc.ufl.edu)")
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE, help="Remote project working directory")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT, help="Remote dataset source root")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Remote output root (default: <remote-base>/experiments/faster-gs/datasets)",
    )
    parser.add_argument("--colmap-container", default=DEFAULT_COLMAP_CONTAINER, help="COLMAP Apptainer image path")
    parser.add_argument("--slurm-time", default="01:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-default", help="SLURM partition")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=4, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="16G", help="SLURM memory request")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--no-wait", action="store_true", help="Submit job and exit without polling")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH connection multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent.parent
    local_logs = backend_dir / "build_logs"
    local_logs.mkdir(parents=True, exist_ok=True)

    remote_base = args.remote_base.rstrip("/")
    source_root = args.source_root.rstrip("/")
    output_root = (
        args.output_root.rstrip("/")
        if args.output_root
        else f"{remote_base}/experiments/faster-gs/datasets"
    )
    remote_slurm_dir = f"{remote_base}/slurm_jobs"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_tag = f"fastergs_undistort_{args.dataset}_{stamp}"
    remote_sbatch_path = f"{remote_slurm_dir}/{job_tag}.sbatch"
    remote_out_path = f"{remote_slurm_dir}/{job_tag}.out"
    remote_err_path = f"{remote_slurm_dir}/{job_tag}.err"
    local_out_path = local_logs / f"{job_tag}.out"
    local_err_path = local_logs / f"{job_tag}.err"

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ssh_base(args.remote, args.port, args.identity_file, ssh_opts)

    log(
        f"Submitting Faster-GS undistort job dataset={args.dataset} source_root={source_root} "
        f"output_root={output_root}"
    )

    run_cmd(
        ssh_cmd(
            ssh,
            f"mkdir -p {shlex.quote(remote_base)} {shlex.quote(remote_slurm_dir)} {shlex.quote(output_root)}",
        ),
        dry_run=args.dry_run,
    )

    sbatch_text = build_sbatch_script(
        source_root=source_root,
        dataset=args.dataset,
        output_root=output_root,
        colmap_container=args.colmap_container,
        slurm_time=args.slurm_time,
        slurm_cpus=args.slurm_cpus,
        slurm_mem=args.slurm_mem,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        out_path=remote_out_path,
        err_path=remote_err_path,
        job_name=f"fgs_und_{args.dataset}",
    )

    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False, encoding="utf-8") as tf:
        tf.write(sbatch_text)
        local_temp = Path(tf.name)
    try:
        run_cmd(
            scp_upload_cmd(args.remote, args.port, args.identity_file, local_temp, remote_sbatch_path, ssh_opts),
            dry_run=args.dry_run,
        )
    finally:
        local_temp.unlink(missing_ok=True)

    if args.dry_run:
        run_cmd(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"), dry_run=True)
        job_id = "DRYRUN_JOB"
    else:
        sbatch_out = run_cmd_capture(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"))
        log(f"sbatch output: {sbatch_out}")
        job_id = parse_job_id(sbatch_out)
        log(f"Submitted SLURM job id: {job_id}")

    if args.no_wait:
        log("[ok] Submission complete (--no-wait).")
        log(f"Track with: squeue -j {job_id}")
        log(f"Remote logs: {remote_out_path} {remote_err_path}")
        return

    final_state = poll_slurm_job(ssh=ssh, job_id=job_id, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    log(f"SLURM job {job_id} final state: {final_state}")

    run_cmd(
        scp_download_cmd(args.remote, args.port, args.identity_file, remote_out_path, local_out_path, ssh_opts),
        dry_run=args.dry_run,
    )
    run_cmd(
        scp_download_cmd(args.remote, args.port, args.identity_file, remote_err_path, local_err_path, ssh_opts),
        dry_run=args.dry_run,
    )

    log(f"Local logs: {local_out_path} {local_err_path}")
    if not args.dry_run and not final_state.startswith("COMPLETED"):
        raise RuntimeError(
            f"Undistort job ended in state {final_state}. Check logs: {local_out_path} {local_err_path}"
        )

    log("[ok] Faster-GS undistort workflow completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
