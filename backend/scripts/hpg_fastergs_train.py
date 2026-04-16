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
DEFAULT_REPO_DIR = "/blue/cis4914/joshuabowman/src/fastergs_inria"
DEFAULT_REPO_URL = "https://github.com/fhahlbohm/gaussian-splatting.git"
DEFAULT_ENV_PREFIX = "/blue/cis4914/joshuabowman/conda/fastergs_inria"
DEFAULT_MODULES = "git cmake gcc/12.2.0 cuda/12.4.1"
DEFAULT_BACKEND_PIP = (
    "git+https://github.com/nerficg-project/faster-gaussian-splatting/#subdirectory=FasterGSCudaBackend"
)


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
    try:
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return proc.stdout.strip()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            log("[cmd stdout]")
            print(exc.stdout.strip(), flush=True)
        if exc.stderr:
            log("[cmd stderr]")
            print(exc.stderr.strip(), flush=True)
        raise


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
    modules: str,
    repo_dir: str,
    repo_url: str,
    repo_branch: str,
    env_prefix: str,
    scene_dir: str,
    run_dir: str,
    iterations: int,
    extra_train_args: str,
    fastergs_backend_pip: str,
    preflight_script: str,
    converter_script: str,
    publish_splat: bool,
    publish_splat_path: str,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_gpus: int,
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
        f"#SBATCH --gres=gpu:{slurm_gpus}",
    ]
    if slurm_partition:
        lines.append(f"#SBATCH --partition={slurm_partition}")
    if slurm_account:
        lines.append(f"#SBATCH --account={slurm_account}")

    module_block = ""
    if modules.strip():
        module_block = f"module purge\nmodule load {modules.strip()}\n"

    publish_splat_str = "1" if publish_splat else "0"
    script = f"""#!/bin/bash
set -euo pipefail
{module_block}
echo "[train] host=$(hostname) started=$(date -Is)"
module -t list 2>&1 || true
nvidia-smi || true

if ! command -v conda >/dev/null 2>&1; then
  for p in \
    "/apps/conda/25.7.0/etc/profile.d/conda.sh" \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$p" ]; then
      source "$p"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "/apps/conda/25.7.0/bin/conda" ]; then
    eval "$(/apps/conda/25.7.0/bin/conda shell.bash hook)"
  fi
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda command not found" >&2
  exit 21
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

REPO_DIR={shlex.quote(repo_dir)}
REPO_URL={shlex.quote(repo_url)}
REPO_BRANCH={shlex.quote(repo_branch)}
ENV_PREFIX={shlex.quote(env_prefix)}
SCENE_DIR={shlex.quote(scene_dir)}
RUN_DIR={shlex.quote(run_dir)}
EXTRA_ARGS={shlex.quote(extra_train_args)}
FASTERGS_BACKEND_PIP={shlex.quote(fastergs_backend_pip)}
PRECHECK_SCRIPT={shlex.quote(preflight_script)}
CONVERTER_SCRIPT={shlex.quote(converter_script)}
PUBLISH_SPLAT={publish_splat_str}
PUBLISH_SPLAT_PATH={shlex.quote(publish_splat_path)}

if [ ! -d "$SCENE_DIR/images" ] || [ ! -d "$SCENE_DIR/sparse/0" ]; then
  echo "[error] Scene dir missing required Faster-GS layout: $SCENE_DIR" >&2
  exit 22
fi
if [ ! -f "$SCENE_DIR/sparse/0/cameras.bin" ] || [ ! -f "$SCENE_DIR/sparse/0/images.bin" ] || [ ! -f "$SCENE_DIR/sparse/0/points3D.bin" ]; then
  echo "[error] Scene sparse/0 is missing COLMAP triplet files" >&2
  exit 23
fi
if [ -f "$PRECHECK_SCRIPT" ]; then
  echo "[train] Running Faster-GS preflight validation"
  python "$PRECHECK_SCRIPT" "$SCENE_DIR"
else
  echo "[warn] Preflight script not found, skipping: $PRECHECK_SCRIPT" >&2
fi

mkdir -p "$(dirname "$REPO_DIR")" "$(dirname "$RUN_DIR")"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --recursive "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch --all
git checkout "$REPO_BRANCH"
git pull --ff-only
git submodule update --init --recursive

if [ ! -d "$ENV_PREFIX" ]; then
  ENV_FILE=""
  if [ -f "environment_cuda12.yml" ]; then
    ENV_FILE="environment_cuda12.yml"
  elif [ -f "environment.yml" ]; then
    ENV_FILE="environment.yml"
  fi
  if [ -z "$ENV_FILE" ]; then
    echo "[error] Could not find environment_cuda12.yml or environment.yml in repo" >&2
    exit 24
  fi
  conda env create -f "$ENV_FILE" -p "$ENV_PREFIX"
fi

conda activate "$ENV_PREFIX"
python --version

# The fhahlbohm Inria integration requires this extra CUDA backend package.
if ! python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("FasterGSCudaBackend") else 1)
PY
then
  echo "[train] Installing missing FasterGSCudaBackend extension"
  pip install "$FASTERGS_BACKEND_PIP" --no-build-isolation
fi

mkdir -p "$RUN_DIR"
echo "[train] scene=$SCENE_DIR"
echo "[train] run_dir=$RUN_DIR"
echo "[train] iterations={iterations}"

if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA_ARGS )
else
  EXTRA_ARR=()
fi

python train.py \
  -s "$SCENE_DIR" \
  -m "$RUN_DIR" \
  --iterations {iterations} \
  --disable_viewer \
  "${{EXTRA_ARR[@]}}"

PLY_PATH="$(ls -1 "$RUN_DIR"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -n 1 || true)"
if [ -z "$PLY_PATH" ]; then
  echo "[error] Could not find trained point_cloud.ply under $RUN_DIR/point_cloud" >&2
  exit 25
fi
echo "[train] ply=$PLY_PATH"

if [ "$PUBLISH_SPLAT" = "1" ]; then
  if [ -f "$CONVERTER_SCRIPT" ]; then
    python "$CONVERTER_SCRIPT" "$PLY_PATH"
    SPLAT_PATH="${{PLY_PATH%.ply}}.splat"
    if [ -f "$SPLAT_PATH" ]; then
      mkdir -p "$(dirname "$PUBLISH_SPLAT_PATH")"
      cp -f "$SPLAT_PATH" "$PUBLISH_SPLAT_PATH"
      echo "[train] splat=$SPLAT_PATH"
      echo "[train] published_splat=$PUBLISH_SPLAT_PATH"
    else
      echo "[warn] converter did not produce expected splat file: $SPLAT_PATH" >&2
    fi
  else
    echo "[warn] converter script not found, skipping splat publish: $CONVERTER_SCRIPT" >&2
  fi
fi

echo "[ok] Faster-GS training finished"
echo "[train] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + script + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Submit Faster-GS training to HiPerGator via SLURM (experimental path)."
    )
    parser.add_argument("dataset", help="Prepared dataset name (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg)")
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE, help="Remote project base directory")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Remote Faster-GS dataset root (default: <remote-base>/experiments/faster-gs/datasets)",
    )
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR, help="Remote Faster-GS repo checkout directory")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Faster-GS repo URL")
    parser.add_argument("--repo-branch", default="main", help="Faster-GS repo branch/tag")
    parser.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX, help="Remote conda env prefix")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Remote output root (default: <remote-base>/experiments/faster-gs/runs)",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--extra-train-args", default="", help="Extra args appended to train.py")
    parser.add_argument(
        "--fastergs-backend-pip",
        default=DEFAULT_BACKEND_PIP,
        help="Pip source for FasterGSCudaBackend (Inria integration dependency)",
    )
    parser.add_argument(
        "--preflight-script",
        default=None,
        help="Remote fastergs_preflight.py path (default: <remote-base>/backend/scripts/fastergs_preflight.py)",
    )
    parser.add_argument("--modules", default=DEFAULT_MODULES, help="Space-separated modules to load")
    parser.add_argument("--slurm-time", default="08:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-default", help="SLURM partition")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=8, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="32G", help="SLURM memory request")
    parser.add_argument("--slurm-gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--no-wait", action="store_true", help="Submit job and exit without polling")
    parser.add_argument("--no-publish-splat", action="store_true", help="Skip .splat conversion/publish step")
    parser.add_argument(
        "--publish-splat-dir",
        default=None,
        help="Remote publish dir for converted splat (default: <remote-base>/hipergator)",
    )
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
    dataset_root = (
        args.dataset_root.rstrip("/")
        if args.dataset_root
        else f"{remote_base}/experiments/faster-gs/datasets"
    )
    output_root = (
        args.output_root.rstrip("/")
        if args.output_root
        else f"{remote_base}/experiments/faster-gs/runs"
    )
    publish_splat_dir = (
        args.publish_splat_dir.rstrip("/")
        if args.publish_splat_dir
        else f"{remote_base}/hipergator"
    )

    scene_dir = f"{dataset_root}/{args.dataset}"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = f"{args.dataset}_{run_stamp}"
    run_dir = f"{output_root}/{run_tag}"
    publish_splat_path = f"{publish_splat_dir}/{run_tag}.splat"
    preflight_script = (
        args.preflight_script
        if args.preflight_script
        else f"{remote_base}/backend/scripts/fastergs_preflight.py"
    )
    converter_script = f"{remote_base}/backend/scripts/converter.py"
    remote_slurm_dir = f"{remote_base}/slurm_jobs"

    job_tag = f"fastergs_train_{run_tag}"
    remote_sbatch_path = f"{remote_slurm_dir}/{job_tag}.sbatch"
    remote_out_path = f"{remote_slurm_dir}/{job_tag}.out"
    remote_err_path = f"{remote_slurm_dir}/{job_tag}.err"
    local_out_path = local_logs / f"{job_tag}.out"
    local_err_path = local_logs / f"{job_tag}.err"

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ssh_base(args.remote, args.port, args.identity_file, ssh_opts)

    log(
        f"Submitting Faster-GS train job dataset={args.dataset} scene_dir={scene_dir} "
        f"run_dir={run_dir}"
    )

    run_cmd(
        ssh_cmd(
            ssh,
            "mkdir -p "
            + " ".join(
                [
                    shlex.quote(remote_base),
                    shlex.quote(remote_slurm_dir),
                    shlex.quote(output_root),
                    shlex.quote(publish_splat_dir),
                ]
            ),
        ),
        dry_run=args.dry_run,
    )

    sbatch_text = build_sbatch_script(
        modules=args.modules,
        repo_dir=args.repo_dir,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        env_prefix=args.env_prefix,
        scene_dir=scene_dir,
        run_dir=run_dir,
        iterations=args.iterations,
        extra_train_args=args.extra_train_args,
        fastergs_backend_pip=args.fastergs_backend_pip,
        preflight_script=preflight_script,
        converter_script=converter_script,
        publish_splat=not args.no_publish_splat,
        publish_splat_path=publish_splat_path,
        slurm_time=args.slurm_time,
        slurm_cpus=args.slurm_cpus,
        slurm_mem=args.slurm_mem,
        slurm_gpus=args.slurm_gpus,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        out_path=remote_out_path,
        err_path=remote_err_path,
        job_name=f"fgs_tr_{args.dataset}",
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

    if not args.dry_run and final_state.startswith("COMPLETED") and not args.no_publish_splat:
        local_splats_dir = backend_dir / "hipergator"
        local_splats_dir.mkdir(parents=True, exist_ok=True)
        local_splat_path = local_splats_dir / f"{run_tag}.splat"
        run_cmd(
            scp_download_cmd(args.remote, args.port, args.identity_file, publish_splat_path, local_splat_path, ssh_opts),
            dry_run=args.dry_run,
        )
        log(f"[ok] Fetched splat: {local_splat_path}")

    if not args.dry_run and not final_state.startswith("COMPLETED"):
        raise RuntimeError(
            f"Faster-GS train job ended in state {final_state}. Check logs: {local_out_path} {local_err_path}"
        )

    log("[ok] Faster-GS train workflow completed")
    log(f"[ok] Remote run dir: {run_dir}")
    if not args.no_publish_splat:
        log(f"[ok] Remote published splat (if generated): {publish_splat_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
