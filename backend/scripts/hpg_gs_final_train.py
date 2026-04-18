# Third-party code and techniques used in this file:
#  - Faster-GS (Hahlbohm et al., CVPR 2026)
#    https://github.com/nerficg-project/faster-gaussian-splatting
#  - Faster-GS Inria fork (train.py, arguments module patched for SHORTGS_*)
#    https://github.com/fhahlbohm/gaussian-splatting
#  - Original 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)
#    https://github.com/graphdeco-inria/gaussian-splatting
#  - Shorter-Splatting training techniques (scale reset, entropy, progressive)
#    arXiv 2603.09277
"""
Submits the Faster-GS GPU training job to SLURM and fetches the result.

Generates + uploads an sbatch script that:
  1. Loads modules, builds/activates the pinned conda env.
  2. Pulls the Inria Faster-GS fork (git clone/pull) into ``$REPO_DIR``.
  3. Runs ``shortgs_apply_patches.py`` to insert the paper's techniques
     into ``train.py`` behind ``SHORTGS_*`` env vars (no-op when unset).
  4. Tees ``python train.py`` stdout to ``$TRAIN_LOG`` and runs at
     ``EVAL_STRIDE=500`` for both ``--test_iterations`` and
     ``--save_iterations`` so the Reports page gets a PSNR + gaussian-
     count sample every 500 steps. Loss is denser (every 100 iters,
     parsed from tqdm by ``metrics_collector.py``).
  5. Converts the final ``point_cloud.ply`` -> ``.splat`` via ``converter.py``.
  6. Collects structured metrics with ``metrics_collector.py`` and renders
     per-metric PNGs with ``metrics_plotter.py``.

After SLURM reports COMPLETED the script scp's ``.splat`` + ``.ply`` +
metrics back to ``backend/hipergator/gs_final/`` + ``backend/datasets/<ds>/metrics/<run_tag>/``.

Most HPG-specific paths default to ``/blue/cis4914/joshuabowman/...`` but
every one can be overridden via CLI flag or environment variable; see
``--help`` for the full list.
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

# HPG paths + pinned toolchain.
# The repo URL is the Hahlbohm fork of the Inria gaussian-splatting repo, which
# has the Faster-GS modifications applied. We install the CUDA backend (the
# custom rasterizer) separately from the NeRFICG project.
# Python 3.10.14 + CUDA 12.8 are pinned to match what B200 drivers expect.
# TORCH_CUDA_ARCH_LIST "8.9;10.0" covers both L4 (hpg-turin, sm_89) and
# B200 (hpg-b200, sm_100) so the env's CUDA extensions run on either
# partition without a rebuild. Override if your target partition uses a
# different compute capability.
# HPG workspace root. Every default path below is derived from this so you
# can point the whole pipeline at a different user's /blue space by setting
# one env var (FASTERGS_REMOTE_ROOT). CLI flags still override per-path.
DEFAULT_REMOTE_ROOT = os.environ.get("FASTERGS_REMOTE_ROOT", "/blue/cis4914/joshuabowman/gs_final")
DEFAULT_REPO_DIR = f"{DEFAULT_REMOTE_ROOT}/src/fastergs_inria"
DEFAULT_REPO_URL = "https://github.com/fhahlbohm/gaussian-splatting.git"
DEFAULT_ENV_PREFIX = f"{DEFAULT_REMOTE_ROOT}/envs/fastergs_cuda128"
DEFAULT_MODULES = "git cmake gcc/12.2.0 conda/25.7.0 cuda/12.8.1"
DEFAULT_BACKEND_PIP = (
    "git+https://github.com/nerficg-project/faster-gaussian-splatting/#subdirectory=FasterGSCudaBackend"
)
DEFAULT_CONVERTER_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/converter.py"
DEFAULT_PREFLIGHT_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/fastergs_preflight.py"
DEFAULT_METRICS_COLLECTOR_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/metrics_collector.py"
DEFAULT_METRICS_PLOTTER_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/metrics_plotter.py"
# Shorter-Splatting paper patch applier - runs right after the repo is
# cloned/updated to insert the SHORTGS_* technique blocks into train.py.
# Idempotent; safe to run on every training job.
DEFAULT_SHORTGS_PATCHES_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/shortgs_apply_patches.py"
# OpenSplat install root on the remote. The SLURM body invokes the
# prebuilt C++ binary under this dir when --backend opensplat is set.
# Override with FASTERGS_OPENSPLAT_ROOT (or --opensplat-root) if you've
# installed OpenSplat somewhere other than $REMOTE_ROOT/src/OpenSplat.
DEFAULT_OPENSPLAT_ROOT = os.environ.get(
    "FASTERGS_OPENSPLAT_ROOT",
    f"{DEFAULT_REMOTE_ROOT}/src/OpenSplat",
)
DEFAULT_PINNED_PYTHON = "3.10.14"
DEFAULT_PINNED_TORCH = "auto"
DEFAULT_PINNED_TORCHVISION = "auto"
DEFAULT_PINNED_TORCHAUDIO = "auto"
DEFAULT_PINNED_PYTORCH_CUDA = "12.8"
DEFAULT_TORCH_CUDA_ARCH_LIST = "8.9;10.0"


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
            log(f"SLURM job {job_id} still {state} after {elapsed_min:.1f} min")
            next_heartbeat = now + max(60, poll_seconds * 6)
        time.sleep(poll_seconds)

    final_state = run_cmd_capture(ssh + [bash_lc(sacct_cmd)], log_cmd=False).strip()
    return final_state or "UNKNOWN"


def print_log_tail(path: Path, label: str, lines: int = 80):
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


def check_remote_dataset(*, ssh: list[str], scene_dir: str, preflight_script: str, dry_run: bool):
    check_script = f"""
set -euo pipefail
SCENE={shlex.quote(scene_dir)}
PREFLIGHT={shlex.quote(preflight_script)}

if [ ! -d "$SCENE" ]; then
  echo "[error] Dataset directory not found: $SCENE" >&2
  exit 41
fi
if [ ! -d "$SCENE/images" ]; then
  echo "[error] Missing images dir: $SCENE/images" >&2
  exit 42
fi
if [ ! -d "$SCENE/sparse/0" ]; then
  echo "[error] Missing sparse model dir: $SCENE/sparse/0" >&2
  exit 43
fi
IMAGE_COUNT="$(find "$SCENE/images" -maxdepth 1 -type f | wc -l)"
echo "[check] image_count=$IMAGE_COUNT"
if [ "$IMAGE_COUNT" -lt 8 ]; then
  echo "[error] Too few images for Faster-GS ($IMAGE_COUNT < 8): $SCENE/images" >&2
  exit 44
fi
if [ ! -f "$SCENE/sparse/0/cameras.bin" ] || [ ! -f "$SCENE/sparse/0/images.bin" ]; then
  echo "[error] Missing COLMAP sparse binaries in $SCENE/sparse/0 (need cameras.bin and images.bin)" >&2
  exit 45
fi

if [ -f "$PREFLIGHT" ]; then
  PY_CMD="$(command -v python || command -v python3 || true)"
  if [ -z "$PY_CMD" ]; then
    echo "[warn] No python interpreter found for preflight; skipping script check."
  else
    "$PY_CMD" "$PREFLIGHT" "$SCENE"
  fi
else
  echo "[warn] Preflight script missing at $PREFLIGHT; used basic dataset checks only."
fi
"""
    run_cmd(ssh + [bash_lc(check_script)], dry_run=dry_run)


def sync_remote_file(
    *,
    ssh: list[str],
    scp_base: list[str],
    remote_host: str,
    local_source: Path,
    remote_target: str,
    label: str,
    dry_run: bool,
):
    if not local_source.is_file():
        log(f"[warn] Local {label} missing, skipping upload: {local_source}")
        return
    remote_parent = remote_target.rsplit("/", 1)[0]
    run_cmd(ssh + [bash_lc(f"mkdir -p {shlex.quote(remote_parent)}")], dry_run=dry_run)
    run_cmd(scp_base + [str(local_source), f"{remote_host}:{remote_target}"], dry_run=dry_run)
    log(f"[sync] Uploaded {label} to {remote_target}")


def build_opensplat_sbatch_script(
    *,
    remote_root: str,
    dataset: str,
    stage: str,
    iterations: int,
    env_prefix: str,
    opensplat_root: str,
    converter_script: str,
    preflight_script: str,
    metrics_collector_script: str,
    metrics_plotter_script: str,
    dataset_label_for_metrics: str,
    run_tag: str,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_gpus: int,
    slurm_partition: str | None,
    slurm_account: str | None,
    out_path: str,
    err_path: str,
) -> str:
    # OpenSplat sbatch. Separate from the fastergs body because it doesn't
    # touch conda, doesn't clone the Inria fork, and doesn't apply any
    # shortgs/fastergs patches. It links against libtorch from the existing
    # fastergs_cuda128 env purely for the shared object runtime.
    #
    # Binary selection: hpg-b200 partitions use the build_b200/ binary
    # (compiled with sm_89;100); everything else uses build/ (sm_80;89;90).
    # The run dir mirrors the fastergs layout so the fetch + gallery code
    # paths on the local side don't need to care which backend produced it.
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=opensplat_{stage}",
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

    # OpenSplat doesn't read any SHORTGS_* vars, but we forward SHORTGS_SEED
    # so the metrics summary can still record which seed the matrix asked
    # for (value is informational only; OpenSplat is non-seedable).
    seed_forward = f"export SHORTGS_SEED={shlex.quote(os.environ.get('SHORTGS_SEED', ''))}"

    script = f"""#!/bin/bash
set -euo pipefail
module purge
module load cuda/12.8.1 gcc/12.2.0 opencv/4.7.0

echo "[opensplat] host=$(hostname) started=$(date -Is) stage={stage}"
module -t list 2>&1 || true
nvidia-smi || true
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1 || true

ROOT={shlex.quote(remote_root)}
DATASET={shlex.quote(dataset)}
STAGE={shlex.quote(stage)}
RUN_TAG={shlex.quote(run_tag)}
OPENSPLAT_ROOT={shlex.quote(opensplat_root)}
ENV_PREFIX={shlex.quote(env_prefix)}
CONVERTER={shlex.quote(converter_script)}
PREFLIGHT={shlex.quote(preflight_script)}
METRICS_COLLECTOR={shlex.quote(metrics_collector_script)}
METRICS_PLOTTER={shlex.quote(metrics_plotter_script)}
METRICS_DATASET_LABEL={shlex.quote(dataset_label_for_metrics)}
{seed_forward}

# Libtorch is pulled from the existing fastergs_cuda128 env (same env we
# linked OpenSplat against at build time). If the env is missing we fail
# fast; no point running with a different torch ABI.
TORCH_LIB="$ENV_PREFIX/lib/python3.10/site-packages/torch/lib"
if [ ! -d "$TORCH_LIB" ]; then
  echo "[error] torch lib dir missing: $TORCH_LIB" >&2
  exit 51
fi
export LD_LIBRARY_PATH="$TORCH_LIB:/apps/opencv/4.7.0/lib64:${{LD_LIBRARY_PATH:-}}"

# Partition-aware binary selection. The build/ variant covers L4 + A100 +
# H100; build_b200/ covers B200. We check the dir exists before committing
# to it so a stale/incomplete install doesn't silently fall back.
PARTITION="${{SLURM_JOB_PARTITION:-}}"
case "$PARTITION" in
  hpg-b200)
    BIN="$OPENSPLAT_ROOT/build_b200/opensplat"
    ;;
  *)
    BIN="$OPENSPLAT_ROOT/build/opensplat"
    ;;
esac
if [ ! -x "$BIN" ]; then
  echo "[error] OpenSplat binary missing or not executable: $BIN" >&2
  exit 52
fi
echo "[opensplat] using binary: $BIN"

SCENE="$ROOT/experiments/faster-gs/datasets/$DATASET"
if [ ! -d "$SCENE/images" ] || [ ! -d "$SCENE/sparse/0" ]; then
  echo "[error] Missing prepared dataset layout at: $SCENE" >&2
  exit 53
fi

# Preflight reuses the same script the fastergs branch uses. It just
# asserts COLMAP sparse/0 + images/ exist and has a minimum image count.
if [ -f "$PREFLIGHT" ]; then
  PY_CMD="$(command -v python3 || command -v python || true)"
  if [ -n "$PY_CMD" ]; then
    "$PY_CMD" "$PREFLIGHT" "$SCENE" || echo "[warn] preflight returned non-zero"
  fi
fi

# Mirror the fastergs run_dir layout so downstream fetch/gallery code finds
# the PLY at RUN_DIR/point_cloud/iteration_N/point_cloud.ply. The collector
# then picks this path up the same way it does for fastergs runs.
RUN_DIR="$ROOT/outputs/faster-gs/$RUN_TAG"
SPLAT_OUT="$ROOT/outputs/$RUN_TAG.splat"
METRICS_DIR="$RUN_DIR/metrics"
TRAIN_LOG="$RUN_DIR/train_stdout.log"
PC_ITER_DIR="$RUN_DIR/point_cloud/iteration_{iterations}"
mkdir -p "$RUN_DIR" "$METRICS_DIR" "$PC_ITER_DIR"

TRAIN_START_EPOCH=$(date +%s)
echo "[opensplat] train_start_epoch=$TRAIN_START_EPOCH"
echo "[opensplat] scene=$SCENE run_dir=$RUN_DIR iterations={iterations}"

# Invoke OpenSplat. Output PLY lands in the iteration_N dir so the fetch +
# gallery pipeline treat it identically to a fastergs checkpoint.
"$BIN" "$SCENE" -o "$PC_ITER_DIR/point_cloud.ply" -n {iterations} 2>&1 | tee "$TRAIN_LOG"

if [ ! -f "$PC_ITER_DIR/point_cloud.ply" ]; then
  echo "[error] OpenSplat did not produce a PLY: $PC_ITER_DIR/point_cloud.ply" >&2
  exit 54
fi

# Convert ply->splat on HPG so we only scp the smaller splat artifact back.
# The converter drops the splat next to the input PLY, so we copy it into
# place under $ROOT/outputs/<run_tag>.splat the same way the fastergs branch
# does.
if [ -f "$CONVERTER" ]; then
  PY_CMD="$(command -v python3 || command -v python || true)"
  if [ -n "$PY_CMD" ]; then
    "$PY_CMD" "$CONVERTER" "$PC_ITER_DIR/point_cloud.ply" || echo "[warn] converter failed"
    SPLAT_PATH="$PC_ITER_DIR/point_cloud.splat"
    if [ -f "$SPLAT_PATH" ]; then
      cp -f "$SPLAT_PATH" "$SPLAT_OUT"
      echo "[opensplat] splat=$SPLAT_OUT"
    fi
  fi
fi

# Run the shared metrics collector in opensplat mode. The parser reads the
# "Step N: loss (pct%)" lines tee'd into TRAIN_LOG and writes records with
# the same schema fastergs uses (iteration/loss/psnr/num_gaussians/wall_seconds).
if [ -f "$METRICS_COLLECTOR" ]; then
  SEED_ARG=""
  if [ -n "${{SHORTGS_SEED:-}}" ]; then SEED_ARG="--seed $SHORTGS_SEED"; fi
  PY_CMD="$(command -v python3 || command -v python || true)"
  if [ -n "$PY_CMD" ]; then
    "$PY_CMD" "$METRICS_COLLECTOR" \\
      --run-dir "$RUN_DIR" \\
      --log-file "$TRAIN_LOG" \\
      --out-dir "$METRICS_DIR" \\
      --backend opensplat \\
      --dataset "$METRICS_DATASET_LABEL" \\
      --run-tag "$RUN_TAG" \\
      --iterations {iterations} \\
      --partition "${{SLURM_JOB_PARTITION:-}}" \\
      --start-epoch "$TRAIN_START_EPOCH" \\
      $SEED_ARG || echo "[warn] metrics_collector returned non-zero"
  fi
fi

# Plots are best-effort; matplotlib might not be on the system python.
if [ -f "$METRICS_PLOTTER" ]; then
  PY_CMD="$(command -v python3 || command -v python || true)"
  if [ -n "$PY_CMD" ]; then
    "$PY_CMD" "$METRICS_PLOTTER" \\
      --metrics-dir "$METRICS_DIR" \\
      --title-prefix "$METRICS_DATASET_LABEL $RUN_TAG" || echo "[warn] metrics_plotter returned non-zero"
  fi
fi

echo "[opensplat] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + script + "\n"


def build_sbatch_script(
    *,
    remote_root: str,
    dataset: str,
    repo_dir: str,
    repo_url: str,
    repo_branch: str,
    env_prefix: str,
    modules: str,
    stage: str,
    reuse_env: bool,
    iterations: int,
    pinned_python: str,
    pinned_torch: str,
    pinned_torchvision: str,
    pinned_torchaudio: str,
    pinned_pytorch_cuda: str,
    torch_cuda_arch_list: str,
    backend_pip: str,
    converter_script: str,
    preflight_script: str,
    metrics_collector_script: str,
    metrics_plotter_script: str,
    shortgs_patches_script: str,
    dataset_label_for_metrics: str,
    run_tag: str,
    use_fastergs_adam: bool,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_gpus: int,
    slurm_partition: str | None,
    slurm_account: str | None,
    out_path: str,
    err_path: str,
    backend: str = "fastergs",
    opensplat_root: str = DEFAULT_OPENSPLAT_ROOT,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=gsf_{stage}",
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

    # Forward SHORTGS_* env vars from whoever invoked this script (typically
    # fastergs_pipeline.py with flags from the user / experiment harness).
    # Each becomes an export line in the SLURM bash; missing ones are
    # explicitly unset so the trainer sees consistent state regardless of
    # what might already be in the login shell env.
    shortgs_vars = [
        "SHORTGS_SCALE_RESET_EVERY",
        "SHORTGS_SCALE_RESET_FACTOR",
        "SHORTGS_ENTROPY_WEIGHT",
        "SHORTGS_PROGRESSIVE_RESOLUTION",
        "SHORTGS_SEED",
    ]
    shortgs_lines = []
    for name in shortgs_vars:
        value = os.environ.get(name, "")
        if value:
            shortgs_lines.append(f"export {name}={shlex.quote(value)}")
        else:
            shortgs_lines.append(f"unset {name}")
    shortgs_env_block = "\n".join(shortgs_lines)

    script = f"""#!/bin/bash
set -euo pipefail
{module_block}
echo "[train] host=$(hostname) started=$(date -Is) stage={stage}"
module -t list 2>&1 || true
nvidia-smi || true

ROOT={shlex.quote(remote_root)}
DATASET={shlex.quote(dataset)}
STAGE={shlex.quote(stage)}
REUSE_ENV={1 if reuse_env else 0}
REPO_DIR={shlex.quote(repo_dir)}
REPO_URL={shlex.quote(repo_url)}
REPO_BRANCH={shlex.quote(repo_branch)}
ENV_PREFIX={shlex.quote(env_prefix)}
BACKEND_PIP={shlex.quote(backend_pip)}
CONVERTER={shlex.quote(converter_script)}
PREFLIGHT={shlex.quote(preflight_script)}
METRICS_COLLECTOR={shlex.quote(metrics_collector_script)}
METRICS_PLOTTER={shlex.quote(metrics_plotter_script)}
SHORTGS_PATCHES={shlex.quote(shortgs_patches_script)}
METRICS_DATASET_LABEL={shlex.quote(dataset_label_for_metrics)}
RUN_TAG={shlex.quote(run_tag)}
USE_FASTERGS_ADAM_TOGGLE={1 if use_fastergs_adam else 0}

# Shorter-Splatting opt-in flags forwarded from the local invocation.
# Empty means the technique is off. The vendored fork doesn't read these
# yet; once it's patched, train.py can pick them up from env.
{shortgs_env_block}

SCENE="$ROOT/experiments/faster-gs/datasets/$DATASET"
RUN_DIR="$ROOT/outputs/faster-gs/$RUN_TAG"
SPLAT_OUT="$ROOT/outputs/$RUN_TAG.splat"
METRICS_DIR="$RUN_DIR/metrics"
TRAIN_LOG="$RUN_DIR/train_stdout.log"

ensure_conda() {{
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
}}

ensure_dataset_layout() {{
  if [ ! -d "$SCENE/images" ] || [ ! -d "$SCENE/sparse/0" ]; then
    echo "[error] Missing prepared dataset layout at: $SCENE" >&2
    exit 22
  fi
}}

run_preflight() {{
  if [ -f "$PREFLIGHT" ]; then
    python "$PREFLIGHT" "$SCENE"
  fi
}}

ensure_repo() {{
  mkdir -p "$ROOT/src" "$ROOT/envs" "$ROOT/outputs/faster-gs"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --recursive "$REPO_URL" "$REPO_DIR"
  fi
  cd "$REPO_DIR"
  git fetch --all
  git checkout "$REPO_BRANCH"
  # Our shortgs patcher may have edited train.py on a previous run. git pull
  # --ff-only fails on a dirty tree, so reset tracked files first. New files
  # we don't control (if any) are left alone.
  git checkout -- train.py 2>/dev/null || true
  git pull --ff-only
  git submodule update --init --recursive
}}

create_or_reuse_env() {{
  if [ "$STAGE" = "setup" ]; then
    if [ -d "$ENV_PREFIX" ] && [ "$REUSE_ENV" -eq 0 ]; then
      echo "[setup] Removing existing env for clean rebuild: $ENV_PREFIX"
      rm -rf "$ENV_PREFIX"
    fi

    if [ ! -d "$ENV_PREFIX" ]; then
      echo "[setup] Creating clean env at $ENV_PREFIX"
      echo "[setup] Pinned stack: python={pinned_python} pytorch={pinned_torch} torchvision={pinned_torchvision} torchaudio={pinned_torchaudio} pytorch-cuda={pinned_pytorch_cuda}"
      conda create -y -p "$ENV_PREFIX" -c conda-forge \
        python={pinned_python} \
        pip setuptools wheel ninja

      TORCH_SPEC="pytorch"
      VISION_SPEC="torchvision"
      AUDIO_SPEC="torchaudio"
      PIP_TORCH_SPEC="torch"
      PIP_VISION_SPEC="torchvision"
      PIP_AUDIO_SPEC="torchaudio"
      if [ "{pinned_torch}" != "auto" ]; then
        TORCH_SPEC="pytorch={pinned_torch}"
        PIP_TORCH_SPEC="torch=={pinned_torch}"
      fi
      if [ "{pinned_torchvision}" != "auto" ]; then
        VISION_SPEC="torchvision={pinned_torchvision}"
        PIP_VISION_SPEC="torchvision=={pinned_torchvision}"
      fi
      if [ "{pinned_torchaudio}" != "auto" ]; then
        AUDIO_SPEC="torchaudio={pinned_torchaudio}"
        PIP_AUDIO_SPEC="torchaudio=={pinned_torchaudio}"
      fi

      echo "[setup] Installing torch stack with CUDA target {pinned_pytorch_cuda}"
      if ! conda install -y -p "$ENV_PREFIX" -c pytorch -c nvidia -c conda-forge \
        "$TORCH_SPEC" "$VISION_SPEC" "$AUDIO_SPEC" "pytorch-cuda={pinned_pytorch_cuda}"; then
        echo "[warn] Conda torch install failed for pytorch-cuda={pinned_pytorch_cuda}; falling back to pip cu128 wheels"
        conda run -p "$ENV_PREFIX" python -m pip install --upgrade pip
        conda run -p "$ENV_PREFIX" python -m pip install \
          --index-url https://download.pytorch.org/whl/cu128 \
          "$PIP_TORCH_SPEC" "$PIP_VISION_SPEC" "$PIP_AUDIO_SPEC" || {{
            echo "[error] Could not install torch from pip cu128 wheels." >&2
            exit 33
          }}
      fi
    else
      echo "[setup] Reusing existing env: $ENV_PREFIX"
    fi
  else
    if [ ! -d "$ENV_PREFIX" ]; then
      echo "[error] Stage '$STAGE' requires an existing env at $ENV_PREFIX. Run --stage setup first." >&2
      exit 34
    fi
  fi
}}

activate_env() {{
  conda activate "$ENV_PREFIX"
}}

print_runtime_info() {{
  python --version
  echo "[stage:$STAGE] nvcc=$(command -v nvcc || true)"
  nvcc --version || true
  nvidia-smi || true
}}

validate_torch_cuda() {{
python - <<'PY'
import sys
import torch

print(f"[validate] torch.__version__={{torch.__version__}}")
print(f"[validate] torch.version.cuda={{torch.version.cuda}}")
print(f"[validate] torch.cuda.is_available={{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"[validate] torch.cuda.device_count={{torch.cuda.device_count()}}")
    print(f"[validate] torch.cuda.device_name={{torch.cuda.get_device_name(0)}}")
    print(f"[validate] torch.cuda.capability={{torch.cuda.get_device_capability(0)}}")
else:
    print("[error] torch.cuda.is_available() is False.", file=sys.stderr)
    sys.exit(31)
PY
}}

install_runtime_python_deps() {{
  python -m pip install --upgrade pip
  if [ -f requirements.txt ]; then
    TMP_REQ="$(mktemp)"
    awk 'NF && $0 !~ /^#/ && $0 !~ /^([.][/])?submodules[/]/' requirements.txt > "$TMP_REQ"
    if [ -s "$TMP_REQ" ]; then
      python -m pip install -r "$TMP_REQ"
    fi
    rm -f "$TMP_REQ"
  else
    python -m pip install opencv-python joblib plyfile tqdm
  fi
}}

clean_extension_artifacts() {{
  echo "[stage:$STAGE] Cleaning extension artifacts"
  rm -rf submodules/diff-gaussian-rasterization/build \
         submodules/simple-knn/build \
         submodules/fused-ssim/build
  rm -rf "$HOME/.cache/torch_extensions"
  find submodules -type f -name "*.so" -delete || true
  find submodules -type f -name "*.o" -delete || true
  find submodules -type f -name "*.obj" -delete || true
}}

build_extensions() {{
  export TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list}
  export MAX_JOBS="${{SLURM_CPUS_PER_TASK:-2}}"
  export CC=gcc
  export CXX=g++

  echo "[stage:$STAGE] Building local CUDA extensions from source"
  python -m pip install --no-build-isolation --no-cache-dir ./submodules/diff-gaussian-rasterization
  python -m pip install --no-build-isolation --no-cache-dir ./submodules/simple-knn
  python -m pip install --no-build-isolation --no-cache-dir ./submodules/fused-ssim
}}

validate_imports() {{
python - <<'PY'
import importlib.util, sys
mods = ["diff_gaussian_rasterization", "simple_knn", "fused_ssim", "FasterGSCudaBackend"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    print(f"[error] Missing required modules: {{missing}}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
}}

ensure_backend_module() {{
  if ! python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("FasterGSCudaBackend") else 1)
PY
  then
    echo "[stage:$STAGE] Installing FasterGSCudaBackend"
    pip install "$BACKEND_PIP" --no-build-isolation
  fi
}}

run_training() {{
  mkdir -p "$RUN_DIR" "$METRICS_DIR"
  echo "[stage:$STAGE] scene=$SCENE"
  echo "[stage:$STAGE] run_dir=$RUN_DIR"
  echo "[stage:$STAGE] iterations={iterations}"

  # Kernel toggles for the vendored FasterGS Inria fork.
  #
  # The fork ships with USE_FASTERGS_RASTERIZER = False and
  # USE_FASTERGS_ADAM = False. Our experiments expose two independent
  # knobs:
  #
  #   - Custom rasterizer: DISABLED unconditionally. Phase 1 (Apr 17)
  #     confirmed it fails on both L4 (sm_89) and B200 (sm_100) with
  #     cudaErrorInvalidConfiguration at iter 0. Recompile doesn't fix
  #     it; the bug is in the kernel launch parameters themselves.
  #     The B200-fallback sed and the enable sed are both commented
  #     out so we always run the stock Inria rasterizer. Re-enable
  #     upstream if/when FasterGS patches the launch config.
  #
  #   - Fused Adam: controlled by USE_FASTERGS_ADAM_TOGGLE (plumbed in
  #     from --use-fastergs-adam via fastergs_pipeline / run_matrix).
  #     0 = stock torch.optim.Adam (vendored default; safe on all
  #     architectures). 1 = flip the source line to True and train
  #     with FasterGS's fused Adam kernel. Verified working on B200
  #     (SLURM 30208346: 2000 iters, PSNR 30.46, 16s wall) via the
  #     multi-arch "8.9;10.0" FasterGSCudaBackend wheel, which means
  #     it should also work on L4, but we only sanity-check that in
  #     the Apr 17 smoke tests.
  if [ -f gaussian_renderer/__init__.py ]; then
    # B200 full-fallback:     sed -i 's/^USE_FASTERGS_RASTERIZER = True/USE_FASTERGS_RASTERIZER = False/' gaussian_renderer/__init__.py || true
    # Enable custom rasterizer (broken on sm_89 + sm_100 as of Apr 17):
    # sed -i 's/^USE_FASTERGS_RASTERIZER = False/USE_FASTERGS_RASTERIZER = True/' gaussian_renderer/__init__.py || true
    if grep -q "^USE_FASTERGS_RASTERIZER = True" gaussian_renderer/__init__.py; then
      echo "[stage:$STAGE] rasterizer_mode=fastergs"
    else
      echo "[stage:$STAGE] rasterizer_mode=standard"
    fi
  fi
  if [ -f scene/gaussian_model.py ]; then
    # Always reset to False first so previous runs' flips don't leak
    # across matrix rows. Then apply the runtime toggle: when it's on,
    # flip False→True so the next `import FusedAdam` path activates.
    sed -i 's/^USE_FASTERGS_ADAM = True/USE_FASTERGS_ADAM = False/' scene/gaussian_model.py || true
    if [ "$USE_FASTERGS_ADAM_TOGGLE" = "1" ]; then
      sed -i 's/^USE_FASTERGS_ADAM = False/USE_FASTERGS_ADAM = True/' scene/gaussian_model.py || true
    fi
    if grep -q "^USE_FASTERGS_ADAM = True" scene/gaussian_model.py; then
      echo "[stage:$STAGE] optimizer_mode=fastergs_adam"
    else
      echo "[stage:$STAGE] optimizer_mode=adam"
    fi
  fi

  # Apply Shorter-Splatting paper patches to the vendored Inria fork BEFORE
  # training starts. Patches read SHORTGS_* env vars at runtime so they stay
  # no-ops unless the user set flags. Idempotent - safe to run every job.
  # If the patcher is missing or errors, we log a warning and proceed with
  # stock training so a bad patch never breaks a baseline run.
  if [ -f "$SHORTGS_PATCHES" ]; then
    python "$SHORTGS_PATCHES" --repo-dir "$REPO_DIR" || echo "[warn] shortgs patches failed, continuing with stock train.py"
  else
    echo "[stage:$STAGE] shortgs patches script not present ($SHORTGS_PATCHES); stock train.py"
  fi

  # Record when training starts so metrics_collector can derive wall_seconds
  # from ply file mtimes after the run.
  TRAIN_START_EPOCH=$(date +%s)
  echo "[stage:$STAGE] train_start_epoch=$TRAIN_START_EPOCH"

  # Dense eval + save schedule so the Reports page gets smooth curves.
  # Two frequencies:
  #   - EVAL_ITERS (every 500): drives --test_iterations + --save_iterations.
  #     PSNR/SSIM/LPIPS passes cost real time, so keep these sparser. The
  #     saved PLYs also give us gaussian-count + splats-per-frame + wall-
  #     seconds samples every 500 iters (these come from the saved PLY
  #     file's vertex count + mtime - the Inria trainer doesn't print
  #     them in the tqdm line, so that's the cheapest source we have).
  #   - The dense loss curve comes from the tqdm progress lines the
  #     trainer prints continuously, which metrics_collector downsamples
  #     to every 100 iters (PROGRESS_STRIDE).
  # Net effect: ~100 loss points and ~20 PSNR / gaussian-count points on
  # a 10k-iter run, without meaningfully slowing training down.
  TOTAL_ITERS={iterations}
  EVAL_STRIDE=500
  if [ "$TOTAL_ITERS" -lt "$EVAL_STRIDE" ]; then
    # Very short runs still get at least one eval + save at the end.
    EVAL_STRIDE=$TOTAL_ITERS
  fi
  EVAL_ITERS=""
  ITER_N=$EVAL_STRIDE
  while [ "$ITER_N" -le "$TOTAL_ITERS" ]; do
    EVAL_ITERS="$EVAL_ITERS $ITER_N"
    ITER_N=$(( ITER_N + EVAL_STRIDE ))
  done
  # Always include the final iteration so we end with a clean datapoint
  # even when TOTAL_ITERS isn't a clean multiple of EVAL_STRIDE.
  case " $EVAL_ITERS " in
    *" $TOTAL_ITERS "*) : ;;
    *) EVAL_ITERS="$EVAL_ITERS $TOTAL_ITERS" ;;
  esac
  echo "[stage:$STAGE] eval_iters=$EVAL_ITERS"

  # Tee stdout to a log file so metrics_collector can parse loss/PSNR lines
  # after the fact. stderr stays on the SLURM err stream.
  # --test_iterations drives [ITER N] PSNR eval lines in the log.
  # --save_iterations drives point_cloud/iteration_N/point_cloud.ply writes
  # (which metrics_collector uses to count gaussians + stamp wall_seconds).
  python train.py \
    -s "$SCENE" \
    -m "$RUN_DIR" \
    --optimizer_type default \
    --iterations {iterations} \
    --test_iterations $EVAL_ITERS \
    --save_iterations $EVAL_ITERS \
    --disable_viewer 2>&1 | tee "$TRAIN_LOG"

  # -v (version sort) so "iteration_10000" sorts AFTER "iteration_9000".
  # Default lexicographic sort puts "10000" before "9000" and we'd pick the
  # wrong checkpoint when saving is dense (every 1000 iters).
  PLY_PATH="$(ls -1v "$RUN_DIR"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -n 1 || true)"
  if [ -z "$PLY_PATH" ]; then
    echo "[error] point_cloud.ply not found under $RUN_DIR/point_cloud" >&2
    exit 24
  fi
  echo "[stage:$STAGE] ply=$PLY_PATH"

  # The trainer writes a .ply, but the frontend viewer loads .splat.
  # Convert right after training (on HPG) so we only need to scp the smaller
  # .splat back down later.
  if [ -f "$CONVERTER" ]; then
    python "$CONVERTER" "$PLY_PATH"
    SPLAT_PATH="${{PLY_PATH%.ply}}.splat"
    if [ -f "$SPLAT_PATH" ]; then
      cp -f "$SPLAT_PATH" "$SPLAT_OUT"
      echo "[stage:$STAGE] splat=$SPLAT_OUT"
    else
      echo "[warn] Converter did not create expected splat file: $SPLAT_PATH" >&2
    fi
  else
    echo "[warn] Converter script missing: $CONVERTER" >&2
  fi

  # Collect metrics from the tee'd log + saved PLYs. The collector is tolerant
  # of missing deps (scikit-image / lpips); if they aren't installed in this
  # env it just records loss/PSNR/gaussian counts and leaves SSIM/LPIPS null.
  # We don't fail training if this step errors - worst case the frontend just
  # shows "no metrics for this run".
  if [ -f "$METRICS_COLLECTOR" ]; then
    SEED_ARG=""
    if [ -n "${{SHORTGS_SEED:-}}" ]; then SEED_ARG="--seed $SHORTGS_SEED"; fi
    python "$METRICS_COLLECTOR" \
      --run-dir "$RUN_DIR" \
      --log-file "$TRAIN_LOG" \
      --out-dir "$METRICS_DIR" \
      --backend fastergs \
      --dataset "$METRICS_DATASET_LABEL" \
      --run-tag "$RUN_TAG" \
      --iterations {iterations} \
      --partition "${{SLURM_JOB_PARTITION:-}}" \
      --start-epoch "$TRAIN_START_EPOCH" \
      $SEED_ARG || echo "[warn] metrics_collector returned non-zero"

    # Append shortgs config to the summary so compare.py can group by it.
    # Uses python -c to do the in-place json edit - keeps the logic out
    # of fragile bash string manipulation.
    python -c "
import json, os, sys
p = os.path.join(r'''$METRICS_DIR''', 'metrics_summary.json')
if not os.path.isfile(p):
    sys.exit(0)
with open(p) as f:
    d = json.load(f)
d['shortgs'] = {{
    'scale_reset_every': int(os.environ.get('SHORTGS_SCALE_RESET_EVERY') or 0),
    'scale_reset_factor': float(os.environ.get('SHORTGS_SCALE_RESET_FACTOR') or 1.0),
    'entropy_weight': float(os.environ.get('SHORTGS_ENTROPY_WEIGHT') or 0.0),
    'progressive_resolution': os.environ.get('SHORTGS_PROGRESSIVE_RESOLUTION') or '',
}}
with open(p, 'w') as f:
    json.dump(d, f, indent=2)
" || echo "[warn] summary shortgs annotation failed"
  else
    echo "[warn] metrics collector script missing: $METRICS_COLLECTOR"
  fi

  # Render PNGs from the jsonl. matplotlib uses Agg backend (headless) so this
  # runs fine on a GPU compute node with no display.
  if [ -f "$METRICS_PLOTTER" ]; then
    python "$METRICS_PLOTTER" \
      --metrics-dir "$METRICS_DIR" \
      --title-prefix "$METRICS_DATASET_LABEL $RUN_TAG" || echo "[warn] metrics_plotter returned non-zero"
  else
    echo "[warn] metrics plotter script missing: $METRICS_PLOTTER"
  fi
}}

ensure_conda
ensure_dataset_layout
if [ "$STAGE" != "validate" ]; then
  run_preflight
fi
if [ "$STAGE" = "setup" ] || [ "$STAGE" = "smoke" ] || [ "$STAGE" = "train" ]; then
  ensure_repo
fi

case "$STAGE" in
  setup)
    create_or_reuse_env
    activate_env
    print_runtime_info
    validate_torch_cuda
    install_runtime_python_deps
    clean_extension_artifacts
    build_extensions
    ensure_backend_module
    validate_imports
    echo "[ok] setup stage complete"
    ;;
  validate)
    create_or_reuse_env
    activate_env
    print_runtime_info
    validate_torch_cuda
    validate_imports
    echo "[ok] validate stage complete"
    ;;
  smoke)
    create_or_reuse_env
    activate_env
    print_runtime_info
    validate_torch_cuda
    validate_imports
    run_training
    echo "[ok] smoke stage complete"
    ;;
  train)
    create_or_reuse_env
    activate_env
    print_runtime_info
    validate_torch_cuda
    validate_imports
    run_training
    echo "[ok] train stage complete"
    ;;
  *)
    echo "[error] Unknown stage: $STAGE" >&2
    exit 35
    ;;
esac

echo "[train] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + script + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="SLURM training stage for gs_final Faster-GS: train, convert .ply->.splat, fetch artifacts."
    )
    parser.add_argument("dataset", help="Dataset name (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg)")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote gs_final root")
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR, help="Remote Faster-GS repo dir")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Faster-GS repo URL")
    parser.add_argument("--repo-branch", default="main", help="Faster-GS branch/tag")
    parser.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX, help="Remote env path")
    parser.add_argument("--modules", default=DEFAULT_MODULES, help="Space-separated modules to load")
    parser.add_argument("--stage", choices=["setup", "validate", "smoke", "train"], default="train", help="Execution stage")
    parser.add_argument("--reuse-env", action="store_true", help="Reuse an existing env for this stage")
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations (default: 25 for smoke, 1000 for train)")
    parser.add_argument("--pinned-python", default=DEFAULT_PINNED_PYTHON, help="Pinned Python version")
    parser.add_argument("--pinned-torch", default=DEFAULT_PINNED_TORCH, help="Pinned PyTorch version, or 'auto'")
    parser.add_argument("--pinned-torchvision", default=DEFAULT_PINNED_TORCHVISION, help="Pinned torchvision version, or 'auto'")
    parser.add_argument("--pinned-torchaudio", default=DEFAULT_PINNED_TORCHAUDIO, help="Pinned torchaudio version, or 'auto'")
    parser.add_argument("--pinned-pytorch-cuda", default=DEFAULT_PINNED_PYTORCH_CUDA, help="Pinned pytorch-cuda version")
    parser.add_argument("--torch-cuda-arch-list", default=DEFAULT_TORCH_CUDA_ARCH_LIST, help="TORCH_CUDA_ARCH_LIST for extension builds")
    parser.add_argument("--backend-pip", default=DEFAULT_BACKEND_PIP, help="FasterGSCudaBackend pip source")
    parser.add_argument("--converter-script", default=DEFAULT_CONVERTER_SCRIPT, help="Remote converter.py path")
    parser.add_argument("--preflight-script", default=DEFAULT_PREFLIGHT_SCRIPT, help="Remote preflight script path")
    parser.add_argument("--metrics-collector-script", default=DEFAULT_METRICS_COLLECTOR_SCRIPT, help="Remote metrics_collector.py path")
    parser.add_argument("--metrics-plotter-script", default=DEFAULT_METRICS_PLOTTER_SCRIPT, help="Remote metrics_plotter.py path")
    parser.add_argument("--shortgs-patches-script", default=DEFAULT_SHORTGS_PATCHES_SCRIPT, help="Remote shortgs_apply_patches.py path")
    # Opt into FasterGS's fused Adam kernel. Default off because the
    # vendored fork ships with USE_FASTERGS_ADAM=False. On both L4 and
    # B200 the multi-arch FasterGSCudaBackend wheel supports it; the
    # custom rasterizer remains off either way (broken on sm_89+sm_100
    # as of Apr 17). The matrix config name `fastergs_adam` sets this.
    parser.add_argument("--use-fastergs-adam", action=argparse.BooleanOptionalAction, default=False,
                        help="Flip USE_FASTERGS_ADAM=True in the vendored fork before training.")
    # Training backend. fastergs (default) keeps every path on this file
    # unchanged: conda env, Inria fork clone, sed toggles, etc. opensplat
    # invokes a prebuilt C++ binary under /blue/cis4914/joshuabowman/gs_final/src/OpenSplat/
    # and skips the fastergs env entirely (only uses torch env for libtorch
    # runtime linkage).
    parser.add_argument("--backend", choices=["fastergs", "opensplat"], default="fastergs",
                        help="Which trainer to invoke inside the SLURM job.")
    parser.add_argument("--opensplat-root", default=DEFAULT_OPENSPLAT_ROOT,
                        help="Remote OpenSplat install dir (contains build/ and build_b200/ binaries). "
                             "Defaults to FASTERGS_OPENSPLAT_ROOT env var or $REMOTE_ROOT/src/OpenSplat.")
    parser.add_argument("--run-label", default="",
                        help="Short tag injected into the run_tag (e.g. 's1-baseline' or 's1-shortgs-sr-ent'). "
                             "When set, run_tag becomes '<dataset>_<label>_<stage>_<timestamp>'. "
                             "Empty falls back to the legacy '<dataset>_<stage>_<timestamp>' format.")
    parser.add_argument("--metrics-local-dir", default=None, help="Local dir root for fetched metrics (default: backend/datasets/<dataset>/metrics)")
    parser.add_argument("--metrics-dataset-label", default=None, help="Dataset label recorded in metrics_summary.json (default: --dataset)")
    parser.add_argument("--slurm-time", default="06:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-turin", help="SLURM partition (hpg-turin = L4/sm_89; FasterGS custom rasterizer currently disabled on both sm_89 and sm_100)")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=1, help="SLURM CPUs")
    parser.add_argument("--slurm-mem", default="12G", help="SLURM memory")
    parser.add_argument("--slurm-gpus", type=int, default=1, help="SLURM GPU count")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    parser.add_argument("--fetch-local-dir", default=None, help="Local dir for fetched .splat/.ply (default: backend/hipergator/gs_final)")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()
    if args.stage in {"validate", "smoke", "train"} and not args.reuse_env:
        raise ValueError(f"--reuse-env is required for --stage {args.stage}")
    iterations = args.iterations if args.iterations is not None else (25 if args.stage == "smoke" else 1000)

    backend_dir = Path(__file__).resolve().parent.parent
    local_logs = backend_dir / "build_logs"
    local_logs.mkdir(parents=True, exist_ok=True)
    fetch_dir = (
        Path(args.fetch_local_dir).expanduser().resolve()
        if args.fetch_local_dir
        else (backend_dir / "hipergator" / "gs_final").resolve()
    )
    fetch_dir.mkdir(parents=True, exist_ok=True)

    root = args.remote_root.rstrip("/")
    remote_logs_dir = f"{root}/logs"
    # Validate run-label against the same regex dataset names use, so the
    # resulting run_tag stays safe in paths and URL segments. Allow letters,
    # digits, '_', '-', '.'. Empty is fine - we fall back to the legacy
    # "<dataset>_<stage>_<timestamp>" shape.
    if args.run_label and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_label):
        raise SystemExit(f"invalid --run-label: {args.run_label!r} (allowed: letters, digits, _ . -)")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_label:
        run_tag = f"{args.dataset}_{args.run_label}_{args.stage}_{stamp}"
    else:
        run_tag = f"{args.dataset}_{args.stage}_{stamp}"
    remote_run_dir = f"{root}/outputs/faster-gs/{run_tag}"
    remote_splat = f"{root}/outputs/{run_tag}.splat"
    remote_latest_ply = f"{remote_run_dir}/latest_point_cloud.ply"
    remote_scene_dir = f"{root}/experiments/faster-gs/datasets/{args.dataset}"

    tag = f"gsf_train_{run_tag}"
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
    scp_base = ["scp", "-P", str(args.port), *(["-i", args.identity_file] if args.identity_file else []), *ssh_opts]

    log(
        f"Starting gs_final {args.stage} stage dataset={args.dataset} "
        f"(partition={args.slurm_partition}, gpus={args.slurm_gpus})"
    )
    run_cmd(
        ssh + [bash_lc(f"mkdir -p {shlex.quote(root)} {shlex.quote(remote_logs_dir)} {shlex.quote(root + '/outputs/faster-gs')}")],
        dry_run=args.dry_run,
    )
    local_converter = backend_dir / "scripts" / "converter.py"
    local_preflight = backend_dir / "scripts" / "fastergs_preflight.py"
    if args.converter_script == DEFAULT_CONVERTER_SCRIPT:
        sync_remote_file(
            ssh=ssh,
            scp_base=scp_base,
            remote_host=args.remote,
            local_source=local_converter,
            remote_target=args.converter_script,
            label="converter script",
            dry_run=args.dry_run,
        )
    if args.preflight_script == DEFAULT_PREFLIGHT_SCRIPT:
        sync_remote_file(
            ssh=ssh,
            scp_base=scp_base,
            remote_host=args.remote,
            local_source=local_preflight,
            remote_target=args.preflight_script,
            label="preflight script",
            dry_run=args.dry_run,
        )

    # Upload metrics helper scripts alongside converter/preflight so the
    # SLURM job has local copies rather than importing anything over the wire.
    local_metrics_collector = backend_dir / "scripts" / "metrics_collector.py"
    local_metrics_plotter = backend_dir / "scripts" / "metrics_plotter.py"
    if args.metrics_collector_script == DEFAULT_METRICS_COLLECTOR_SCRIPT and local_metrics_collector.is_file():
        sync_remote_file(
            ssh=ssh,
            scp_base=scp_base,
            remote_host=args.remote,
            local_source=local_metrics_collector,
            remote_target=args.metrics_collector_script,
            label="metrics collector script",
            dry_run=args.dry_run,
        )
    if args.metrics_plotter_script == DEFAULT_METRICS_PLOTTER_SCRIPT and local_metrics_plotter.is_file():
        sync_remote_file(
            ssh=ssh,
            scp_base=scp_base,
            remote_host=args.remote,
            local_source=local_metrics_plotter,
            remote_target=args.metrics_plotter_script,
            label="metrics plotter script",
            dry_run=args.dry_run,
        )

    # Shorter-Splatting paper patcher. Uploaded every run so updates to the
    # local patch script propagate to HPG without a manual sync step.
    local_shortgs_patches = backend_dir / "scripts" / "shortgs_apply_patches.py"
    if args.shortgs_patches_script == DEFAULT_SHORTGS_PATCHES_SCRIPT and local_shortgs_patches.is_file():
        sync_remote_file(
            ssh=ssh,
            scp_base=scp_base,
            remote_host=args.remote,
            local_source=local_shortgs_patches,
            remote_target=args.shortgs_patches_script,
            label="shortgs patches script",
            dry_run=args.dry_run,
        )
    try:
        check_remote_dataset(ssh=ssh, scene_dir=remote_scene_dir, preflight_script=args.preflight_script, dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Remote dataset check failed. Prepare the dataset first with:\n"
            "python scripts/hpg_gs_final_prepare.py can --remote hpg --slurm-account cis4914 "
            "--slurm-partition hpg-default --slurm-cpus 1 --slurm-mem 8G --slurm-time 01:00:00"
        ) from exc

    if args.backend == "opensplat":
        # OpenSplat needs none of the Inria / conda plumbing: just cuda +
        # gcc + opencv modules and libtorch runtime from the existing env.
        sbatch_text = build_opensplat_sbatch_script(
            remote_root=root,
            dataset=args.dataset,
            stage=args.stage,
            iterations=iterations,
            env_prefix=args.env_prefix,
            opensplat_root=args.opensplat_root,
            converter_script=args.converter_script,
            preflight_script=args.preflight_script,
            metrics_collector_script=args.metrics_collector_script,
            metrics_plotter_script=args.metrics_plotter_script,
            dataset_label_for_metrics=(args.metrics_dataset_label or args.dataset),
            run_tag=run_tag,
            slurm_time=args.slurm_time,
            slurm_cpus=args.slurm_cpus,
            slurm_mem=args.slurm_mem,
            slurm_gpus=args.slurm_gpus,
            slurm_partition=args.slurm_partition,
            slurm_account=args.slurm_account,
            out_path=remote_out,
            err_path=remote_err,
        )
    else:
        sbatch_text = build_sbatch_script(
            remote_root=root,
            dataset=args.dataset,
            repo_dir=args.repo_dir,
            repo_url=args.repo_url,
            repo_branch=args.repo_branch,
            env_prefix=args.env_prefix,
            modules=args.modules,
            stage=args.stage,
            reuse_env=args.reuse_env,
            iterations=iterations,
            pinned_python=args.pinned_python,
            pinned_torch=args.pinned_torch,
            pinned_torchvision=args.pinned_torchvision,
            pinned_torchaudio=args.pinned_torchaudio,
            pinned_pytorch_cuda=args.pinned_pytorch_cuda,
            torch_cuda_arch_list=args.torch_cuda_arch_list,
            backend_pip=args.backend_pip,
            converter_script=args.converter_script,
            preflight_script=args.preflight_script,
            metrics_collector_script=args.metrics_collector_script,
            metrics_plotter_script=args.metrics_plotter_script,
            shortgs_patches_script=args.shortgs_patches_script,
            dataset_label_for_metrics=(args.metrics_dataset_label or args.dataset),
            run_tag=run_tag,
            use_fastergs_adam=args.use_fastergs_adam,
            slurm_time=args.slurm_time,
            slurm_cpus=args.slurm_cpus,
            slurm_mem=args.slurm_mem,
            slurm_gpus=args.slurm_gpus,
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
            scp_base + [str(temp_path), f"{args.remote}:{remote_sbatch}"],
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
        log(f"[ok] Stage {args.stage} submission complete (--no-wait)")
        log(f"Track with: squeue -j {job_id}")
        log(f"Remote logs: {remote_out} {remote_err}")
        return

    final_state = poll_slurm_job(ssh=ssh, job_id=job_id, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    log(f"SLURM job {job_id} final state: {final_state}")

    run_cmd(
        scp_base + [f"{args.remote}:{remote_out}", str(local_out)],
        dry_run=args.dry_run,
    )
    run_cmd(
        scp_base + [f"{args.remote}:{remote_err}", str(local_err)],
        dry_run=args.dry_run,
    )
    log(f"Local logs: {local_out} {local_err}")

    if not args.dry_run and final_state.startswith("COMPLETED") and args.stage in {"smoke", "train"}:
        local_ply_path = fetch_dir / f"{run_tag}.ply"
        local_splat_path = fetch_dir / f"{run_tag}.splat"
        splat_fetched = False
        ply_fetched = False
        run_cmd(
            ssh
            + [
                bash_lc(
                    f'PLY_PATH="$(ls -1v {shlex.quote(remote_run_dir)}/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -n 1 || true)"; '
                    + f'if [ -n "$PLY_PATH" ]; then cp -f "$PLY_PATH" {shlex.quote(remote_latest_ply)}; fi'
                )
            ]
        )
        splat_exists = run_cmd_capture(
            ssh + [bash_lc(f'if [ -f {shlex.quote(remote_splat)} ]; then echo yes; else echo no; fi')]
        )
        if splat_exists.strip() == "yes":
            run_cmd(
                scp_base + [f"{args.remote}:{remote_splat}", str(local_splat_path)]
            )
            splat_fetched = True
        else:
            log("[warn] No .splat artifact found to fetch.")

        ply_exists = run_cmd_capture(
            ssh + [bash_lc(f'if [ -f {shlex.quote(remote_latest_ply)} ]; then echo yes; else echo no; fi')]
        )
        if ply_exists.strip() == "yes":
            run_cmd(
                scp_base + [f"{args.remote}:{remote_latest_ply}", str(local_ply_path)]
            )
            ply_fetched = True
        else:
            log("[warn] No .ply artifact found to fetch.")

        if ply_fetched and not splat_fetched and local_converter.is_file():
            run_cmd([sys.executable, str(local_converter), str(local_ply_path)], dry_run=args.dry_run)
            if local_splat_path.is_file():
                log(f"[ok] Created local .splat from fetched .ply: {local_splat_path}")
                splat_fetched = True
        log(f"[ok] Fetched available {args.stage} artifacts to: {fetch_dir}")

        # Fetch the metrics directory. Lands under either the explicit
        # --metrics-local-dir or backend/datasets/<dataset>/metrics/<run_tag>/.
        # The remote metrics dir may be missing if collector failed mid-run;
        # we check existence and skip cleanly in that case.
        if args.metrics_local_dir:
            metrics_local_root = Path(args.metrics_local_dir).expanduser().resolve()
        else:
            metrics_local_root = (backend_dir / "datasets" / args.dataset / "metrics").resolve()
        metrics_local_dir = metrics_local_root / run_tag
        metrics_local_dir.mkdir(parents=True, exist_ok=True)

        remote_metrics_dir = f"{remote_run_dir}/metrics"
        metrics_exists = run_cmd_capture(
            ssh + [bash_lc(f'if [ -d {shlex.quote(remote_metrics_dir)} ]; then echo yes; else echo no; fi')]
        )
        if metrics_exists.strip() == "yes":
            # rsync whole directory so we pick up jsonl + summary + PNGs in one go.
            rsync_ssh = " ".join(shlex.quote(p) for p in (["ssh", "-p", str(args.port)] + (["-i", args.identity_file] if args.identity_file else []) + ssh_opts))
            run_cmd(
                [
                    "rsync",
                    "-az",
                    "-e",
                    rsync_ssh,
                    f"{args.remote}:{remote_metrics_dir}/",
                    str(metrics_local_dir) + "/",
                ],
                dry_run=args.dry_run,
            )
            log(f"[ok] Fetched metrics to: {metrics_local_dir}")
        else:
            log(f"[warn] No metrics dir on remote ({remote_metrics_dir}); skipping metrics fetch.")

    if not args.dry_run and not final_state.startswith("COMPLETED"):
        print_log_tail(local_err, f"{args.stage} stderr log")
        print_log_tail(local_out, f"{args.stage} stdout log")
        raise RuntimeError(f"Train job ended in state {final_state}. Check logs: {local_out} {local_err}")
    log(f"[ok] gs_final {args.stage} stage completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
