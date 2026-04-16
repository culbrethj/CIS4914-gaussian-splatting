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

DEFAULT_REMOTE_ROOT = "/blue/cis4914/joshuabowman/gs_final"
DEFAULT_REPO_DIR = "/blue/cis4914/joshuabowman/gs_final/src/fastergs_inria"
DEFAULT_REPO_URL = "https://github.com/fhahlbohm/gaussian-splatting.git"
DEFAULT_ENV_PREFIX = "/blue/cis4914/joshuabowman/gs_final/envs/fastergs_cuda128"
DEFAULT_MODULES = "git cmake gcc/12.2.0 conda/25.7.0 cuda/12.8.1"
DEFAULT_BACKEND_PIP = (
    "git+https://github.com/nerficg-project/faster-gaussian-splatting/#subdirectory=FasterGSCudaBackend"
)
DEFAULT_CONVERTER_SCRIPT = "/blue/cis4914/joshuabowman/gs_final/src/converter.py"
DEFAULT_PREFLIGHT_SCRIPT = "/blue/cis4914/joshuabowman/gs_final/src/fastergs_preflight.py"
DEFAULT_PINNED_PYTHON = "3.10.14"
DEFAULT_PINNED_TORCH = "auto"
DEFAULT_PINNED_TORCHVISION = "auto"
DEFAULT_PINNED_TORCHAUDIO = "auto"
DEFAULT_PINNED_PYTORCH_CUDA = "12.8"
DEFAULT_TORCH_CUDA_ARCH_LIST = "10.0"


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _format_cmd(cmd: list[str]) -> str:
    printable = " ".join(shlex.quote(part) for part in cmd)
    compact = " ".join(printable.replace("\n", " ").split())
    if len(compact) > 260:
        return compact[:257] + "..."
    return compact


def run_cmd(cmd: list[str], *, dry_run: bool = False):
    log(f"[cmd] {_format_cmd(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def run_cmd_capture(cmd: list[str], *, dry_run: bool = False, log_cmd: bool = True) -> str:
    if log_cmd:
        log(f"[cmd] {_format_cmd(cmd)}")
    if dry_run:
        return ""
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return proc.stdout.strip()


def bash_lc(command: str) -> str:
    return f"bash -lc {shlex.quote(command)}"


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
RUN_TAG={shlex.quote(run_tag)}

SCENE="$ROOT/experiments/faster-gs/datasets/$DATASET"
RUN_DIR="$ROOT/outputs/faster-gs/$RUN_TAG"
SPLAT_OUT="$ROOT/outputs/$RUN_TAG.splat"

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
  mkdir -p "$RUN_DIR"
  echo "[stage:$STAGE] scene=$SCENE"
  echo "[stage:$STAGE] run_dir=$RUN_DIR"
  echo "[stage:$STAGE] iterations={iterations}"

  # B200 compatibility fallback: use standard renderer path instead of FasterGSCudaBackend rasterizer.
  if [ -f gaussian_renderer/__init__.py ]; then
    sed -i 's/^USE_FASTERGS_RASTERIZER = True/USE_FASTERGS_RASTERIZER = False/' gaussian_renderer/__init__.py || true
    echo "[stage:$STAGE] rasterizer_mode=standard"
  fi
  if [ -f scene/gaussian_model.py ]; then
    sed -i 's/^USE_FASTERGS_ADAM = True/USE_FASTERGS_ADAM = False/' scene/gaussian_model.py || true
    echo "[stage:$STAGE] optimizer_mode=adam"
  fi

  python train.py \
    -s "$SCENE" \
    -m "$RUN_DIR" \
    --optimizer_type default \
    --iterations {iterations} \
    --disable_viewer

  PLY_PATH="$(ls -1 "$RUN_DIR"/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -n 1 || true)"
  if [ -z "$PLY_PATH" ]; then
    echo "[error] point_cloud.ply not found under $RUN_DIR/point_cloud" >&2
    exit 24
  fi
  echo "[stage:$STAGE] ply=$PLY_PATH"

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
    parser.add_argument("--slurm-time", default="06:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-b200", help="SLURM partition")
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
    run_tag = f"{args.dataset}_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    try:
        check_remote_dataset(ssh=ssh, scene_dir=remote_scene_dir, preflight_script=args.preflight_script, dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Remote dataset check failed. Prepare the dataset first with:\n"
            "python scripts/hpg_gs_final_prepare.py can --remote hpg --slurm-account cis4914 "
            "--slurm-partition hpg-default --slurm-cpus 1 --slurm-mem 8G --slurm-time 01:00:00"
        ) from exc

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
                    f'PLY_PATH="$(ls -1 {shlex.quote(remote_run_dir)}/point_cloud/iteration_*/point_cloud.ply 2>/dev/null | tail -n 1 || true)"; '
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
