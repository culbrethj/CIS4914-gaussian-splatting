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
DEFAULT_SOURCE_DIR = "/blue/cis4914/joshuabowman/src/OpenSplat"
DEFAULT_ENV_PREFIX = "/blue/cis4914/joshuabowman/conda/opensplat_clean"
DEFAULT_REPO_URL = "https://github.com/pierotofy/OpenSplat.git"
DEFAULT_MODULES = "git cmake gcc/12.2.0 cuda/12.4.1"


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def stage(name: str):
    log(f"==> {name}")
    return time.perf_counter()


def stage_done(name: str, start_time: float):
    elapsed = time.perf_counter() - start_time
    log(f"<== {name} complete ({elapsed:.1f}s)")


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
    remote_workdir: str,
    source_dir: str,
    env_prefix: str,
    repo_url: str,
    repo_branch: str,
    cpus: int,
    slurm_time: str,
    slurm_partition: str | None,
    slurm_account: str | None,
    slurm_mem: str,
    out_path: str,
    err_path: str,
    job_name: str,
    modules: str,
    allow_unsupported_cuda_compiler: bool,
    cuda_host_compiler: str | None,
    use_system_opencv: bool,
    recreate_env: bool,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={out_path}",
        f"#SBATCH --error={err_path}",
        f"#SBATCH --time={slurm_time}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={slurm_mem}",
    ]
    if slurm_partition:
        lines.append(f"#SBATCH --partition={slurm_partition}")
    if slurm_account:
        lines.append(f"#SBATCH --account={slurm_account}")

    module_block = ""
    if modules.strip():
        module_block = f"module purge\nmodule load {modules.strip()}\n"
    allow_unsupported_cuda_compiler_str = "1" if allow_unsupported_cuda_compiler else "0"
    cuda_host_compiler_line = ""
    if cuda_host_compiler:
        cuda_host_compiler_line = f"  -DCMAKE_CUDA_HOST_COMPILER={shlex.quote(cuda_host_compiler)} \\\n"

    conda_create_packages = "python=3.10 pytorch torchvision pytorch-cuda=12.1"
    if not use_system_opencv:
        conda_create_packages += " opencv"

    opencv_mode_echo = "system-opencv" if use_system_opencv else "conda-opencv"
    use_system_opencv_str = "1" if use_system_opencv else "0"
    recreate_env_str = "1" if recreate_env else "0"

    build_script = f"""#!/bin/bash
set -eo pipefail
{module_block}echo "[build] host=$(hostname) started=$(date -Is)"
echo "[build] OpenCV mode: {opencv_mode_echo}"
mkdir -p {shlex.quote(remote_workdir)}
mkdir -p {shlex.quote(str(Path(source_dir).parent))}
LOADED_MODULES="$(module -t list 2>&1 || true)"
echo "$LOADED_MODULES"

if [ "{use_system_opencv_str}" = "0" ] && echo "$LOADED_MODULES" | grep -Eq '^opencv/'; then
  echo "[error] OpenCV module is loaded while --use-system-opencv is disabled. Refusing mixed OpenCV setup." >&2
  exit 6
fi

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
  echo "[error] conda command not found. Install user conda or verify /apps/conda/25.7.0 availability." >&2
  exit 2
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER="${{MKL_INTERFACE_LAYER:-LP64}}"
export MKL_THREADING_LAYER="${{MKL_THREADING_LAYER:-GNU}}"

if [ "{recreate_env_str}" = "1" ] && [ -d {shlex.quote(env_prefix)} ]; then
  echo "[build] Removing existing env for clean rebuild: {shlex.quote(env_prefix)}"
  rm -rf {shlex.quote(env_prefix)}
fi

if [ ! -d {shlex.quote(env_prefix)} ]; then
  conda create -y -p {shlex.quote(env_prefix)} \
    {conda_create_packages} \
    -c pytorch -c nvidia -c conda-forge
fi

conda activate {shlex.quote(env_prefix)}
if [ "{use_system_opencv_str}" = "0" ]; then
  echo "[build] Ensuring conda OpenCV is installed"
  conda install -y -p {shlex.quote(env_prefix)} -c conda-forge opencv
else
  echo "[build] Using system OpenCV module; skipping conda OpenCV install"
  if ! command -v opencv4-config >/dev/null 2>&1 && ! command -v opencv-config >/dev/null 2>&1; then
    echo "[error] --use-system-opencv requested but opencv-config tool not found. Ensure opencv/4.7.0 is loaded." >&2
    exit 7
  fi
  # If an existing env still has conda OpenCV from earlier runs, remove it to avoid ABI/toolchain mixing.
  if conda list -p {shlex.quote(env_prefix)} | awk '{{print $1}}' | grep -q '^opencv$'; then
    echo "[build] Removing conda OpenCV packages from existing env"
    conda remove -y -p {shlex.quote(env_prefix)} opencv libopencv || true
  fi
fi

# Use module GCC toolchain for C/C++; keep conda env isolated for Python/Torch only.
export CC="${{CC:-$(command -v gcc || true)}}"
export CXX="${{CXX:-$(command -v g++ || true)}}"
if [ ! -x "$CC" ] || [ ! -x "$CXX" ]; then
  echo "[error] GCC toolchain not found/executable from loaded modules: CC=$CC CXX=$CXX" >&2
  exit 5
fi
echo "[build] CC=$CC"
echo "[build] CXX=$CXX"
echo "[build] Dependency split: modules provide gcc/cmake/cuda/opencv; conda env provides python+pytorch."

# Detect CUDA toolkit root early so we can search for compatibility libraries.
if command -v nvcc >/dev/null 2>&1; then
  export CUDAToolkit_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_TOOLKIT_ROOT_DIR="${{CUDAToolkit_ROOT}}"
  echo "[build] nvcc=$(command -v nvcc)"
  nvcc --version || true
fi

if [ -n {shlex.quote(cuda_host_compiler or "")} ]; then
  if [ ! -x {shlex.quote(cuda_host_compiler or "")} ]; then
    echo "[error] --cuda-host-compiler path is not executable: {shlex.quote(cuda_host_compiler or '')}" >&2
    exit 4
  fi
  echo "[build] CUDA host compiler override: {shlex.quote(cuda_host_compiler or '')}"
fi

NVTOOLS_SO="${{CONDA_PREFIX}}/lib/libnvToolsExt.so"
NVTOOLS_SO1="${{CONDA_PREFIX}}/lib/libnvToolsExt.so.1"

# Some torch/caffe2 releases require legacy nvToolsExt; verify by file presence, not package name.
if [ ! -f "$NVTOOLS_SO" ] && [ ! -f "$NVTOOLS_SO1" ]; then
  echo "[build] nvToolsExt not found in conda env; attempting install"
  conda install -y -p {shlex.quote(env_prefix)} -c nvidia cuda-nvtx cuda-nvtx-dev || true
  conda install -y -p {shlex.quote(env_prefix)} -c conda-forge nvtx || true
fi

# Fallback: if toolkit still ships a compatibility lib, symlink it into the env.
if [ ! -f "$NVTOOLS_SO" ] && [ ! -f "$NVTOOLS_SO1" ] && [ -n "${{CUDAToolkit_ROOT:-}}" ]; then
  CANDIDATE_NVTOOLS="$(find "${{CUDAToolkit_ROOT}}" -name 'libnvToolsExt.so*' 2>/dev/null | head -n 1 || true)"
  if [ -n "$CANDIDATE_NVTOOLS" ]; then
    echo "[build] Found toolkit nvToolsExt candidate: $CANDIDATE_NVTOOLS"
    ln -sf "$CANDIDATE_NVTOOLS" "$NVTOOLS_SO"
  fi
fi

if [ ! -f "$NVTOOLS_SO" ] && [ ! -f "$NVTOOLS_SO1" ]; then
  echo "[error] nvToolsExt library still missing after install attempts." >&2
  echo "[error] Try building with an older CUDA module (for example: cuda/12.4.1)." >&2
  conda list | grep -Ei 'nvtx|cuda' || true
  find "${{CONDA_PREFIX}}" -maxdepth 4 \\( -name '*nvtx*' -o -name '*nvToolsExt*' \\) 2>/dev/null || true
  exit 3
fi

if [ ! -d {shlex.quote(source_dir)}/.git ]; then
  git clone --recursive {shlex.quote(repo_url)} {shlex.quote(source_dir)}
fi

cd {shlex.quote(source_dir)}
git fetch --all
git checkout {shlex.quote(repo_branch)}
git pull --ff-only
git submodule update --init --recursive

# Focused OpenCV link fix:
# Ensure opensplat target links OpenCV libs explicitly to avoid final-link unresolved symbols.
if [ -f CMakeLists.txt ]; then
  if grep -Eq 'target_link_libraries[[:space:]]*\\([[:space:]]*opensplat' CMakeLists.txt; then
    if ! grep -Eq '\\$\\{{OpenCV_LIBS\\}}|opencv_imgcodecs|opencv_imgproc|opencv_core' CMakeLists.txt; then
      echo "[build] Applying focused OpenCV link patch to CMakeLists.txt"
      awk '
        BEGIN {{ inblock=0; patched=0 }}
        /target_link_libraries[[:space:]]*\\([[:space:]]*opensplat([[:space:]]|$)/ {{ inblock=1 }}
        {{
          if (inblock && !patched && $0 ~ /^[[:space:]]*\\)[[:space:]]*$/) {{
            print "  ${{OpenCV_LIBS}}"
            print "  opencv_imgcodecs opencv_imgproc opencv_core"
            patched=1
            inblock=0
          }}
          print
        }}
        END {{ if (!patched) exit 2 }}
      ' CMakeLists.txt > CMakeLists.txt.codex_tmp && mv CMakeLists.txt.codex_tmp CMakeLists.txt || true

      if ! grep -Eq '\\$\\{{OpenCV_LIBS\\}}|opencv_imgcodecs|opencv_imgproc|opencv_core' CMakeLists.txt; then
        echo "[build] OpenCV patch fallback: appending libs on single-line target_link_libraries(opensplat ...)"
        sed -E -i '0,/target_link_libraries\\([[:space:]]*opensplat[^)]*\\)/s//& ${{OpenCV_LIBS}} opencv_imgcodecs opencv_imgproc opencv_core/' CMakeLists.txt || true
      fi

      echo "[build] opensplat link line(s):"
      grep -n 'target_link_libraries[[:space:]]*\\([[:space:]]*opensplat' CMakeLists.txt || true
      grep -nE '\\$\\{{OpenCV_LIBS\\}}|opencv_imgcodecs|opencv_imgproc|opencv_core' CMakeLists.txt || true
    else
      echo "[build] OpenCV link entries already present in CMakeLists.txt"
    fi
  fi
fi

# Always reconfigure from a clean build dir to avoid stale CUDA paths in CMake cache.
rm -rf build
mkdir -p build
cd build

# Help torch/caffe2 locate NVTX on newer CUDA toolkits where nvToolsExt is absent.
CMAKE_NVTX_ARGS=()
if [ -d "${{CONDA_PREFIX}}/include/nvtx3" ]; then
  CMAKE_NVTX_ARGS+=("-Dnvtx3_dir=${{CONDA_PREFIX}}/include")
fi
if [ -f "${{CONDA_PREFIX}}/lib/libnvToolsExt.so" ]; then
  CMAKE_NVTX_ARGS+=("-DCUDA_nvToolsExt_LIBRARY=${{CONDA_PREFIX}}/lib/libnvToolsExt.so")
fi
if [ -f "${{CONDA_PREFIX}}/lib/libnvToolsExt.so.1" ]; then
  CMAKE_NVTX_ARGS+=("-DCUDA_nvToolsExt_LIBRARY=${{CONDA_PREFIX}}/lib/libnvToolsExt.so.1")
fi

CMAKE_PREFIX_PATH_BASE="${{CONDA_PREFIX}};$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
if [ "{use_system_opencv_str}" = "1" ]; then
  OPENCV_PREFIX=""
  if command -v opencv4-config >/dev/null 2>&1; then
    OPENCV_PREFIX="$(opencv4-config --prefix 2>/dev/null || true)"
  elif command -v opencv-config >/dev/null 2>&1; then
    OPENCV_PREFIX="$(opencv-config --prefix 2>/dev/null || true)"
  fi
  if [ -n "$OPENCV_PREFIX" ]; then
    # Put system OpenCV first in search path when this mode is enabled.
    CMAKE_PREFIX_PATH_BASE="${{OPENCV_PREFIX}};${{CMAKE_PREFIX_PATH_BASE}}"
    if [ -d "${{OPENCV_PREFIX}}/lib64/cmake/opencv4" ]; then
      export OpenCV_DIR="${{OpenCV_DIR:-${{OPENCV_PREFIX}}/lib64/cmake/opencv4}}"
    elif [ -d "${{OPENCV_PREFIX}}/lib/cmake/opencv4" ]; then
      export OpenCV_DIR="${{OpenCV_DIR:-${{OPENCV_PREFIX}}/lib/cmake/opencv4}}"
    fi
  fi
fi

SYSTEM_OPENCV_ARGS=()
if [ -n "${{OpenCV_DIR:-}}" ]; then
  SYSTEM_OPENCV_ARGS+=("-DOpenCV_DIR=${{OpenCV_DIR}}")
fi

OPENCV_FORCE_LINK_FLAGS=""
if [ "{use_system_opencv_str}" = "1" ]; then
  OPENCV_LIB_DIR=""
  if [ -n "${{OPENCV_PREFIX:-}}" ] && [ -d "${{OPENCV_PREFIX}}/lib64" ]; then
    OPENCV_LIB_DIR="${{OPENCV_PREFIX}}/lib64"
  elif [ -n "${{OPENCV_PREFIX:-}}" ] && [ -d "${{OPENCV_PREFIX}}/lib" ]; then
    OPENCV_LIB_DIR="${{OPENCV_PREFIX}}/lib"
  elif [ -d "/apps/opencv/4.7.0/lib64" ]; then
    OPENCV_LIB_DIR="/apps/opencv/4.7.0/lib64"
  elif [ -d "/apps/opencv/4.7.0/lib" ]; then
    OPENCV_LIB_DIR="/apps/opencv/4.7.0/lib"
  fi
  if [ -n "$OPENCV_LIB_DIR" ]; then
    OPENCV_FORCE_LINK_FLAGS="-L${{OPENCV_LIB_DIR}} -Wl,--no-as-needed -lopencv_imgcodecs -lopencv_imgproc -lopencv_core"
  fi
fi

CUDA_CMAKE_FLAGS="-Xcompiler=-fPIC"
if [ "{allow_unsupported_cuda_compiler_str}" = "1" ]; then
  CUDA_CMAKE_FLAGS="${{CUDA_CMAKE_FLAGS}} --allow-unsupported-compiler"
fi

cmake_configure() {{
  local extra_linker_flags="$1"
  local linker_flags="${{extra_linker_flags}}"
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${{CC}}" \
    -DCMAKE_CXX_COMPILER="${{CXX}}" \
    -DCMAKE_PREFIX_PATH="${{CMAKE_PREFIX_PATH_BASE}}" \
    -DCUDAToolkit_ROOT="${{CUDAToolkit_ROOT:-}}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_C_FLAGS=-fPIC \
    -DCMAKE_CXX_FLAGS=-fPIC \
    "-DCMAKE_CUDA_FLAGS=${{CUDA_CMAKE_FLAGS}}" \
    -DUSE_SYSTEM_NVTX=ON \
{cuda_host_compiler_line}\
    ${{linker_flags:+-DCMAKE_EXE_LINKER_FLAGS=${{linker_flags}}}} \
    "${{SYSTEM_OPENCV_ARGS[@]}}" \
    "${{CMAKE_NVTX_ARGS[@]}}"
}}

print_link_debug() {{
  echo "[build] CMake OpenCV detection:"
  cmake -LA -N . | grep -E '^OpenCV_DIR:|^OpenCV_VERSION:' || true
  grep -E -- '-- Found OpenCV:' CMakeFiles/CMakeOutput.log CMakeFiles/CMakeError.log 2>/dev/null || true
  if [ -f CMakeFiles/opensplat.dir/link.txt ]; then
    echo "[build] opensplat link.txt (full line):"
    sed -n '1,3p' CMakeFiles/opensplat.dir/link.txt
    echo "[build] opensplat link.txt OpenCV token summary:"
    grep -oE 'opencv_[^[:space:]]+' CMakeFiles/opensplat.dir/link.txt | sort -u || true
  else
    echo "[build] WARNING: CMakeFiles/opensplat.dir/link.txt not found yet"
  fi
}}

cmake_configure ""

print_link_debug

if [ -f CMakeFiles/opensplat.dir/link.txt ] && ! grep -q 'opencv_imgcodecs' CMakeFiles/opensplat.dir/link.txt; then
  if [ -n "${{OPENCV_FORCE_LINK_FLAGS:-}}" ]; then
    echo "[build] opencv_imgcodecs missing in link.txt; reconfiguring with explicit OpenCV linker flags"
    echo "[build] forced linker flags: $OPENCV_FORCE_LINK_FLAGS"
    rm -f CMakeCache.txt
    cmake_configure "${{OPENCV_FORCE_LINK_FLAGS}}"
    print_link_debug
  else
    echo "[build] opencv_imgcodecs missing in link.txt, but no OPENCV_FORCE_LINK_FLAGS could be determined"
  fi
fi

set +e
cmake --build . -j {cpus} 2>&1 | tee build_attempt1.log
BUILD_RC=${{PIPESTATUS[0]}}
set -e

if [ "$BUILD_RC" -ne 0 ]; then
  if grep -q "can not be used when making a PIE object" build_attempt1.log; then
    echo "[build] Detected PIE relocation link error. Retrying with -DCMAKE_EXE_LINKER_FLAGS=-no-pie"
    rm -f CMakeCache.txt
    cmake_configure "-no-pie ${{OPENCV_FORCE_LINK_FLAGS:-}}"
    print_link_debug
    cmake --build . -j {cpus}
  else
    exit "$BUILD_RC"
  fi
fi

echo "[build] finished=$(date -Is)"
ls -lh ./opensplat
./opensplat --help | head -n 20 || true
"""
    return "\n".join(lines) + "\n\n" + build_script + "\n"


def tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(text[-lines:])


def main():
    parser = argparse.ArgumentParser(
        description="Build OpenSplat on HiPerGator through SLURM and fetch build logs locally."
    )
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg or user@hpg.rc.ufl.edu)")
    parser.add_argument("--remote-base", default=DEFAULT_REMOTE_BASE, help="Remote project working directory")
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR, help="Remote OpenSplat source directory")
    parser.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX, help="Remote conda env prefix")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="OpenSplat repository URL")
    parser.add_argument("--repo-branch", default="main", help="OpenSplat branch/tag to build")

    parser.add_argument("--slurm-time", default="02:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default=None, help="SLURM partition/queue")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=8, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="32G", help="SLURM memory request")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--modules", default=DEFAULT_MODULES, help="Space-separated modules to load")
    parser.add_argument(
        "--use-system-opencv",
        action="store_true",
        help="Use OpenCV from an HPC module instead of conda OpenCV (recommended to avoid ABI mismatches).",
    )
    parser.add_argument(
        "--opencv-module",
        default="opencv/4.7.0",
        help="OpenCV module to append when --use-system-opencv is enabled",
    )
    parser.add_argument(
        "--cuda-host-compiler",
        default=None,
        help=(
            "Path to C++ compiler used by nvcc (for example: /apps/compilers/gcc/12.2.0/bin/g++). "
            "Useful when CUDA toolkit does not support the active GCC module."
        ),
    )
    parser.add_argument(
        "--strict-cuda-host-compiler",
        action="store_true",
        help="Do not pass --allow-unsupported-compiler to nvcc",
    )
    parser.add_argument(
        "--recreate-env",
        action="store_true",
        help="Delete and recreate the conda env before build (recommended for a clean UFRC-compliant rebuild).",
    )
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH connection multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")

    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    allowed_blue_prefix = "/blue/cis4914/joshuabowman"
    for label, path_value in (
        ("remote-base", args.remote_base),
        ("source-dir", args.source_dir),
        ("env-prefix", args.env_prefix),
    ):
        if not path_value.startswith(allowed_blue_prefix):
            raise ValueError(
                f"--{label} must be under {allowed_blue_prefix} for UFRC-safe workflow. Got: {path_value}"
            )
    if args.env_prefix.rstrip("/") == "/blue/cis4914/joshuabowman/conda/opensplat":
        raise ValueError(
            "Old polluted env is blocked. Use --env-prefix /blue/cis4914/joshuabowman/conda/opensplat_clean"
        )

    effective_modules = args.modules.strip()
    if args.use_system_opencv:
        modules_list = effective_modules.split() if effective_modules else []
        if args.opencv_module not in modules_list:
            effective_modules = f"{effective_modules} {args.opencv_module}".strip()
            log(f"[info] --use-system-opencv enabled; appending module {args.opencv_module}")
    else:
        if any(part.startswith("opencv/") or part == "opencv" for part in effective_modules.split()):
            raise ValueError(
                "OpenCV module in --modules while --use-system-opencv is disabled. "
                "Use either system OpenCV OR conda OpenCV, not both."
            )
    if not effective_modules:
        effective_modules = DEFAULT_MODULES

    backend_dir = Path(__file__).resolve().parent.parent
    local_build_logs = backend_dir / "build_logs"
    local_build_logs.mkdir(parents=True, exist_ok=True)

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ssh_base(args.remote, args.port, args.identity_file, ssh_opts)
    remote_workdir = args.remote_base.rstrip("/")
    remote_slurm_dir = f"{remote_workdir}/slurm_jobs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_tag = f"opensplat_build_{stamp}"
    remote_sbatch_path = f"{remote_slurm_dir}/{job_tag}.sbatch"
    remote_out_path = f"{remote_slurm_dir}/{job_tag}.out"
    remote_err_path = f"{remote_slurm_dir}/{job_tag}.err"
    local_out_path = local_build_logs / f"{job_tag}.out"
    local_err_path = local_build_logs / f"{job_tag}.err"

    total_start = time.perf_counter()
    log(
        f"Starting OpenSplat build workflow remote={args.remote} remote_base={remote_workdir} "
        f"source_dir={args.source_dir}"
    )
    if args.dry_run:
        log("Dry-run mode enabled.")

    stage_name = "Stage 1/4: Prepare remote folders and submit build job"
    st = stage(stage_name)
    run_cmd(
        ssh_cmd(ssh, f"mkdir -p {shlex.quote(remote_workdir)} {shlex.quote(remote_slurm_dir)}"),
        dry_run=args.dry_run,
    )

    sbatch_script = build_sbatch_script(
        remote_workdir=remote_workdir,
        source_dir=args.source_dir,
        env_prefix=args.env_prefix,
        repo_url=args.repo_url,
        repo_branch=args.repo_branch,
        cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        slurm_mem=args.slurm_mem,
        out_path=remote_out_path,
        err_path=remote_err_path,
        job_name="build_opensplat",
        modules=effective_modules,
        allow_unsupported_cuda_compiler=not args.strict_cuda_host_compiler,
        cuda_host_compiler=args.cuda_host_compiler,
        use_system_opencv=args.use_system_opencv,
        recreate_env=args.recreate_env,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(sbatch_script)
        local_temp_script = Path(temp_file.name)
    try:
        run_cmd(
            scp_upload_cmd(args.remote, args.port, args.identity_file, local_temp_script, remote_sbatch_path, ssh_opts),
            dry_run=args.dry_run,
        )
    finally:
        local_temp_script.unlink(missing_ok=True)

    if args.dry_run:
        run_cmd(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"), dry_run=True)
        job_id = "DRYRUN_JOB"
    else:
        sbatch_out = run_cmd_capture(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"))
        log(f"sbatch output: {sbatch_out}")
        job_id = parse_job_id(sbatch_out)
        log(f"Submitted SLURM job id: {job_id}")
    stage_done(stage_name, st)

    stage_name = "Stage 2/4: Poll SLURM job"
    st = stage(stage_name)
    final_state = poll_slurm_job(ssh=ssh, job_id=job_id, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    log(f"SLURM job {job_id} final state: {final_state}")
    stage_done(stage_name, st)

    stage_name = "Stage 3/4: Fetch build logs"
    st = stage(stage_name)
    run_cmd(
        scp_download_cmd(args.remote, args.port, args.identity_file, remote_out_path, local_out_path, ssh_opts),
        dry_run=args.dry_run,
    )
    run_cmd(
        scp_download_cmd(args.remote, args.port, args.identity_file, remote_err_path, local_err_path, ssh_opts),
        dry_run=args.dry_run,
    )
    stage_done(stage_name, st)

    if not args.dry_run and not final_state.startswith("COMPLETED"):
        err_tail = tail_text(local_err_path, lines=60)
        out_tail = tail_text(local_out_path, lines=60)
        if err_tail:
            log("Last lines from build stderr log:")
            print(err_tail, flush=True)
        if out_tail:
            log("Last lines from build stdout log:")
            print(out_tail, flush=True)
        raise RuntimeError(
            f"Build job ended in state {final_state}. Check logs: {local_out_path} and {local_err_path}"
        )

    stage_name = "Stage 4/4: Verify binary path"
    st = stage(stage_name)
    remote_binary = f"{args.source_dir.rstrip('/')}/build/opensplat"
    run_cmd(ssh_cmd(ssh, f"ls -lh {shlex.quote(remote_binary)}"), dry_run=args.dry_run)
    stage_done(stage_name, st)

    elapsed = time.perf_counter() - total_start
    log(f"[ok] Build logs: {local_out_path}, {local_err_path}")
    log(f"[ok] Remote binary: {remote_binary}")
    log(f"Workflow complete ({elapsed:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
