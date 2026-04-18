"""
Alternative SfM stage: submits VGGT (Visual Geometry Grounded Transformer)
to SLURM on a GPU node instead of running COLMAP on CPU.

Why: COLMAP takes 8-12 minutes on CPU for a typical video dataset. VGGT
runs a single feed-forward pass on a GPU and produces the same
camera/point cloud output in under a minute. The paper is Wang et al.,
CVPR 2025, from Facebook Research. See their repo + demo_colmap.py which
is what this wrapper actually invokes.

Output contract (drop-in replacement for hpg_gs_final_prepare.py):
  $REMOTE_ROOT/experiments/faster-gs/datasets/<dataset>/
    images/               (rsync'd earlier; unchanged)
    sparse/0/cameras.bin  (VGGT writes to sparse/, we flatten to sparse/0/)
    sparse/0/images.bin
    sparse/0/points3D.bin
    sparse/points.ply     (viz only, not consumed)

VGGT emits PINHOLE / SIMPLE_PINHOLE cameras directly - no lens distortion
parameter, so we skip COLMAP's image_undistorter step the COLMAP path
uses. fastergs_preflight.py still runs at the end to validate.
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


# HPG workspace defaults. FASTERGS_REMOTE_ROOT env var moves everything
# at once; individual paths can still be overridden via CLI flag.
DEFAULT_REMOTE_ROOT = os.environ.get("FASTERGS_REMOTE_ROOT", "/blue/cis4914/joshuabowman/gs_final")
DEFAULT_VGGT_REPO = f"{DEFAULT_REMOTE_ROOT}/src/vggt"
DEFAULT_ENV_PREFIX = f"{DEFAULT_REMOTE_ROOT}/envs/fastergs_cuda128"
DEFAULT_MODELS_DIR = f"{DEFAULT_REMOTE_ROOT}/models"
DEFAULT_PREFLIGHT_SCRIPT = f"{DEFAULT_REMOTE_ROOT}/src/fastergs_preflight.py"
DEFAULT_MODULES = "git cmake gcc/12.2.0 conda/25.7.0 cuda/12.8.1"


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
                "-o", "ControlMaster=auto",
                "-o", f"ControlPersist={control_persist}",
                "-o", "ControlPath=/tmp/ssh_mux_%r_%h_%p",
            ]
        )
    return opts


def parse_job_id(sbatch_output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse job id from sbatch output: {sbatch_output!r}")
    return match.group(1)


def poll_slurm_job(*, ssh: list[str], job_id: str, poll_seconds: int, dry_run: bool) -> str:
    # Straight adaptation of the polling helper in hpg_gs_final_prepare.py.
    # Keeps wall time output + last state summary so the user sees activity.
    squeue_cmd = f"squeue -h -j {job_id} -o %T\\|%R"
    sacct_cmd = f"sacct -n -X -j {job_id} -o State | head -n 1 | awk '{{print $1}}'"
    if dry_run:
        run_cmd(ssh + [bash_lc(squeue_cmd)], dry_run=True)
        return "COMPLETED"

    last_state = None
    start = time.time()
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
        if state != last_state:
            if reason and reason not in {"None", "(None)", "null", "(null)"}:
                log(f"SLURM job {job_id} state: {state} ({reason})")
            else:
                log(f"SLURM job {job_id} state: {state}")
            last_state = state
        if time.time() - start > 60 and int((time.time() - start) / 60) != int((time.time() - start - poll_seconds) / 60):
            mins = (time.time() - start) / 60
            log(f"SLURM job {job_id} still {state} after {mins:.1f} min")
        time.sleep(poll_seconds)

    # Job exited the queue - look up the final state via sacct.
    time.sleep(2)
    final = run_cmd_capture(ssh + [bash_lc(sacct_cmd)], log_cmd=False).strip() or "UNKNOWN"
    return final


def build_sbatch_script(
    *,
    remote_root: str,
    dataset: str,
    vggt_repo: str,
    env_prefix: str,
    models_dir: str,
    preflight_script: str,
    modules: str,
    use_ba: bool,
    camera_type: str,
    slurm_time: str,
    slurm_cpus: int,
    slurm_mem: str,
    slurm_gpus: int,
    slurm_partition: str | None,
    slurm_account: str | None,
    out_path: str,
    err_path: str,
) -> str:
    # SBATCH preamble. VGGT needs a GPU, so we target the same partition as
    # training (hpg-turin by default).
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gsf_vggt",
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

    use_ba_flag = "--use_ba" if use_ba else ""

    # Body runs on the GPU compute node. Note that $HF_HOME is pinned to
    # the shared models dir so we don't redownload the 5 GB weights every
    # job - first run pays the download cost, every subsequent run is
    # ~2 s hub cache hit.
    script = f"""#!/bin/bash
set -euo pipefail
{module_block}
echo "[vggt] host=$(hostname) started=$(date -Is)"
nvidia-smi || true

ROOT={shlex.quote(remote_root)}
DATASET={shlex.quote(dataset)}
VGGT_REPO={shlex.quote(vggt_repo)}
ENV_PREFIX={shlex.quote(env_prefix)}
MODELS_DIR={shlex.quote(models_dir)}
PREFLIGHT={shlex.quote(preflight_script)}

SRC="$ROOT/datasets/$DATASET"
IMG="$SRC/images"
SCENE="$ROOT/experiments/faster-gs/datasets/$DATASET"
SCENE_SPARSE="$SCENE/sparse"
SCENE_SPARSE_FINAL="$SCENE/sparse/0"

# HuggingFace cache + torch hub cache both land under $MODELS_DIR so the
# first run fills the cache and every future run reuses the weights.
export HF_HOME="$MODELS_DIR/hf"
export TORCH_HOME="$MODELS_DIR/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$SCENE"

if [ ! -d "$IMG" ]; then
  echo "[error] Missing images directory: $IMG" >&2
  exit 12
fi

echo "[vggt] Input image count: $(find "$IMG" -maxdepth 1 -type f | wc -l)"

# Mirror the rsynced images into the scene dir VGGT's demo_colmap.py
# expects. We copy with cp -l (hardlink) so we don't double-store 500 MB
# per dataset on /blue.
mkdir -p "$SCENE/images"
cp -f -l "$IMG"/* "$SCENE/images/" 2>/dev/null || cp -f "$IMG"/* "$SCENE/images/"

# Nuke any stale sparse dir from a previous run (COLMAP could have left
# something behind) so demo_colmap.py starts from a clean slate.
rm -rf "$SCENE_SPARSE"

# Activate the conda env that has vggt + torch + pycolmap + hydra installed.
source /apps/conda/25.7.0/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"

# VGGT's demo_colmap.py was written against pycolmap 3.10 and does
# pycolmap.Image(... cam_from_world=...). pycolmap 4.x made cam_from_world
# read-only (it became a method rather than a settable attribute), which
# is not patchable by rewriting the call. So we hard-require 3.x here and
# bail before the long model download if someone rebuilt the env.
PYCOLMAP_VER=$(python -c 'import pycolmap; print(pycolmap.__version__)')
case "$PYCOLMAP_VER" in
  3.*) echo "[vggt] pycolmap $PYCOLMAP_VER OK" ;;
  *)   echo "[vggt] ERROR: pycolmap $PYCOLMAP_VER is incompatible with VGGT. Reinstall with: pip install --no-deps 'pycolmap==3.10.0'" >&2
       exit 2 ;;
esac

cd "$VGGT_REPO"

# tee demo_colmap.py output to a local file so the post-run grep can
# classify failures (inlier shortage vs. pycolmap mismatch vs. other)
# and print a user-actionable hint to the SLURM log. Stored under the
# job's own tmp dir so concurrent runs don't stomp on each other.
RUN_LOG=$(mktemp -t vggt_run.XXXXXX.log)

# Run VGGT. Default camera_type is SIMPLE_PINHOLE which our preflight
# and Faster-GS loader both accept. --use_ba is optional; without it,
# VGGT's feed-forward output is used directly (the paper reports this
# is already competitive with COLMAP).
#
# Capture stdout+stderr so we can map VGGT's generic Python tracebacks
# into actionable hints (e.g. "BA couldn't find enough inliers - try
# COLMAP for this dataset") before the SLURM job exits non-zero. If
# the tracker fails on a dataset with wide baseline / repetitive
# texture, the stock demo throws `No reconstruction can be built with
# BA` and a traceback that looks scarier than it is.
VGGT_START=$(date +%s)
set +e
python demo_colmap.py \
  --scene_dir "$SCENE" \
  --camera_type {shlex.quote(camera_type)} \
  {use_ba_flag} 2>&1 | tee "$RUN_LOG"
VGGT_RC=${{PIPESTATUS[0]}}
set -e
VGGT_END=$(date +%s)
echo "[vggt] VGGT wall seconds: $((VGGT_END - VGGT_START))"

if [ "$VGGT_RC" -ne 0 ]; then
  echo "[vggt] demo_colmap.py exited $VGGT_RC" >&2
  if grep -q "Not enough inliers per frame" "$RUN_LOG"; then
    cat >&2 <<'HINT'
[vggt] hint: VGGT's learned tracker found too few cross-frame inliers
[vggt]       to build a reconstruction. This happens on datasets with
[vggt]       wide baselines or repetitive texture (e.g. the COLMAP
[vggt]       benchmark scenes). Switch SfM method to COLMAP in the
[vggt]       LiveDemos Advanced Settings for this dataset - COLMAP's
[vggt]       exhaustive matcher is more tolerant of those conditions.
HINT
  elif grep -q "No reconstruction can be built with BA" "$RUN_LOG"; then
    cat >&2 <<'HINT'
[vggt] hint: VGGT's bundle adjustment could not converge. Try COLMAP
[vggt]       for this dataset, or rerun with --no-vggt-ba (cameras
[vggt]       only, no 3D points - training still won't work, but lets
[vggt]       you verify VGGT loads and produces camera poses).
HINT
  fi
  exit "$VGGT_RC"
fi

# demo_colmap.py writes to $SCENE/sparse/{{cameras,images,points3D}}.bin
# but our trainer expects the standard $SCENE/sparse/0/ layout. Flatten.
mkdir -p "$SCENE_SPARSE_FINAL"
for f in cameras.bin images.bin points3D.bin; do
  if [ -f "$SCENE_SPARSE/$f" ]; then
    mv "$SCENE_SPARSE/$f" "$SCENE_SPARSE_FINAL/$f"
  fi
done
# Keep the viz point cloud where VGGT put it (sparse/points.ply) - it
# doesn't affect training either way.

echo "[vggt] Sparse model:"
ls -la "$SCENE_SPARSE_FINAL" || true

# Preflight check: fails loudly if the output is not compatible with the
# Faster-GS Inria loader (e.g. a non-PINHOLE-family camera model slipped
# through). Same gate the COLMAP path runs.
if [ -f "$PREFLIGHT" ]; then
  python "$PREFLIGHT" "$SCENE"
else
  echo "[warn] Preflight script not found: $PREFLIGHT"
fi

echo "[ok] vggt SfM stage complete"
echo "[vggt] finished=$(date -Is)"
"""
    return "\n".join(lines) + "\n\n" + script + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Run VGGT SfM on HiPerGator (GPU) as a drop-in replacement for hpg_gs_final_prepare.py.",
    )
    parser.add_argument("dataset", help="Dataset name (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg)")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote gs_final root")
    parser.add_argument("--vggt-repo", default=DEFAULT_VGGT_REPO, help="Path to cloned VGGT repo on HPG")
    parser.add_argument("--env-prefix", default=DEFAULT_ENV_PREFIX, help="Conda env with vggt installed")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Shared HF/torch cache dir")
    parser.add_argument("--preflight-script", default=DEFAULT_PREFLIGHT_SCRIPT, help="Remote preflight script path")
    parser.add_argument("--modules", default=DEFAULT_MODULES, help="module-load line for SBATCH body")
    parser.add_argument("--use-ba", action="store_true", help="Enable VGGT bundle adjustment (slower, more accurate)")
    parser.add_argument("--camera-type", default="SIMPLE_PINHOLE",
                        help="Camera model VGGT emits. SIMPLE_PINHOLE (default) is Faster-GS-compatible.")
    parser.add_argument("--slurm-time", default="00:30:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default="hpg-turin", help="SLURM partition (needs GPU; hpg-turin = L4/sm_89)")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-cpus", type=int, default=2, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="24G", help="SLURM memory request")
    parser.add_argument("--slurm-gpus", type=int, default=1, help="SLURM GPU count")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval")
    parser.add_argument("--no-wait", action="store_true", help="Submit and return without waiting")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    root = args.remote_root.rstrip("/")
    remote_logs_dir = f"{root}/logs"
    tag = f"gsf_vggt_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    remote_sbatch = f"{remote_logs_dir}/{tag}.sbatch"
    remote_out = f"{remote_logs_dir}/{tag}.out"
    remote_err = f"{remote_logs_dir}/{tag}.err"

    backend_dir = Path(__file__).resolve().parent.parent
    local_logs = backend_dir / "build_logs"
    local_logs.mkdir(parents=True, exist_ok=True)
    local_out = local_logs / f"{tag}.out"
    local_err = local_logs / f"{tag}.err"

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ["ssh", "-p", str(args.port)]
    if args.identity_file:
        ssh.extend(["-i", args.identity_file])
    ssh.extend(ssh_opts)
    ssh.append(args.remote)
    scp_base = ["scp", "-P", str(args.port), *(["-i", args.identity_file] if args.identity_file else []), *ssh_opts]

    log(f"Starting VGGT SfM dataset={args.dataset} remote_root={root} (partition={args.slurm_partition}, gpus={args.slurm_gpus})")
    run_cmd(
        ssh + [bash_lc(f"mkdir -p {shlex.quote(root)} {shlex.quote(remote_logs_dir)} {shlex.quote(args.models_dir)}")],
        dry_run=args.dry_run,
    )

    # Auto-upload the preflight script in case the remote copy is stale;
    # training uses the same script so keeping them in sync matters.
    local_preflight = backend_dir / "scripts" / "fastergs_preflight.py"
    if args.preflight_script == DEFAULT_PREFLIGHT_SCRIPT and local_preflight.is_file():
        run_cmd(
            ssh + [bash_lc(f"mkdir -p $(dirname {shlex.quote(args.preflight_script)})")],
            dry_run=args.dry_run,
        )
        run_cmd(
            scp_base + [str(local_preflight), f"{args.remote}:{args.preflight_script}"],
            dry_run=args.dry_run,
        )

    sbatch_text = build_sbatch_script(
        remote_root=root,
        dataset=args.dataset,
        vggt_repo=args.vggt_repo,
        env_prefix=args.env_prefix,
        models_dir=args.models_dir,
        preflight_script=args.preflight_script,
        modules=args.modules,
        use_ba=args.use_ba,
        camera_type=args.camera_type,
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
        log("[ok] VGGT SfM submission complete (--no-wait)")
        log(f"Track with: squeue -j {job_id}")
        log(f"Remote logs: {remote_out} {remote_err}")
        return

    final_state = poll_slurm_job(ssh=ssh, job_id=job_id, poll_seconds=args.poll_seconds, dry_run=args.dry_run)
    log(f"SLURM job {job_id} final state: {final_state}")

    # Always fetch both logs so a failure is easy to diagnose locally.
    try:
        run_cmd(scp_base + [f"{args.remote}:{remote_out}", str(local_out)], dry_run=args.dry_run)
        run_cmd(scp_base + [f"{args.remote}:{remote_err}", str(local_err)], dry_run=args.dry_run)
    except subprocess.CalledProcessError as exc:
        log(f"[warn] could not fetch remote logs: {exc}")
    log(f"Local logs: {local_out} {local_err}")

    if not args.dry_run and not final_state.startswith("COMPLETED"):
        raise RuntimeError(f"VGGT job ended in state {final_state}. Check logs: {local_out} {local_err}")

    log("[ok] vggt SfM stage completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
