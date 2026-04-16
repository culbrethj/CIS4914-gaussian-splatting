from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    from .converter import ply_to_splat
except ImportError:
    from converter import ply_to_splat

DEFAULT_REMOTE_BASE = "/blue/cis4914/joshuabowman/gaussian-splatting"


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


def rsync_ssh_option(port: int, identity_file: str | None, ssh_opts: list[str]) -> str:
    opts = ["ssh", "-p", str(port)]
    if identity_file:
        opts.extend(["-i", identity_file])
    opts.extend(ssh_opts)
    return " ".join(shlex.quote(p) for p in opts)


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


def parse_job_id(sbatch_output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job id from sbatch output: {sbatch_output!r}")
    return match.group(1)


def build_sbatch_script(
    *,
    remote_workdir: str,
    remote_dataset: str,
    remote_ply: str,
    remote_opensplat: str,
    iters: int,
    out_path: str,
    err_path: str,
    job_name: str,
    slurm_time: str,
    slurm_partition: str | None,
    slurm_account: str | None,
    slurm_gres: str,
    slurm_cpus: int,
    slurm_mem: str,
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
    if slurm_gres:
        lines.append(f"#SBATCH --gres={slurm_gres}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            'echo "[slurm] host=$(hostname) started=$(date -Is)"',
            f"cd {shlex.quote(remote_workdir)}",
            f"{remote_opensplat} {shlex.quote(remote_dataset)} -o {shlex.quote(remote_ply)} -n {iters}",
            f'echo "[slurm] finished=$(date -Is) output={shlex.quote(remote_ply)}"',
            f"ls -lh {shlex.quote(remote_ply)}",
        ]
    )
    return "\n".join(lines) + "\n"


def remote_file_exists(ssh: list[str], path: str, *, dry_run: bool) -> bool:
    if dry_run:
        run_cmd(ssh_cmd(ssh, f"test -f {shlex.quote(path)}"), dry_run=True)
        return True
    try:
        run_cmd(ssh_cmd(ssh, f"test -f {shlex.quote(path)}"))
        return True
    except subprocess.CalledProcessError:
        return False


def poll_slurm_job(*, ssh: list[str], job_id: str, poll_seconds: int, dry_run: bool) -> str:
    if dry_run:
        run_cmd(ssh_cmd(ssh, f"squeue -h -j {job_id} -o %T"), dry_run=True)
        log("[dry-run] Skipping SLURM polling loop.")
        return "COMPLETED"

    last_state = None
    while True:
        state = run_cmd_capture(ssh_cmd(ssh, f"squeue -h -j {job_id} -o %T"))
        state = state.strip()
        if not state:
            break
        if state != last_state:
            log(f"SLURM job {job_id} state: {state}")
            last_state = state
        time.sleep(poll_seconds)

    final_state = run_cmd_capture(
        ssh_cmd(ssh, f"sacct -n -X -j {job_id} -o State | head -n 1 | awk '{{print $1}}'")
    ).strip()
    if not final_state:
        final_state = "UNKNOWN"
    log(f"SLURM job {job_id} final state: {final_state}")
    return final_state


def main():
    parser = argparse.ArgumentParser(
        description="Sync dataset to HiPerGator, submit OpenSplat via SLURM, fetch outputs/logs, and publish to backend/hipergator"
    )
    parser.add_argument("dataset", help="Dataset name (must exist under backend/datasets/<dataset>)")
    parser.add_argument("--remote", required=True, help="SSH target in form user@host")
    parser.add_argument(
        "--remote-base",
        default=DEFAULT_REMOTE_BASE,
        help=f"Remote working directory (default: {DEFAULT_REMOTE_BASE})",
    )
    parser.add_argument("--remote-opensplat", default="opensplat", help="Remote OpenSplat executable on compute node")
    parser.add_argument("--iters", type=int, default=1000, help="OpenSplat iteration count")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", help="SSH private key path")

    parser.add_argument("--slurm-time", default="02:00:00", help="SLURM time limit")
    parser.add_argument("--slurm-partition", default=None, help="SLURM partition/queue")
    parser.add_argument("--slurm-account", default=None, help="SLURM account")
    parser.add_argument("--slurm-gres", default="gpu:1", help="SLURM GRES request")
    parser.add_argument("--slurm-cpus", type=int, default=8, help="SLURM CPUs per task")
    parser.add_argument("--slurm-mem", default="32G", help="SLURM memory request")
    parser.add_argument("--poll-seconds", type=int, default=20, help="Polling interval while job is queued/running")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH connection multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")

    parser.add_argument("--skip-sync", action="store_true", help="Skip rsync dataset upload")
    parser.add_argument("--skip-run", action="store_true", help="Skip SLURM submit/run")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetching outputs and local conversion")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    backend_dir = scripts_dir.parent
    local_datasets = backend_dir / "datasets"
    local_hpg = backend_dir / "hipergator"

    local_dataset = local_datasets / args.dataset
    if not local_dataset.exists() or not local_dataset.is_dir():
        raise FileNotFoundError(f"Local dataset not found: {local_dataset}")

    remote_workdir = args.remote_base.rstrip("/")
    remote_dataset = f"{remote_workdir}/datasets/{args.dataset}"
    remote_ply = f"{remote_dataset}/splat.ply"
    remote_slurm_dir = f"{remote_workdir}/slurm_jobs"

    local_ply = local_dataset / "splat.ply"
    local_slurm_dir = local_dataset / "slurm_logs"

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ssh_base(args.remote, args.port, args.identity_file, ssh_opts)
    run_start = time.perf_counter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_tag = f"{args.dataset}_{stamp}"
    remote_sbatch_path = f"{remote_slurm_dir}/opensplat_{job_tag}.sbatch"
    remote_log_out = f"{remote_slurm_dir}/opensplat_{job_tag}.out"
    remote_log_err = f"{remote_slurm_dir}/opensplat_{job_tag}.err"
    local_log_out = local_slurm_dir / Path(remote_log_out).name
    local_log_err = local_slurm_dir / Path(remote_log_err).name

    log(
        f"Starting HiPerGator workflow dataset={args.dataset} remote={args.remote} "
        f"remote_workdir={remote_workdir}"
    )
    if args.dry_run:
        log("Dry-run mode enabled (commands will be printed only).")

    if not args.skip_sync:
        stage_name = "Stage 1/5: Sync dataset to remote"
        stage_start = stage(stage_name)
        run_cmd(ssh_cmd(ssh, f"mkdir -p {shlex.quote(remote_workdir)} {shlex.quote(remote_dataset)} {shlex.quote(remote_slurm_dir)}"), dry_run=args.dry_run)
        run_cmd(
            [
                "rsync",
                "-az",
                "--exclude",
                "video",
                "--exclude",
                "raw",
                "-e",
                rsync_ssh_option(args.port, args.identity_file, ssh_opts),
                f"{local_dataset}/",
                f"{args.remote}:{remote_dataset}/",
            ],
            dry_run=args.dry_run,
        )
        stage_done(stage_name, stage_start)
    else:
        log("Skipping sync stage (--skip-sync).")

    slurm_job_id = None
    slurm_final_state = None
    if not args.skip_run:
        stage_name = "Stage 2/5: Build and submit SLURM job"
        stage_start = stage(stage_name)

        sbatch_content = build_sbatch_script(
            remote_workdir=remote_workdir,
            remote_dataset=remote_dataset,
            remote_ply=remote_ply,
            remote_opensplat=args.remote_opensplat,
            iters=args.iters,
            out_path=remote_log_out,
            err_path=remote_log_err,
            job_name=f"gs_{args.dataset}",
            slurm_time=args.slurm_time,
            slurm_partition=args.slurm_partition,
            slurm_account=args.slurm_account,
            slurm_gres=args.slurm_gres,
            slurm_cpus=args.slurm_cpus,
            slurm_mem=args.slurm_mem,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False, encoding="utf-8") as fh:
            fh.write(sbatch_content)
            local_temp_sbatch = Path(fh.name)

        try:
            run_cmd(
                scp_upload_cmd(
                    args.remote,
                    args.port,
                    args.identity_file,
                    local_temp_sbatch,
                    remote_sbatch_path,
                    ssh_opts,
                ),
                dry_run=args.dry_run,
            )
        finally:
            local_temp_sbatch.unlink(missing_ok=True)

        if args.dry_run:
            run_cmd(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"), dry_run=True)
            slurm_job_id = "DRYRUN_JOB"
        else:
            sbatch_out = run_cmd_capture(ssh_cmd(ssh, f"sbatch {shlex.quote(remote_sbatch_path)}"))
            log(f"sbatch output: {sbatch_out}")
            slurm_job_id = parse_job_id(sbatch_out)
            log(f"Submitted SLURM job id: {slurm_job_id}")

        stage_done(stage_name, stage_start)

        stage_name = "Stage 3/5: Poll SLURM job until completion"
        stage_start = stage(stage_name)
        slurm_final_state = poll_slurm_job(
            ssh=ssh,
            job_id=slurm_job_id,
            poll_seconds=args.poll_seconds,
            dry_run=args.dry_run,
        )
        stage_done(stage_name, stage_start)
    else:
        log("Skipping SLURM run stage (--skip-run).")

    if not args.skip_fetch:
        stage_name = "Stage 4/5: Fetch remote outputs and logs"
        stage_start = stage(stage_name)
        local_slurm_dir.mkdir(parents=True, exist_ok=True)

        if remote_file_exists(ssh, remote_log_out, dry_run=args.dry_run):
            run_cmd(
                scp_download_cmd(args.remote, args.port, args.identity_file, remote_log_out, local_log_out, ssh_opts),
                dry_run=args.dry_run,
            )
        else:
            log(f"[warn] Remote stdout log not found: {remote_log_out}")

        if remote_file_exists(ssh, remote_log_err, dry_run=args.dry_run):
            run_cmd(
                scp_download_cmd(args.remote, args.port, args.identity_file, remote_log_err, local_log_err, ssh_opts),
                dry_run=args.dry_run,
            )
        else:
            log(f"[warn] Remote stderr log not found: {remote_log_err}")

        ply_exists = remote_file_exists(ssh, remote_ply, dry_run=args.dry_run)
        if ply_exists:
            run_cmd(
                scp_download_cmd(args.remote, args.port, args.identity_file, remote_ply, local_ply, ssh_opts),
                dry_run=args.dry_run,
            )
        else:
            log(f"[warn] Remote PLY not found: {remote_ply}")

        stage_done(stage_name, stage_start)

        if args.dry_run:
            log("[dry-run] Skipping local conversion, publish, and final-state validation.")
            total_elapsed = time.perf_counter() - run_start
            log(f"Workflow complete ({total_elapsed:.1f}s)")
            return

        if slurm_final_state and not slurm_final_state.startswith("COMPLETED"):
            raise RuntimeError(
                f"SLURM job ended in state {slurm_final_state}. "
                f"Inspect logs at {local_log_out} and {local_log_err}"
            )
        if not local_ply.exists():
            raise RuntimeError(f"Expected local PLY not found after fetch: {local_ply}")

        stage_name = "Stage 5/5: Convert and publish .splat"
        stage_start = stage(stage_name)
        local_splat = Path(ply_to_splat(str(local_ply)))
        local_hpg.mkdir(parents=True, exist_ok=True)
        gallery_splat = local_hpg / f"{args.dataset}.splat"
        shutil.copy2(local_splat, gallery_splat)
        stage_done(stage_name, stage_start)

        log(f"[ok] Local PLY: {local_ply}")
        log(f"[ok] Local SPLAT: {local_splat}")
        log(f"[ok] Gallery SPLAT: {gallery_splat}")
        log(f"[ok] Local SLURM logs: {local_slurm_dir}")
    else:
        log("Skipping fetch/convert stage (--skip-fetch).")

    total_elapsed = time.perf_counter() - run_start
    log(f"Workflow complete ({total_elapsed:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
