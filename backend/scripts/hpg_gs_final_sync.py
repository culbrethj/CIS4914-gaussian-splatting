from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime

DEFAULT_REMOTE_ROOT = "/blue/cis4914/joshuabowman/gs_final"
DEFAULT_SOURCE_DATASETS_ROOT = "/blue/cis4914/joshuabowman/gaussian-splatting/experiments/faster-gs/datasets"


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_cmd(cmd: list[str], *, dry_run: bool):
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"[cmd] {printable}")
    if dry_run:
        return
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


def main():
    parser = argparse.ArgumentParser(
        description="Initialize gs_final workspace and sync cleaned dataset images into gs_final/datasets/<dataset>/images."
    )
    parser.add_argument("dataset", help="Dataset name (e.g. can)")
    parser.add_argument("--remote", required=True, help="SSH target (example: hpg)")
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT, help="Remote gs_final root directory")
    parser.add_argument(
        "--source-datasets-root",
        default=DEFAULT_SOURCE_DATASETS_ROOT,
        help="Remote source datasets root containing cleaned images (default: existing faster-gs datasets root)",
    )
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity-file", default=None, help="SSH identity file")
    parser.add_argument("--no-ssh-mux", action="store_true", help="Disable SSH multiplexing")
    parser.add_argument("--ssh-control-persist", default="8h", help="SSH control socket keepalive duration")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh = ["ssh", "-p", str(args.port)]
    if args.identity_file:
        ssh.extend(["-i", args.identity_file])
    ssh.extend(ssh_opts)
    ssh.append(args.remote)

    root = args.remote_root.rstrip("/")
    src_images = f"{args.source_datasets_root.rstrip('/')}/{args.dataset}/images"
    dst_dataset = f"{root}/datasets/{args.dataset}"
    dst_images = f"{dst_dataset}/images"

    remote_cmd = f"""
set -euo pipefail
ROOT={shlex.quote(root)}
SRC_IMAGES={shlex.quote(src_images)}
DST_DATASET={shlex.quote(dst_dataset)}
DST_IMAGES={shlex.quote(dst_images)}

mkdir -p "$ROOT/datasets" "$ROOT/outputs" "$ROOT/logs" "$ROOT/src" "$ROOT/envs" "$ROOT/experiments/faster-gs"
mkdir -p "$DST_DATASET"

if [ ! -d "$SRC_IMAGES" ]; then
  echo "[error] Source images directory not found: $SRC_IMAGES" >&2
  exit 11
fi

mkdir -p "$DST_IMAGES"
rsync -a --delete "$SRC_IMAGES"/ "$DST_IMAGES"/

echo "[ok] Synced images to $DST_IMAGES"
echo "[info] Image count: $(find "$DST_IMAGES" -maxdepth 1 -type f | wc -l)"
"""
    run_cmd(ssh + [f"bash -lc {shlex.quote(remote_cmd)}"], dry_run=args.dry_run)
    log("[ok] gs_final sync stage complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"[error] {exc}")
        sys.exit(1)
