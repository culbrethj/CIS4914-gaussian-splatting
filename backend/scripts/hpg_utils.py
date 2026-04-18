"""Shared helpers used by the HPG pipeline scripts.

Each HPG orchestrator script (fastergs_pipeline, hpg_gs_final_prepare,
hpg_gs_final_sync, hpg_gs_final_train, hpg_gs_vggt_sfm) had its own
copy of these before; factoring them out kept the scripts in sync and
made the behavior easy to audit in one place.

Intentionally small. ``run_cmd`` stays per-script because signatures
and return handling differ between scripts; the bits below are the
strictly-identical pieces.
"""

from __future__ import annotations

import shlex
from datetime import datetime


def log(message: str) -> None:
    """Timestamped log line. Matches the format every HPG script used."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def format_cmd(cmd: list[str]) -> str:
    """Shell-quote a command list and collapse to a single display line.

    Truncates anything longer than 260 chars so giant sbatch scripts don't
    drown the log.
    """
    printable = " ".join(shlex.quote(part) for part in cmd)
    compact = " ".join(printable.replace("\n", " ").split())
    if len(compact) > 260:
        return compact[:257] + "..."
    return compact


def common_ssh_options(*, use_mux: bool, control_persist: str) -> list[str]:
    """SSH multiplexing options.

    Reuses one TCP connection + auth for every ssh/rsync in a job. Without
    this we re-auth every step and HPG's login nodes get grumpy.
    """
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
