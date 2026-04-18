"""
Local orchestrator for the Faster-GS (and OpenSplat) backends.

Ties the local preprocess step to the remote HiPerGator stages:
preprocess + rsync -> SfM on HPG (COLMAP or VGGT, picked via
``--sfm-method``) -> GPU training on HPG (``--backend fastergs``
or ``opensplat``) -> fetch + publish the resulting .splat back to
``backend/datasets/<name>/splat.splat`` (for the frontend viewer) and
``backend/hipergator/<name>_fastergs_latest.splat`` (for the Gallery
page). COLMAP runs on CPU + undistorts; VGGT is a GPU feed-forward
transformer and emits PINHOLE cameras directly so undistort is
skipped.

Forwards the Shorter-Splatting paper flags (``--shortgs-*``) through
as ``SHORTGS_*`` environment variables for the SLURM training job; the
patch script on the remote side reads them from ``os.environ``. Also
exposes ``--use-fastergs-adam`` to flip on just the fused-Adam kernel
without the (currently-broken-on-sm_89/sm_100) custom rasterizer.

Entry points:
  python scripts/fastergs_pipeline.py <dataset> --video <path>
  python scripts/fastergs_pipeline.py <dataset> --use-existing-frames
"""

from __future__ import annotations

import argparse
import os
import shlex

try:
    from .prep_fingerprint import (
        build_fingerprint,
        diff_fingerprints,
        fingerprints_match,
        load_fingerprint,
        save_fingerprint,
    )
except ImportError:
    # Running as a script (not as a package); add scripts dir to sys.path.
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from prep_fingerprint import (
        build_fingerprint,
        diff_fingerprints,
        fingerprints_match,
        load_fingerprint,
        save_fingerprint,
    )
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def log(message: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_cmd(cmd: list[str], extra_env: dict[str, str] | None = None):
    printable = " ".join(shlex.quote(part) for part in cmd)
    log(f"[cmd] {printable}")
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    subprocess.run(cmd, check=True, env=env)


def common_ssh_options(*, use_mux: bool, control_persist: str) -> list[str]:
    # SSH multiplexing reuses one TCP connection for every ssh/rsync we run during a job.
    # Without this we'd re-auth on every step and HPG's login nodes get grumpy about that.
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


def rsync_ssh_option(port: int, identity_file: str | None, ssh_opts: list[str]) -> str:
    parts = ["ssh", "-p", str(port)]
    if identity_file:
        parts.extend(["-i", identity_file])
    parts.extend(ssh_opts)
    return " ".join(shlex.quote(p) for p in parts)


def latest_matching_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main():
    # End-to-end orchestrator for the Faster-GS backend.
    # Flow: local preprocess -> rsync images to HPG -> SLURM SfM+undistort
    # -> SLURM GPU training -> fetch .splat back -> publish into datasets/.
    # We keep preprocess local (faster turnaround + we already have OpenCV here)
    # and only push compute-heavy steps to HPG.
    parser = argparse.ArgumentParser(
        description="Orchestrate local prepare + HiPerGator Faster-GS prepare/train and publish .splat for frontend viewer."
    )
    parser.add_argument("dataset", help="Dataset name")
    # --video is optional when --use-existing-frames is passed: the
    # preprocessed images/ dir is assumed to already be in place and we
    # skip the prepare stage entirely. Without either flag the pipeline
    # still needs a video to extract frames from.
    parser.add_argument("--video", default=None, help="Local dataset video file path (omit when reusing existing images)")
    parser.add_argument(
        "--use-existing-frames",
        action="store_true",
        help="Skip frame extraction and reuse the existing images/ dir for this dataset.",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Faster-GS train iterations")
    parser.add_argument("--duplicate-threshold", type=float, default=1.5)
    parser.add_argument("--blur-threshold", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--downscale", type=float, default=0.75)
    parser.add_argument("--max-width", type=int, default=1280)

    parser.add_argument("--remote", default=os.getenv("FASTERGS_REMOTE", "hpg"), help="SSH target/alias")
    parser.add_argument(
        "--remote-root",
        default=os.getenv("FASTERGS_REMOTE_ROOT", "/blue/cis4914/joshuabowman/gs_final"),
        help="Remote gs_final root",
    )
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity-file", default=None)
    parser.add_argument("--no-ssh-mux", action="store_true")
    parser.add_argument("--ssh-control-persist", default="8h")

    parser.add_argument("--prepare-partition", default=os.getenv("FASTERGS_PREP_PARTITION", "hpg-default"))
    parser.add_argument("--prepare-cpus", type=int, default=int(os.getenv("FASTERGS_PREP_CPUS", "1")))
    parser.add_argument("--prepare-mem", default=os.getenv("FASTERGS_PREP_MEM", "8G"))
    parser.add_argument("--prepare-time", default=os.getenv("FASTERGS_PREP_TIME", "01:00:00"))

    parser.add_argument("--train-partition", default=os.getenv("FASTERGS_TRAIN_PARTITION", "hpg-turin"))
    parser.add_argument("--train-gpus", type=int, default=int(os.getenv("FASTERGS_TRAIN_GPUS", "1")))
    parser.add_argument("--train-cpus", type=int, default=int(os.getenv("FASTERGS_TRAIN_CPUS", "2")))
    parser.add_argument("--train-mem", default=os.getenv("FASTERGS_TRAIN_MEM", "24G"))
    parser.add_argument("--train-time", default=os.getenv("FASTERGS_TRAIN_TIME", "06:00:00"))
    parser.add_argument("--slurm-account", default=os.getenv("FASTERGS_SLURM_ACCOUNT", "cis4914"))

    # SfM method. colmap = traditional feature-extraction + matching on CPU
    # (8-12 min per dataset); vggt = feed-forward transformer on GPU that
    # produces the same COLMAP-format output in ~1 minute. Default stays
    # colmap so the behavior of existing workflows is unchanged.
    parser.add_argument("--sfm-method", choices=["colmap", "vggt"], default="colmap",
                        help="Which SfM backend to use on HPG. 'colmap' (default) runs COLMAP on CPU; "
                             "'vggt' runs the VGGT feed-forward transformer on a GPU.")
    # Bundle adjustment is ON by default for VGGT because without it the
    # demo writes zero 3D points and Faster-GS has nothing to seed gaussians
    # from. Pass --no-vggt-ba for the raw feed-forward output (cameras only,
    # useful for timing comparisons, NOT for actual training).
    parser.add_argument("--vggt-use-ba", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable VGGT bundle adjustment (default on). "
                             "Without BA, VGGT emits cameras but no 3D points.")

    # Shorter-Splatting paper flags. Off by default. These get forwarded
    # through to the trainer as env vars (SHORTGS_*). The vendored fork
    # currently ignores them, so they're no-ops until the fork is patched.
    # See backend/experiments/faster-gs/shortgs/README.md.
    parser.add_argument("--shortgs-scale-reset-every", type=int, default=0,
                        help="0 disables. Otherwise shrink gaussian scales every K iterations.")
    parser.add_argument("--shortgs-scale-reset-factor", type=float, default=1.0,
                        help="Multiplier applied to log-scale when scale-reset fires (<1 = shrink).")
    parser.add_argument("--shortgs-entropy-weight", type=float, default=0.0,
                        help="0 disables. Coefficient for the entropy-of-alpha loss term.")
    parser.add_argument("--shortgs-progressive-resolution", default="",
                        help="Empty disables. Schedule like '0:0.25,5000:0.5,10000:1.0'.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Pass a seed to the trainer and record it in metrics_summary.json.")
    # Orthogonal to the shortgs flags: flip the vendored fork's
    # USE_FASTERGS_ADAM line to True before training. The custom
    # rasterizer stays off either way (broken on sm_89+sm_100 as of
    # Apr 17). Default off so stock-baseline runs don't accidentally
    # pick up the fused kernel.
    parser.add_argument("--use-fastergs-adam", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable FasterGS's fused Adam optimizer (rasterizer stays disabled).")
    # Training backend selector. `fastergs` (default) runs the Inria fork
    # via the existing conda env + patch pipeline. `opensplat` invokes the
    # prebuilt C++ binary under /blue/cis4914/joshuabowman/gs_final/src/OpenSplat/;
    # shortgs + fastergs-adam flags are ignored in that branch because they
    # don't apply to OpenSplat's codebase.
    parser.add_argument("--backend", choices=["fastergs", "opensplat"], default="fastergs",
                        help="Which trainer to invoke on HPG (default fastergs).")

    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent.parent
    scripts_dir = backend_dir / "scripts"
    dataset_dir = backend_dir / "datasets" / args.dataset
    images_dir = dataset_dir / "images"

    fetch_dir = backend_dir / "hipergator" / "gs_final"
    fetch_dir.mkdir(parents=True, exist_ok=True)

    # Two input modes:
    #   1. Fresh video (--video ...) runs the full pipeline and extracts
    #      frames under images/ before SfM.
    #   2. Existing dataset (--use-existing-frames) skips frame extraction
    #      but STILL re-runs the preprocessor on raw/ using whatever
    #      blur / dup / downscale / max_width settings were passed. Lets
    #      users tweak filter knobs without re-uploading the video.
    prepare_cmd = [
        sys.executable,
        str(scripts_dir / "pipeline.py"),
        args.dataset,
        "--only",
        "prepare",
        "--duplicate-threshold",
        str(args.duplicate_threshold),
        "--blur-threshold",
        str(args.blur_threshold),
        "--fps",
        str(args.fps),
        "--downscale",
        str(args.downscale),
        "--max-width",
        str(args.max_width),
    ]
    if args.use_existing_frames:
        raw_dir = dataset_dir / "raw"
        if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
            if not images_dir.is_dir() or not any(images_dir.iterdir()):
                raise RuntimeError(
                    f"--use-existing-frames was set but no raw/ or images/ found under {dataset_dir}"
                )
            log(
                "WARN: No raw/ frames present; preprocessor will re-filter images/ in place (no downscale safety net)"
            )
        log("INFO: Re-running preprocessing on existing frames (skipping video extraction)")
        prepare_cmd.append("--use-existing-frames")
        run_cmd(prepare_cmd)
    else:
        if not args.video:
            raise RuntimeError("Either --video <path> or --use-existing-frames is required.")
        video_path = Path(args.video).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        log("INFO: Starting preprocessing")
        # Splice --video into the shared prepare_cmd right after the
        # dataset positional so pipeline.py sees the expected arg order.
        prepare_cmd = prepare_cmd[:3] + ["--video", str(video_path)] + prepare_cmd[3:]
        run_cmd(prepare_cmd)

    if not images_dir.exists() or not any(images_dir.iterdir()):
        raise RuntimeError(f"Preprocessing did not generate images: {images_dir}")
    log(f"INFO: Preprocessing finished ({len(list(images_dir.glob('*')))} files in images)")

    # Smart SfM Reuse: if we already ran SfM+undistort on HPG for this
    # dataset with the exact same preprocessing settings AND the remote
    # sparse model is still there, skip both the rsync AND the SLURM
    # prepare job. Saves ~10 minutes per follow-up run.
    ssh_opts = common_ssh_options(use_mux=not args.no_ssh_mux, control_persist=args.ssh_control_persist)
    ssh_base = ["ssh", "-p", str(args.port), *(["-i", args.identity_file] if args.identity_file else []), *ssh_opts]

    # Fingerprint includes sfm_method so the COLMAP and VGGT caches don't
    # collide — switching methods always triggers a fresh SfM pass, even
    # when the preprocessing settings are identical.
    remote_sfm_fp_path = dataset_dir / "remote_sfm_fingerprint.json"
    current_fp = build_fingerprint(
        fps=args.fps,
        downscale=args.downscale,
        blur_threshold=args.blur_threshold,
        duplicate_threshold=args.duplicate_threshold,
        max_width=args.max_width,
        extra={"remote_root": args.remote_root, "sfm_method": args.sfm_method},
    )
    prior_remote_fp = load_fingerprint(remote_sfm_fp_path)
    remote_sparse_marker = (
        f"{args.remote_root}/experiments/faster-gs/datasets/{args.dataset}/sparse/0/cameras.bin"
    )

    skip_remote_prepare = False
    prior_method = (prior_remote_fp or {}).get("sfm_method") or "colmap"
    if (
        fingerprints_match(prior_remote_fp, current_fp)
        and prior_remote_fp
        and prior_remote_fp.get("remote_root") == args.remote_root
        and prior_method == args.sfm_method
    ):
        # Fingerprint matches. Verify the remote sparse file actually exists
        # before skipping - covers the case where someone cleaned up /blue.
        check = subprocess.run(
            ssh_base + [args.remote, f"bash -lc {shlex.quote(f'test -f {remote_sparse_marker} && echo ok || echo no')}"],
            check=False, capture_output=True, text=True,
        )
        if "ok" in (check.stdout or ""):
            log(
                "INFO: SfM cache hit on HPG — reusing existing sparse model "
                f"(fps={args.fps} downscale={args.downscale} blur={args.blur_threshold} "
                f"duplicate={args.duplicate_threshold} max_width={args.max_width})"
            )
            skip_remote_prepare = True
        else:
            log("INFO: SfM fingerprint matched locally but remote sparse is missing; rerunning")
    elif prior_remote_fp:
        reasons = diff_fingerprints(prior_remote_fp, current_fp)
        for reason in reasons:
            log(f"INFO: SfM cache invalidated — {reason}")

    if skip_remote_prepare:
        log("INFO: Skipping image upload and HPG SfM (cached)")
    else:
        log("INFO: Starting Faster-GS sync (uploading preprocessed images to HiPerGator)")
        # make sure the target dirs exist on HPG before rsync tries to write into them.
        # bash -lc so the remote sees the login environment (modules etc) in case any are sourced.
        run_cmd(
            ssh_base
            + [
                args.remote,
                f"bash -lc {shlex.quote(f'mkdir -p {args.remote_root}/datasets/{args.dataset}/images {args.remote_root}/logs')}",
            ]
        )
        run_cmd(
            [
                "rsync",
                "-az",
                "--delete",
                "-e",
                rsync_ssh_option(args.port, args.identity_file, ssh_opts),
                str(images_dir) + "/",
                f"{args.remote}:{args.remote_root}/datasets/{args.dataset}/images/",
            ]
        )
        log("INFO: Faster-GS sync finished")

        if args.sfm_method == "vggt":
            # VGGT path: feed-forward transformer on a single GPU. Emits
            # PINHOLE-family cameras directly, so we can also skip the
            # COLMAP image_undistorter step (the scene dir produced here
            # is already trainer-ready). ~1 minute wall time vs ~10 min
            # for COLMAP on the same dataset.
            log("INFO: Starting VGGT SfM step on HiPerGator GPU (feed-forward transformer, typically under a minute)")
            vggt_cmd = [
                sys.executable,
                str(scripts_dir / "hpg_gs_vggt_sfm.py"),
                args.dataset,
                "--remote",
                args.remote,
                "--remote-root",
                args.remote_root,
                "--slurm-account",
                args.slurm_account,
                # VGGT needs GPU - use the same partition as training.
                "--slurm-partition",
                args.train_partition,
                "--preflight-script",
                f"{args.remote_root}/src/fastergs_preflight.py",
                "--port",
                str(args.port),
                "--ssh-control-persist",
                args.ssh_control_persist,
                *(["--identity-file", args.identity_file] if args.identity_file else []),
                *(["--no-ssh-mux"] if args.no_ssh_mux else []),
            ]
            if args.vggt_use_ba:
                vggt_cmd.append("--use-ba")
            run_cmd(vggt_cmd)
        else:
            log("INFO: Starting SfM step on HiPerGator (queue + SfM + undistort; typically several minutes)")
            run_cmd(
                [
                    sys.executable,
                    str(scripts_dir / "hpg_gs_final_prepare.py"),
                    args.dataset,
                    "--remote",
                    args.remote,
                    "--remote-root",
                    args.remote_root,
                    "--slurm-account",
                    args.slurm_account,
                    "--slurm-partition",
                    args.prepare_partition,
                    "--preflight-script",
                    f"{args.remote_root}/src/fastergs_preflight.py",
                    "--slurm-cpus",
                    str(args.prepare_cpus),
                    "--slurm-mem",
                    args.prepare_mem,
                    "--slurm-time",
                    args.prepare_time,
                    "--port",
                    str(args.port),
                    "--ssh-control-persist",
                    args.ssh_control_persist,
                    *(["--identity-file", args.identity_file] if args.identity_file else []),
                    *(["--no-ssh-mux"] if args.no_ssh_mux else []),
                ]
            )
        # Remote prepare succeeded - cache the fingerprint (now with
        # sfm_method baked in) so the next run with the same preprocessing
        # settings AND same sfm method can skip the rsync + SLURM job.
        save_fingerprint(remote_sfm_fp_path, current_fp)
        log(f"INFO: SfM step finished successfully (method={args.sfm_method})")

    log(f"INFO: Starting Gaussian Splatting ({args.backend}) on HiPerGator GPU (may take 10-60+ minutes)")
    # Build the shortgs env so hpg_gs_final_train.py can inject them as
    # env exports in the SLURM bash. Empty/zero values mean "technique off".
    # OpenSplat doesn't read these, but we still propagate the seed so its
    # metrics_summary.json carries the same field for cross-backend comparison.
    shortgs_env = {}
    active_techniques = []
    if args.backend == "fastergs":
        if args.shortgs_scale_reset_every and args.shortgs_scale_reset_every > 0:
            shortgs_env["SHORTGS_SCALE_RESET_EVERY"] = str(args.shortgs_scale_reset_every)
            shortgs_env["SHORTGS_SCALE_RESET_FACTOR"] = str(args.shortgs_scale_reset_factor)
            active_techniques.append("sr")
        if args.shortgs_entropy_weight and args.shortgs_entropy_weight > 0:
            shortgs_env["SHORTGS_ENTROPY_WEIGHT"] = str(args.shortgs_entropy_weight)
            active_techniques.append("ent")
        if args.shortgs_progressive_resolution:
            shortgs_env["SHORTGS_PROGRESSIVE_RESOLUTION"] = args.shortgs_progressive_resolution
            active_techniques.append("pr")
    if args.seed is not None:
        shortgs_env["SHORTGS_SEED"] = str(args.seed)

    # Build a short run label so run_tags are self-describing, e.g.:
    #   test123456_s1-baseline_train_20260416_195114
    #   test123456_s1-shortgs-sr-ent-pr_train_20260416_195114
    #   test123456_s1-fastergsadam_train_20260417_160000
    #   test123456_s1-opensplat_train_20260417_170000
    # Tags get embedded in file paths, so stick to [A-Za-z0-9_.-].
    label_parts = []
    if args.seed is not None:
        label_parts.append(f"s{args.seed}")
    if args.backend == "opensplat":
        label_parts.append("opensplat")
    elif active_techniques:
        label_parts.append("shortgs-" + "-".join(active_techniques))
    elif args.use_fastergs_adam:
        label_parts.append("fastergsadam")
    else:
        label_parts.append("baseline")
    run_label = "-".join(label_parts)

    run_cmd(
        [
            sys.executable,
            str(scripts_dir / "hpg_gs_final_train.py"),
            args.dataset,
            "--remote",
            args.remote,
            "--remote-root",
            args.remote_root,
            "--slurm-account",
            args.slurm_account,
            "--slurm-partition",
            args.train_partition,
            "--slurm-gpus",
            str(args.train_gpus),
            "--slurm-cpus",
            str(args.train_cpus),
            "--slurm-mem",
            args.train_mem,
            "--slurm-time",
            args.train_time,
            "--stage",
            "train",
            "--reuse-env",
            "--iterations",
            str(args.iters),
            "--fetch-local-dir",
            str(fetch_dir),
            "--run-label",
            run_label,
            "--port",
            str(args.port),
            "--ssh-control-persist",
            args.ssh_control_persist,
            *(["--identity-file", args.identity_file] if args.identity_file else []),
            *(["--no-ssh-mux"] if args.no_ssh_mux else []),
            *(["--use-fastergs-adam"] if args.use_fastergs_adam and args.backend == "fastergs" else []),
            "--backend", args.backend,
        ],
        extra_env=shortgs_env if shortgs_env else None,
    )
    log("INFO: Gaussian Splatting finished successfully")

    # Training writes timestamped artifacts (so we can keep old runs around).
    # We pick the most recent one and publish it to two stable paths:
    #   datasets/<ds>/splat.splat  -> what the frontend viewer loads for this dataset
    #   hipergator/<ds>_fastergs_latest.splat -> what the gallery page lists
    # Pattern is loose ({dataset}_*) because the run_tag now includes an
    # optional run label between the dataset and the stage (e.g.
    # "{dataset}_s1-baseline_train_{stamp}.splat"). mtime sort picks the
    # most recent matching artifact.
    latest_splat = latest_matching_file(fetch_dir, f"{args.dataset}_*.splat")
    latest_ply = latest_matching_file(fetch_dir, f"{args.dataset}_*.ply")
    if latest_splat is None:
        raise RuntimeError(f"No fetched Faster-GS .splat found in {fetch_dir} for dataset {args.dataset}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_dataset_splat = dataset_dir / "splat.splat"
    shutil.copy2(latest_splat, target_dataset_splat)

    # Tag the gallery copy with the backend so fastergs + opensplat runs for
    # the same dataset don't clobber each other's "latest" marker.
    gallery_tag = "opensplat" if args.backend == "opensplat" else "fastergs"
    target_gallery_splat = backend_dir / "hipergator" / f"{args.dataset}_{gallery_tag}_latest.splat"
    shutil.copy2(latest_splat, target_gallery_splat)

    if latest_ply is not None:
        shutil.copy2(latest_ply, dataset_dir / "splat.ply")

    log(f"INFO: Final .splat published to {target_dataset_splat}")
    log(f"INFO: Gallery .splat published to {target_gallery_splat}")

    # Local fallback plotting pass. hpg_gs_final_train.py already runs the
    # plotter on HPG, but if matplotlib isn't installed in the remote env the
    # PNGs won't be there. We look for any run dirs that have metrics.jsonl
    # but no psnr.png and render them here.
    metrics_root = dataset_dir / "metrics"
    plotter_path = scripts_dir / "metrics_plotter.py"
    if metrics_root.is_dir() and plotter_path.is_file():
        for run_metrics_dir in sorted(metrics_root.iterdir()):
            if not run_metrics_dir.is_dir():
                continue
            if not (run_metrics_dir / "metrics.jsonl").is_file():
                continue
            if (run_metrics_dir / "psnr.png").is_file():
                continue
            try:
                log(f"INFO: Rendering metrics PNGs locally for {run_metrics_dir.name}")
                subprocess.run(
                    [
                        sys.executable,
                        str(plotter_path),
                        "--metrics-dir",
                        str(run_metrics_dir),
                        "--title-prefix",
                        f"{args.dataset} {run_metrics_dir.name}",
                    ],
                    check=True,
                )
            except Exception as exc:
                log(f"WARN: Local metrics plotter failed for {run_metrics_dir.name}: {exc}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
