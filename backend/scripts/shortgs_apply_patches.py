#!/usr/bin/env python3
"""
Apply Shorter-Splatting paper techniques to the vendored Inria Faster-GS fork.

Runs on HiPerGator inside the SLURM training job, right after `ensure_repo`
clones/updates the fork but before `python train.py` is invoked. Each patch is
a plain find-and-replace on the Inria source; we wrap every insertion in a
`# === SHORTGS PATCH ===` marker so this script is idempotent (skips files
that are already patched).

All three techniques are behind env vars, so the stock behavior is unchanged
when no SHORTGS_* vars are set:

  SHORTGS_SCALE_RESET_EVERY        int, 0 disables
  SHORTGS_SCALE_RESET_FACTOR       float, multiplicative factor applied every K iters
  SHORTGS_ENTROPY_WEIGHT           float, 0 disables
  SHORTGS_PROGRESSIVE_RESOLUTION   e.g. "0:0.25,5000:0.5,10000:1.0"; empty disables

Implementation notes:

- Scale reset: adds log(factor) to `gaussians._scaling.data` every K
  iterations. `_scaling` is stored in log-space by Inria, so addition in
  log-space is multiplication in linear space.
- Entropy constraint: the paper targets "per-pixel alpha blending entropy"
  which would need the rasterizer to expose per-pixel alpha lists - the
  stock and Faster-GS rasterizers here don't. So we apply the SAME shape of
  penalty (Bernoulli entropy peaking at opacity=0.5) to each gaussian's
  scalar opacity. Minimizing this pushes opacities towards 0 or 1, which is
  exactly the outcome the paper is chasing (polarized opacities -> fewer
  gaussians dominate each pixel -> shorter alpha blend lists). Not
  bit-for-bit identical to the paper's formulation; we document this gap.
- Progressive resolution: downsamples the rendered image AND the ground
  truth before the L1/SSIM loss computation. Does NOT change render
  resolution (that would require mutating camera intrinsics mid-training),
  so actual forward-pass GPU time is unchanged - but loss gradients are
  computed at lower effective resolution, which changes training dynamics
  and slightly speeds up the loss step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PATCH_MARKER = "# === SHORTGS PATCH ==="


# Block inserted near the top of training(). Reads env vars, parses the
# progressive-resolution schedule, prints a one-line summary of whatever is
# enabled. Everything here is prefixed with _shortgs or _SHORTGS so it can't
# collide with names already defined in Inria's train.py.
SHORTGS_SETUP = """\
    # === SHORTGS PATCH ===
    # Read Shorter-Splatting paper flags from environment. When all three are
    # unset (the normal case), these values disable every addition below and
    # training runs exactly like the stock Inria trainer.
    import math as _shortgs_math
    import torch.nn.functional as _shortgs_F
    _SHORTGS_SCALE_RESET_EVERY = int(os.environ.get("SHORTGS_SCALE_RESET_EVERY", "0") or "0")
    _SHORTGS_SCALE_RESET_FACTOR = float(os.environ.get("SHORTGS_SCALE_RESET_FACTOR", "1.0") or "1.0")
    _SHORTGS_ENTROPY_WEIGHT = float(os.environ.get("SHORTGS_ENTROPY_WEIGHT", "0.0") or "0.0")
    _SHORTGS_PROGRESSIVE_SCHEDULE = []
    _sched_str = os.environ.get("SHORTGS_PROGRESSIVE_RESOLUTION", "")
    if _sched_str:
        try:
            # parse "0:0.25,5000:0.5,10000:1.0" -> [(0, 0.25), (5000, 0.5), (10000, 1.0)]
            for _pair in _sched_str.split(","):
                _i, _s = _pair.split(":")
                _SHORTGS_PROGRESSIVE_SCHEDULE.append((int(_i.strip()), float(_s.strip())))
            _SHORTGS_PROGRESSIVE_SCHEDULE.sort(key=lambda p: p[0])
        except Exception as _e:
            print(f"[shortgs] warning: could not parse SHORTGS_PROGRESSIVE_RESOLUTION='{_sched_str}': {_e}")
            _SHORTGS_PROGRESSIVE_SCHEDULE = []
    if _SHORTGS_SCALE_RESET_EVERY > 0:
        print(f"[shortgs] scale reset every {_SHORTGS_SCALE_RESET_EVERY} iters by factor {_SHORTGS_SCALE_RESET_FACTOR}")
    if _SHORTGS_ENTROPY_WEIGHT > 0:
        print(f"[shortgs] entropy loss weight {_SHORTGS_ENTROPY_WEIGHT} (per-gaussian opacity bernoulli entropy)")
    if _SHORTGS_PROGRESSIVE_SCHEDULE:
        print(f"[shortgs] progressive resolution schedule: {_SHORTGS_PROGRESSIVE_SCHEDULE}")
    # === END SHORTGS PATCH ===
"""


# Inserted right after gt_image is loaded and BEFORE Ll1/SSIM are computed.
# Downscales both render and GT when a progressive schedule is active.
PROGRESSIVE_RES_BLOCK = """\
        # === SHORTGS PATCH ===
        # Progressive resolution: if the schedule is enabled, downscale
        # image + gt_image before computing L1/SSIM so loss is evaluated at
        # a lower effective resolution for early iterations. Render is still
        # produced at full res (camera intrinsics aren't mutated), so this
        # doesn't reduce forward-pass GPU time - but it does change the
        # gradient signal and reduce loss-step compute.
        _shortgs_res_scale = 1.0
        for _boundary, _scale in _SHORTGS_PROGRESSIVE_SCHEDULE:
            if iteration >= _boundary:
                _shortgs_res_scale = _scale
        if _shortgs_res_scale < 0.999:
            image = _shortgs_F.interpolate(image.unsqueeze(0), scale_factor=_shortgs_res_scale,
                                           mode="bilinear", align_corners=False, antialias=True).squeeze(0)
            gt_image = _shortgs_F.interpolate(gt_image.unsqueeze(0), scale_factor=_shortgs_res_scale,
                                              mode="bilinear", align_corners=False, antialias=True).squeeze(0)
        # === END SHORTGS PATCH ===
"""


# Inserted after the base loss is computed and BEFORE loss.backward().
# Adds the entropy regularization term.
ENTROPY_BLOCK = """\
        # === SHORTGS PATCH ===
        # Entropy constraint on gaussian opacities. Bernoulli entropy peaks
        # at opacity=0.5 and is zero at 0 or 1, so adding it to the loss and
        # minimizing pushes every gaussian towards fully-on or fully-off.
        # The paper targets per-pixel alpha entropy (needs rasterizer hook);
        # this per-gaussian version is a faithful proxy that produces the
        # same polarized-opacity outcome the paper is chasing.
        if _SHORTGS_ENTROPY_WEIGHT > 0:
            _op = gaussians.get_opacity.clamp(1e-8, 1.0 - 1e-8)
            _ent = -(_op * _op.log() + (1.0 - _op) * (1.0 - _op).log())
            loss = loss + _SHORTGS_ENTROPY_WEIGHT * _ent.mean()
        # === END SHORTGS PATCH ===
"""


# Inserted inside the `with torch.no_grad():` block that follows loss.backward,
# after the densification + opacity-reset logic. This is where we apply the
# periodic scale reset.
SCALE_RESET_BLOCK = """\
            # === SHORTGS PATCH ===
            # Periodically shrink every gaussian's scale by a multiplicative
            # factor. _scaling is stored in log-space so we add log(factor);
            # factor<1 -> log<0 -> scales shrink. Paper's claim: smaller
            # gaussians cover fewer pixels -> shorter per-pixel alpha blend
            # lists -> faster training and render.
            if _SHORTGS_SCALE_RESET_EVERY > 0 and iteration > 0 and iteration % _SHORTGS_SCALE_RESET_EVERY == 0:
                with torch.no_grad():
                    gaussians._scaling.data += _shortgs_math.log(max(_SHORTGS_SCALE_RESET_FACTOR, 1e-6))
                print(f"[shortgs] iter {iteration}: scale reset applied (factor {_SHORTGS_SCALE_RESET_FACTOR})")
            # === END SHORTGS PATCH ===
"""


def _replace_once(text: str, needle: str, replacement: str, *, where: str) -> str:
    # Strict in-place replace that refuses to touch text when the anchor
    # isn't found. Loud failure is better than a silent no-op.
    if needle not in text:
        raise RuntimeError(f"{where}: anchor not found in source:\n  {needle!r}")
    count = text.count(needle)
    if count != 1:
        raise RuntimeError(f"{where}: anchor is not unique (found {count} occurrences)")
    return text.replace(needle, replacement, 1)


def patch_train_py(path: Path) -> bool:
    text = path.read_text()
    if PATCH_MARKER in text:
        print(f"[shortgs] {path}: already patched, skipping")
        return False

    # 1) Setup block: insert just before "ema_loss_for_log = 0.0" (stable anchor
    #    near the top of training() that sits above the training loop).
    anchor1 = "    ema_loss_for_log = 0.0"
    text = _replace_once(
        text,
        anchor1,
        SHORTGS_SETUP + anchor1,
        where="train.py setup block",
    )

    # 2) Progressive resolution: insert right before the base L1/SSIM loss.
    #    We use "Ll1 = l1_loss(image, gt_image)" as the anchor; the downscale
    #    block must run just before it so both tensors end up at the reduced
    #    resolution. Indent matches the surrounding loop body.
    anchor2 = "        Ll1 = l1_loss(image, gt_image)"
    text = _replace_once(
        text,
        anchor2,
        PROGRESSIVE_RES_BLOCK + anchor2,
        where="train.py progressive-resolution block",
    )

    # 3) Entropy: insert right before loss.backward(). Entropy term is added
    #    to the already-computed loss so it flows through the single backward.
    anchor3 = "        loss.backward()"
    text = _replace_once(
        text,
        anchor3,
        ENTROPY_BLOCK + anchor3,
        where="train.py entropy block",
    )

    # 4) Scale reset: insert after the Densification block, right before
    #    "# Optimizer step" comment. We run the reset at the end of the
    #    no_grad section so _scaling.data gets modified cleanly between
    #    the densify step and the next iteration's forward pass.
    anchor4 = "            # Optimizer step"
    text = _replace_once(
        text,
        anchor4,
        SCALE_RESET_BLOCK + anchor4,
        where="train.py scale-reset block",
    )

    path.write_text(text)
    print(f"[shortgs] {path}: patched (4 inserts)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Apply Shorter-Splatting patches to vendored Inria fork.")
    parser.add_argument("--repo-dir", required=True, help="Path to the fastergs_inria repo on HPG")
    parser.add_argument("--dry-run", action="store_true", help="Report but don't write changes")
    args = parser.parse_args()

    repo = Path(args.repo_dir).resolve()
    train_py = repo / "train.py"
    if not train_py.is_file():
        print(f"[shortgs] ERROR: {train_py} not found", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        text = train_py.read_text()
        if PATCH_MARKER in text:
            print(f"[shortgs] dry-run: {train_py} already patched")
        else:
            print(f"[shortgs] dry-run: {train_py} WOULD be patched with 4 inserts")
        return

    patch_train_py(train_py)
    print("[shortgs] done")


if __name__ == "__main__":
    # Clean single-line error so the SLURM log stays readable for teammates.
    # Training will continue with unpatched train.py (stock behavior) per
    # the wrapper bash "|| echo [warn] ... continuing with stock train.py".
    try:
        main()
    except Exception as exc:
        print(f"[shortgs] error: {exc}", file=sys.stderr)
        sys.exit(1)
