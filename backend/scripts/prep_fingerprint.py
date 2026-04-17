"""
Fingerprint helpers for the Smart SfM Reuse feature.

A "prep fingerprint" records the exact preprocessing settings used for a
dataset's last successful frame-extraction / blur-duplicate-filter / SfM
pass. On the next run we compare the new settings to the stored fingerprint
and either reuse the existing artifacts (saving 10+ minutes) or rerun with
a logged reason for what changed.

Fingerprint settings (changes here trigger reprocessing):
  fps, downscale, blur_threshold, duplicate_threshold, max_width

Settings that do NOT go into the fingerprint (safe to vary without rerunning
SfM): iterations, seed, backend, scale_reset_*, entropy_weight,
progressive_resolution.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path


FINGERPRINT_VERSION = 1

# The fields that get hashed into the fingerprint. Order-independent; keys
# are listed here purely for documentation and for the diff helper below.
FINGERPRINT_FIELDS = (
    "fps",
    "downscale",
    "blur_threshold",
    "duplicate_threshold",
    "max_width",
)


def build_fingerprint(*, fps, downscale, blur_threshold, duplicate_threshold,
                     max_width, extra=None):
    """Produce a fingerprint dict from the current run's preprocessing settings."""
    d = {
        "version": FINGERPRINT_VERSION,
        "fps": float(fps if fps is not None else 0.0),
        "downscale": float(downscale if downscale is not None else 1.0),
        "blur_threshold": float(blur_threshold if blur_threshold is not None else 0.0),
        "duplicate_threshold": float(duplicate_threshold if duplicate_threshold is not None else 0.0),
        "max_width": int(max_width if max_width is not None else 0),
        "completed_at_epoch": time.time(),
    }
    if extra:
        d.update(extra)
    return d


def _close(a, b):
    # Float-safe equality for the fingerprint comparison. A 1e-6 slack
    # catches user-typed values that round-trip through JSON slightly
    # differently ("0.75" -> 0.75 vs "0.7500001") without letting real
    # changes through.
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    except Exception:
        return a == b


def diff_fingerprints(old: dict, new: dict) -> list[str]:
    """Return a list of human-readable "<field> changed (<old> -> <new>)" strings."""
    if not old:
        return ["no prior fingerprint"]
    diffs = []
    for field in FINGERPRINT_FIELDS:
        ov = old.get(field)
        nv = new.get(field)
        if ov is None and nv is None:
            continue
        if ov is None or nv is None or not _close(ov, nv):
            diffs.append(f"{field} changed ({ov} -> {nv})")
    if old.get("version") != new.get("version"):
        diffs.append(f"fingerprint schema version changed ({old.get('version')} -> {new.get('version')})")
    return diffs


def load_fingerprint(path: Path) -> dict | None:
    """Read a fingerprint JSON file. Returns None if missing or corrupt."""
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text())
    except Exception:
        return None


def save_fingerprint(path: Path, fingerprint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2))


def fingerprints_match(old: dict | None, new: dict) -> bool:
    """True iff every tracked field matches (within float tolerance)."""
    if not old:
        return False
    for field in FINGERPRINT_FIELDS:
        if not _close(old.get(field), new.get(field)):
            return False
    return True
