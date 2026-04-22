from __future__ import annotations

import unittest
from argparse import Namespace

from backend.scripts.pipeline import _suspicious_warnings, validate_run_args


class PipelineValidationTests(unittest.TestCase):
    def _args(self, **overrides):
        base = Namespace(
            dataset="demo_dataset",
            img_format="jpg",
            iters=1000,
            duplicate_threshold=0.0,
            blur_threshold=0.0,
            fps=0.0,
            downscale=1.0,
            max_width=1280,
        )
        for key, value in overrides.items():
            setattr(base, key, value)
        return base

    def test_validate_run_args_accepts_defaults(self):
        validate_run_args(self._args())

    def test_validate_run_args_rejects_invalid_dataset(self):
        with self.assertRaises(ValueError):
            validate_run_args(self._args(dataset="../bad"))

    def test_validate_run_args_rejects_invalid_downscale(self):
        with self.assertRaises(ValueError):
            validate_run_args(self._args(downscale=1.5))

    def test_validate_run_args_rejects_invalid_iters(self):
        with self.assertRaises(ValueError):
            validate_run_args(self._args(iters=10))

    def test_suspicious_warnings_detect_overly_high_keep_ratio(self):
        warnings = _suspicious_warnings(
            raw_count=400,
            prep_stats={
                "total": 380,
                "kept": 378,
                "skipped_blur": 0,
                "skipped_duplicate": 0,
            },
        )
        self.assertTrue(any("keep ratio" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main()
