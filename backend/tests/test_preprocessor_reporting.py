from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.scripts.preprocessor import preprocessor


class PreprocessorReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.raw = self.root / "raw"
        self.out = self.root / "images"
        self.raw.mkdir(parents=True, exist_ok=True)

        rng = np.random.default_rng(7)
        sharp = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        duplicate = sharp.copy()
        blurry = np.full((120, 160, 3), 127, dtype=np.uint8)

        cv2.imwrite(str(self.raw / "frame_000000.jpg"), sharp)
        cv2.imwrite(str(self.raw / "frame_000001.jpg"), duplicate)
        cv2.imwrite(str(self.raw / "frame_000002.jpg"), blurry)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_preprocessor_generates_scores_and_stats(self):
        scores_path = self.root / "preprocess_scores.csv"
        stats = preprocessor(
            self.raw,
            self.out,
            duplicate_threshold=0.1,
            blur_threshold=1.0,
            max_output_width=1280,
            write_scores_csv=True,
            scores_csv_path=scores_path,
        )

        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["skipped_duplicate"], 1)
        self.assertEqual(stats["skipped_blur"], 1)
        self.assertTrue(scores_path.exists())

        with open(scores_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        reasons = {r["reason"] for r in rows}
        self.assertIn("kept", reasons)
        self.assertIn("duplicate", reasons)
        self.assertIn("blur", reasons)


if __name__ == "__main__":
    unittest.main()
