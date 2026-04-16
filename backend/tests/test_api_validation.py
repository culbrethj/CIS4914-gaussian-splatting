from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import backend.main as main


class _DummyStdout:
    def __init__(self):
        self._lines = [b"INFO: step\n", b""]

    async def readline(self):
        return self._lines.pop(0)


class _DummyProcess:
    def __init__(self, returncode: int = 0):
        self.stdout = _DummyStdout()
        self.returncode = returncode

    async def wait(self):
        return self.returncode


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_datasets_dir = main.datasets_dir

        main.datasets_dir = Path(self.tmpdir.name) / "datasets"
        main.datasets_dir.mkdir(parents=True, exist_ok=True)
        main.JOB_META.clear()

    def tearDown(self):
        main.datasets_dir = self.orig_datasets_dir
        main.JOB_META.clear()
        self.tmpdir.cleanup()

    def test_validate_dataset_rejects_path_traversal(self):
        with self.assertRaises(HTTPException):
            main._validate_dataset_name("../evil")

    def test_parse_run_payload_rejects_invalid_iters(self):
        with self.assertRaises(HTTPException):
            main._parse_run_payload({"dataset": "demo", "iters": 10})

    def test_parse_run_payload_accepts_advanced_fields(self):
        parsed = main._parse_run_payload(
            {
                "dataset": "demo",
                "iters": 2000,
                "duplicate_threshold": 3.5,
                "blur_threshold": 42,
                "fps": 24,
                "downscale": 0.5,
                "only": "all",
            }
        )
        self.assertEqual(parsed["dataset"], "demo")
        self.assertEqual(parsed["iters"], 2000)
        self.assertEqual(parsed["downscale"], 0.5)

    def test_pick_dataset_video_ignores_unsupported_extensions(self):
        ds = main.datasets_dir / "demo" / "video"
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "notes.txt").write_text("not a video", encoding="utf-8")
        (ds / "video.mp4").write_bytes(b"video")

        picked = main._pick_dataset_video_file("demo")
        self.assertEqual(picked.name, "video.mp4")

    def test_detect_stage_from_line(self):
        self.assertEqual(main._detect_stage_from_line("INFO: Starting video slicing"), "prepare")
        self.assertEqual(main._detect_stage_from_line("INFO: Starting SfM step"), "sfm")
        self.assertEqual(main._detect_stage_from_line("INFO: Starting Gaussian Splatting (opensplat)"), "opensplat")
        self.assertIsNone(main._detect_stage_from_line("INFO: random log line"))

    def test_refresh_job_preview_populates_preview_fields(self):
        ds = main.datasets_dir / "preview_demo"
        (ds / "images").mkdir(parents=True, exist_ok=True)
        (ds / "sparse").mkdir(parents=True, exist_ok=True)
        (ds / "images" / "frame_000001.jpg").write_bytes(b"img")
        (ds / "sparse" / "output_cloud.ply").write_bytes(b"ply")
        (ds / "result.splat").write_bytes(b"splat")

        main.JOB_META["job-preview"] = {"dataset": "preview_demo"}
        main._refresh_job_preview("job-preview")
        meta = main.JOB_META["job-preview"]

        self.assertEqual(meta["preview_image_path"], "/datasets/preview_demo/images/frame_000001.jpg")
        self.assertEqual(meta["preview_cloud_path"], "/datasets/preview_demo/sparse/output_cloud.ply")
        self.assertEqual(meta["final_splat_path"], "/datasets/preview_demo/result.splat")
        self.assertEqual(meta["processed_image_count"], 1)


class RunEndpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.orig_datasets_dir = main.datasets_dir

        main.datasets_dir = Path(self.tmpdir.name) / "datasets"
        main.datasets_dir.mkdir(parents=True, exist_ok=True)

        main.JOB_QUEUES.clear()
        main.JOB_TASKS.clear()
        main.JOB_META.clear()

    async def asyncTearDown(self):
        for task in list(main.JOB_TASKS.values()):
            if not task.done():
                task.cancel()
        main.JOB_QUEUES.clear()
        main.JOB_TASKS.clear()
        main.JOB_META.clear()

        main.datasets_dir = self.orig_datasets_dir
        self.tmpdir.cleanup()

    async def test_run_pipeline_starts_job_and_includes_advanced_args(self):
        ds = main.datasets_dir / "demo_dataset" / "video"
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "clip.mp4").write_bytes(b"video")

        create_proc = AsyncMock(return_value=_DummyProcess(returncode=0))

        with patch("backend.main.asyncio.create_subprocess_exec", new=create_proc):
            response = await main.run_pipeline(
                {
                    "dataset": "demo_dataset",
                    "iters": 1200,
                    "only": "all",
                    "duplicate_threshold": 2.5,
                    "blur_threshold": 55,
                    "fps": 12,
                    "downscale": 0.5,
                }
            )

        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body.decode("utf-8"))
        self.assertIn("job_id", body)

        cmd_args = create_proc.await_args.args
        joined = " ".join(str(part) for part in cmd_args)
        self.assertIn("--duplicate-threshold 2.5", joined)
        self.assertIn("--blur-threshold 55.0", joined)
        self.assertIn("--fps 12.0", joined)
        self.assertIn("--downscale 0.5", joined)
        self.assertIn("--max-width 1280", joined)


if __name__ == "__main__":
    unittest.main()
