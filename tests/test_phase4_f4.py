"""Unit tests for Phase 4 Route F utilities."""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import subprocess

import cv2
import numpy as np

from src.part3.geometry_cue import load_external_masks
from src.part3.mask_fusion import (
    SUPPORTED_FUSION_METHODS,
    apply_fusion,
    fuse_vggt4d_guided,
    fuse_video_vggt4d_guided,
)
from src.part3.motion_flow import mask_flow_reliability
from src.part3.motion_flow import apply_trajectory_consistency
from src.part3.vggt_prior import (
    build_vggt4d_runner_command,
    generate_vggt4d_dynamic_priors,
)


class TestMaskFlowReliability(unittest.TestCase):
    def test_all_reliable(self):
        motion = np.ones((32, 32), dtype=np.float32) * 5.0
        consistency = np.zeros((32, 32), dtype=np.float32)
        out = mask_flow_reliability(motion, consistency, max_consistency_error=3.0)
        np.testing.assert_array_equal(out, motion)

    def test_all_unreliable(self):
        motion = np.ones((32, 32), dtype=np.float32) * 5.0
        consistency = np.ones((32, 32), dtype=np.float32) * 10.0
        out = mask_flow_reliability(motion, consistency, max_consistency_error=3.0)
        self.assertEqual(float(out.max()), 0.0)

    def test_threshold_boundary(self):
        motion = np.ones((8, 8), dtype=np.float32)
        consistency = np.ones((8, 8), dtype=np.float32) * 3.0
        out = mask_flow_reliability(motion, consistency, max_consistency_error=3.0)
        self.assertEqual(float(out.max()), 0.0)


class TestVggtFusion(unittest.TestCase):
    def test_registered(self):
        self.assertIn("vggt4d_guided", SUPPORTED_FUSION_METHODS)

    def test_fuse_vggt4d_guided(self):
        sem = np.ones((32, 32), dtype=np.uint8) * 255
        motion = np.ones((32, 32), dtype=np.float32) * 4.0
        ext = np.ones((32, 32), dtype=np.uint8) * 255
        out = fuse_vggt4d_guided(sem, motion, ext, alpha=0.3, beta=0.3, threshold=0.5)
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))
        self.assertGreater(int((out > 0).sum()), 0)

    def test_video_level(self):
        n = 4
        sem = [np.ones((16, 16), dtype=np.uint8) * 255 for _ in range(n)]
        mm = [np.ones((16, 16), dtype=np.float32) * 2.0 for _ in range(n)]
        ext = [np.ones((16, 16), dtype=np.uint8) * 255 for _ in range(n)]
        out = fuse_video_vggt4d_guided(sem, mm, ext)
        self.assertEqual(len(out), n)

    def test_apply_requires_external(self):
        sem = [np.ones((16, 16), dtype=np.uint8) * 255]
        mm = [np.ones((16, 16), dtype=np.float32) * 2.0]
        scores = [2.0]
        with self.assertRaises(ValueError):
            apply_fusion(sem, mm, scores, method="vggt4d_guided")


class TestExternalMaskLoader(unittest.TestCase):
    def test_load_resize_pad(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ds = root / "tennis"
            ds.mkdir(parents=True, exist_ok=True)
            m = np.zeros((64, 64), dtype=np.uint8)
            m[10:30, 10:30] = 255
            cv2.imwrite(str(ds / "0000.png"), m)

            masks = load_external_masks(root, "tennis", (32, 32), 3)
            self.assertIsNotNone(masks)
            assert masks is not None
            self.assertEqual(len(masks), 3)
            self.assertEqual(masks[0].shape, (32, 32))
            self.assertGreater(int((masks[0] > 0).sum()), 0)
            self.assertEqual(int((masks[2] > 0).sum()), 0)


class TestVggt4dPriorRunner(unittest.TestCase):
    def test_build_runner_command(self):
        cmd = build_vggt4d_runner_command(
            repo_dir=Path("/tmp/vggt4d"),
            script_relpath="run_vggt4d_chunked.py",
            env_name="vggt4d",
            input_dir=Path("/tmp/in"),
            output_dir=Path("/tmp/out"),
            chunk_size=32,
            datasets=["wild", "tennis"],
        )
        self.assertEqual(cmd[:5], ["conda", "run", "-n", "vggt4d", "python"])
        self.assertIn("--input_dir", cmd)
        self.assertIn("--output_dir", cmd)
        self.assertIn("--chunk_size", cmd)
        self.assertIn("--datasets", cmd)
        self.assertIn("wild,tennis", cmd)

    def test_generate_aligns_output_masks(self):
        frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_dir = root / "repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "dummy_runner.py").write_text("print('dummy')\n", encoding="utf-8")
            out_root = root / "out"

            def fake_run(*args, **kwargs):
                ds_dir = out_root / "wild"
                ds_dir.mkdir(parents=True, exist_ok=True)
                m = np.zeros((16, 16), dtype=np.uint8)
                m[4:12, 4:12] = 255
                cv2.imwrite(str(ds_dir / "dynamic_mask_0000.png"), m)
                return subprocess.CompletedProcess(args[0], 0, "ok", "")

            with patch("src.part3.vggt_prior.subprocess.run", side_effect=fake_run):
                priors, meta = generate_vggt4d_dynamic_priors(
                    datasets_frames_bgr={"wild": frames},
                    output_root=out_root,
                    cfg={
                        "env_name": "vggt4d",
                        "repo_dir": str(repo_dir),
                        "script_relpath": "dummy_runner.py",
                        "chunk_size": 4,
                        "strict_backend": True,
                    },
                )

            self.assertIn("wild", priors)
            self.assertEqual(len(priors["wild"]), 3)
            self.assertEqual(priors["wild"][0].shape, (32, 32))
            self.assertGreater(int((priors["wild"][0] > 0).sum()), 0)
            self.assertEqual(int((priors["wild"][1] > 0).sum()), 0)
            self.assertEqual(int((priors["wild"][2] > 0).sum()), 0)
            self.assertEqual(meta.get("backend"), "vggt4d")

    def test_generate_strict_failure_raises(self):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_dir = root / "repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "dummy_runner.py").write_text("print('dummy')\n", encoding="utf-8")

            def fake_fail(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 1, "", "runner failed")

            with patch("src.part3.vggt_prior.subprocess.run", side_effect=fake_fail):
                with self.assertRaises(RuntimeError):
                    generate_vggt4d_dynamic_priors(
                        datasets_frames_bgr={"wild": frames},
                        output_root=root / "out",
                        cfg={
                            "env_name": "vggt4d",
                            "repo_dir": str(repo_dir),
                            "script_relpath": "dummy_runner.py",
                            "chunk_size": 4,
                            "strict_backend": True,
                        },
                    )

    def test_generate_missing_output_raises(self):
        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo_dir = root / "repo"
            repo_dir.mkdir(parents=True, exist_ok=True)
            (repo_dir / "dummy_runner.py").write_text("print('dummy')\n", encoding="utf-8")

            def fake_ok(*args, **kwargs):
                return subprocess.CompletedProcess(args[0], 0, "ok", "")

            with patch("src.part3.vggt_prior.subprocess.run", side_effect=fake_ok):
                with self.assertRaises(RuntimeError):
                    generate_vggt4d_dynamic_priors(
                        datasets_frames_bgr={"wild": frames},
                        output_root=root / "out",
                        cfg={
                            "env_name": "vggt4d",
                            "repo_dir": str(repo_dir),
                            "script_relpath": "dummy_runner.py",
                            "chunk_size": 4,
                            "strict_backend": True,
                        },
                    )


class TestTrajectorySafety(unittest.TestCase):
    def test_preserve_nonzero_when_filter_overdrops(self):
        masks = []
        for _ in range(3):
            m = np.zeros((16, 16), dtype=np.uint8)
            m[4:8, 4:8] = 255
            masks.append(m)
        scores = [0.1, 0.1, 0.1]
        filtered, meta = apply_trajectory_consistency(
            masks_u8=masks,
            motion_scores=scores,
            trajectory_cfg={"min_track_length": 10, "track_motion_threshold": 5.0},
        )
        self.assertTrue(any(int((m > 0).sum()) > 0 for m in filtered))
        self.assertTrue(bool(meta.get("fallback_preserve_original", False)))
        self.assertEqual(meta.get("fallback_reason"), "all_zero_after_filter")

    def test_preserve_when_kept_active_ratio_too_low(self):
        masks: list[np.ndarray] = []
        for idx in range(10):
            m = np.zeros((16, 16), dtype=np.uint8)
            if idx in {0, 1, 2, 7, 8, 9}:
                m[4:8, 4:8] = 255
            masks.append(m)

        scores = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0]
        filtered, meta = apply_trajectory_consistency(
            masks_u8=masks,
            motion_scores=scores,
            trajectory_cfg={
                "min_track_length": 1,
                "track_motion_threshold": 0.5,
                "motion_smooth_window": 1,
                "min_kept_active_ratio": 0.95,
            },
        )

        self.assertTrue(bool(meta.get("fallback_preserve_original", False)))
        self.assertEqual(meta.get("fallback_reason"), "kept_active_ratio_too_low")
        self.assertLess(float(meta.get("filtered_active_frame_ratio_before_fallback", 1.0)), 0.4)
        self.assertGreater(float(meta.get("filtered_active_frame_ratio", 0.0)), 0.5)
        self.assertEqual(
            sum(int((m > 0).sum()) > 0 for m in filtered),
            sum(int((m > 0).sum()) > 0 for m in masks),
        )


if __name__ == "__main__":
    unittest.main()
