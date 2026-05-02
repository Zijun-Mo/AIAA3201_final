import logging
import random
import unittest
from pathlib import Path

import numpy as np

from src.part1.run_baseline import (
    CandidateResult,
    CandidateSpec,
    build_wild_fallback_masks,
    classify_failure_case,
    compute_active_frame_ratio,
    select_best,
    set_global_seed,
)


class TestPhase1Utils(unittest.TestCase):
    def test_seed_reproducibility(self):
        logger = logging.getLogger("test_seed")
        set_global_seed(123, logger)
        a_np = np.random.rand(8)
        a_py = random.random()

        set_global_seed(123, logger)
        b_np = np.random.rand(8)
        b_py = random.random()

        self.assertTrue(np.allclose(a_np, b_np))
        self.assertEqual(a_py, b_py)

    def test_wild_fallback_masks_non_zero_on_synthetic_motion(self):
        h, w = 64, 64
        frames = []
        for i in range(6):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            x0 = 8 + i * 4
            frame[20:36, x0 : x0 + 10] = 255
            frames.append(frame)

        masks, stats = build_wild_fallback_masks(
            frames=frames,
            diff_threshold_percentile=95.0,
            morph_kernel=3,
        )
        self.assertEqual(len(masks), len(frames))
        self.assertGreater(float(stats["fallback_mean_mask_ratio"]), 0.0)
        self.assertTrue(any(int((m > 0).sum()) > 0 for m in masks))

    def test_failure_classification_rules(self):
        self.assertEqual(
            classify_failure_case(
                dataset="wild",
                ros=0.2,
                tcf=0.1,
                bes=0.1,
                wild_fallback_applied=True,
            ),
            "residual_object_fallback_recovery",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="bmx-trees",
                ros=0.01,
                tcf=0.05,
                bes=0.2,
                wild_fallback_applied=False,
            ),
            "boundary_artifact",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                ros=0.08,
                tcf=0.01,
                bes=0.05,
                wild_fallback_applied=False,
            ),
            "residual_object",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                ros=0.01,
                tcf=0.2,
                bes=0.01,
                wild_fallback_applied=False,
            ),
            "temporal_flicker",
        )

    def test_compute_active_frame_ratio(self):
        m0 = np.zeros((8, 8), dtype=np.uint8)
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m1[0:2, 0:2] = 255
        ratio = compute_active_frame_ratio([m0, m1, m2])
        self.assertAlmostEqual(ratio, 1.0 / 3.0, places=6)

    def test_select_best_respects_coverage_constraints(self):
        spec_hi = CandidateSpec("A4", "high_q", "yolo", 1.2, 7, "telea", 1)
        spec_ok = CandidateSpec("A4", "covered", "yolo", 1.2, 7, "telea", 1)

        high_q = CandidateResult(
            spec=spec_hi,
            candidate_root=Path("/tmp/high_q"),
            eval_exp_id="exp_high_q",
            summary_path=Path("/tmp/high_q/summary.json"),
            aggregate={"JM": 0.7, "JR": 0.7, "TCF": 0.95},
            per_dataset={},
            mask_stats={
                "wild": {"mean_mask_ratio": 0.0001, "active_frame_ratio": 0.01},
                "bmx-trees": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
        )
        covered = CandidateResult(
            spec=spec_ok,
            candidate_root=Path("/tmp/covered"),
            eval_exp_id="exp_covered",
            summary_path=Path("/tmp/covered/summary.json"),
            aggregate={"JM": 0.68, "JR": 0.69, "TCF": 0.92},
            per_dataset={},
            mask_stats={
                "wild": {"mean_mask_ratio": 0.01, "active_frame_ratio": 0.6},
                "bmx-trees": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
        )
        best = select_best(
            stage="A4",
            entries=[high_q, covered],
            coverage_constraints={"wild": {"min_mean_mask_ratio": 0.002, "min_active_frame_ratio": 0.25}},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "covered")


if __name__ == "__main__":
    unittest.main()
