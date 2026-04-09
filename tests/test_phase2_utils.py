import unittest
from pathlib import Path

import numpy as np

from src.part2.run_sota import (
    CandidateResult,
    CandidateSpec,
    build_auto_prompts,
    classify_failure_case,
    compute_active_frame_ratio,
    compute_mean_mask_ratio,
    refine_masks,
    select_best,
)


class TestPhase2Utils(unittest.TestCase):
    def test_compute_mean_mask_ratio(self):
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:5, 0:5] = 255
        m2[0:2, 0:2] = 255
        ratio = compute_mean_mask_ratio([m1, m2])
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)

    def test_compute_active_frame_ratio(self):
        m0 = np.zeros((8, 8), dtype=np.uint8)
        m1 = np.zeros((8, 8), dtype=np.uint8)
        m2 = np.zeros((8, 8), dtype=np.uint8)
        m2[1:3, 1:3] = 255
        ratio = compute_active_frame_ratio([m0, m1, m2])
        self.assertAlmostEqual(ratio, 1.0 / 3.0, places=6)

    def test_build_auto_prompts_from_instances(self):
        h, w = 64, 64
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(4)]
        instances = [[], [], [], []]

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:40, 10:30] = 1
        instances[2] = [{"mask": mask}]

        prompts = build_auto_prompts(instances_per_frame=instances, frames=frames, max_prompts=3)
        self.assertEqual(prompts["frame_idx"], 2)
        self.assertTrue(len(prompts["boxes"]) >= 1)
        x1, y1, x2, y2 = prompts["boxes"][0]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_refine_masks_temporal(self):
        masks = []
        for i in range(5):
            m = np.zeros((32, 32), dtype=np.uint8)
            m[10:15, 10 + i : 14 + i] = 255
            masks.append(m)

        refined = refine_masks(masks_u8=masks, morph_kernel=3, temporal_window=1)
        self.assertEqual(len(refined), len(masks))
        self.assertTrue(any(int((m > 0).sum()) > 0 for m in refined))

    def test_failure_classification_priority(self):
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                ros=0.01,
                tcf=0.01,
                bes=0.01,
                backend_fallback=True,
                propainter_fallback=False,
            ),
            "mask_backend_fallback",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                ros=0.01,
                tcf=0.01,
                bes=0.01,
                backend_fallback=False,
                propainter_fallback=True,
            ),
            "propainter_profile_fallback",
        )

    def test_select_best_respects_coverage_constraints(self):
        spec_hi = CandidateSpec(
            stage="B2",
            name="high_q",
            mask_backend="sam2",
            mask_variant="coarse",
            neighbor_length=10,
            ref_stride=10,
            subvideo_length=40,
            resize_ratio=0.75,
            mask_dilation=4,
            fp16=True,
        )
        spec_ok = CandidateSpec(
            stage="B2",
            name="covered",
            mask_backend="sam2",
            mask_variant="coarse",
            neighbor_length=10,
            ref_stride=10,
            subvideo_length=40,
            resize_ratio=0.75,
            mask_dilation=4,
            fp16=True,
        )

        hi = CandidateResult(
            spec=spec_hi,
            candidate_root=Path("/tmp/high_q"),
            eval_exp_id="exp_high_q",
            summary_path=Path("/tmp/high_q/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.99}},
                "bmx-trees": {"metrics": {"JM": 0.6, "JR": 0.7, "Q_REMOVE": 0.9}},
                "tennis": {"metrics": {"JM": 0.6, "JR": 0.7, "Q_REMOVE": 0.9}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.0001, "active_frame_ratio": 0.01},
                "bmx-trees": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
            },
            backend_meta={},
            propainter_meta={},
        )
        ok = CandidateResult(
            spec=spec_ok,
            candidate_root=Path("/tmp/covered"),
            eval_exp_id="exp_covered",
            summary_path=Path("/tmp/covered/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.95}},
                "bmx-trees": {"metrics": {"JM": 0.58, "JR": 0.68, "Q_REMOVE": 0.88}},
                "tennis": {"metrics": {"JM": 0.58, "JR": 0.68, "Q_REMOVE": 0.88}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.01, "active_frame_ratio": 0.6},
                "bmx-trees": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.03, "active_frame_ratio": 0.8},
            },
            backend_meta={},
            propainter_meta={},
        )

        best = select_best(
            stage="B2",
            entries=[hi, ok],
            score_datasets=["wild", "bmx-trees", "tennis"],
            coverage_constraints={"wild": {"min_mean_mask_ratio": 0.002, "min_active_frame_ratio": 0.25}},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "covered")


if __name__ == "__main__":
    unittest.main()
