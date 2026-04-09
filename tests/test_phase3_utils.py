import unittest
from pathlib import Path

import numpy as np

from src.part3.run_explore import (
    CandidateResult,
    CandidateSpec,
    apply_morph_profile,
    decide_e3_flow,
    select_best,
    temporal_smooth_masks,
)


class TestPhase3Utils(unittest.TestCase):
    def test_apply_morph_profile_binary_output(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        mask[0, 0] = 255
        out = apply_morph_profile(
            masks_u8=[mask],
            opening_kernel=3,
            closing_kernel=5,
            dilation_kernel=3,
            smoothing_kernel=3,
        )[0]
        self.assertTrue(set(np.unique(out)).issubset({0, 255}))
        self.assertGreater(int((out > 0).sum()), 0)

    def test_temporal_smooth_removes_single_frame_noise(self):
        m0 = np.zeros((32, 32), dtype=np.uint8)
        m1 = np.zeros((32, 32), dtype=np.uint8)
        m2 = np.zeros((32, 32), dtype=np.uint8)
        m1[10:14, 10:14] = 255

        out = temporal_smooth_masks([m0, m1, m2], temporal_window=1)
        self.assertEqual(len(out), 3)
        self.assertEqual(int((out[1] > 0).sum()), 0)

    def test_selection_excludes_wild(self):
        spec_a = CandidateSpec(
            stage="E1",
            name="a",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )
        spec_b = CandidateSpec(
            stage="E1",
            name="b",
            source_stage="B",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        res_a = CandidateResult(
            spec=spec_a,
            candidate_root=Path("/tmp/a"),
            eval_exp_id="exp_a",
            summary_path=Path("/tmp/a/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.95, "JR": 0.95, "Q_REMOVE": 0.90}},
                "bmx-trees": {"metrics": {"JM": 0.70, "JR": 0.75, "Q_REMOVE": 0.65}},
                "tennis": {"metrics": {"JM": 0.68, "JR": 0.74, "Q_REMOVE": 0.64}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.1},
                "bmx-trees": {"mean_mask_ratio": 0.1},
                "tennis": {"mean_mask_ratio": 0.1},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        res_b = CandidateResult(
            spec=spec_b,
            candidate_root=Path("/tmp/b"),
            eval_exp_id="exp_b",
            summary_path=Path("/tmp/b/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": 0.10, "JR": 0.12, "Q_REMOVE": 0.20}},
                "bmx-trees": {"metrics": {"JM": 0.80, "JR": 0.83, "Q_REMOVE": 0.82}},
                "tennis": {"metrics": {"JM": 0.79, "JR": 0.82, "Q_REMOVE": 0.81}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.1},
                "bmx-trees": {"mean_mask_ratio": 0.1},
                "tennis": {"mean_mask_ratio": 0.1},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_best(stage="E1", entries=[res_a, res_b], score_datasets=["bmx-trees", "tennis"])
        self.assertEqual(best.spec.name, "b")

    def test_decide_e3_flow_split(self):
        self.assertEqual(decide_e3_flow(permission_passed=False, runtime_error=None), "abort")
        self.assertEqual(decide_e3_flow(permission_passed=True, runtime_error="oom"), "skip")
        self.assertEqual(decide_e3_flow(permission_passed=True, runtime_error=None), "continue")

    def test_select_best_respects_coverage_constraints(self):
        spec_hi = CandidateSpec(
            stage="E4",
            name="high_q",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )
        spec_ok = CandidateSpec(
            stage="E4",
            name="covered",
            source_stage="E3",
            e1_profile={},
            temporal_window=0,
            use_sam3=False,
        )

        hi = CandidateResult(
            spec=spec_hi,
            candidate_root=Path("/tmp/high_q"),
            eval_exp_id="exp_high_q",
            summary_path=Path("/tmp/high_q/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.99}},
                "bmx-trees": {"metrics": {"JM": 0.62, "JR": 0.8, "Q_REMOVE": 0.91}},
                "tennis": {"metrics": {"JM": 0.80, "JR": 1.0, "Q_REMOVE": 0.97}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.0001, "active_frame_ratio": 0.01},
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
            stage_mask_meta={},
            propainter_meta={},
        )
        covered = CandidateResult(
            spec=spec_ok,
            candidate_root=Path("/tmp/covered"),
            eval_exp_id="exp_covered",
            summary_path=Path("/tmp/covered/summary.json"),
            aggregate={},
            per_dataset={
                "wild": {"metrics": {"JM": None, "JR": None, "Q_REMOVE": 0.95}},
                "bmx-trees": {"metrics": {"JM": 0.61, "JR": 0.79, "Q_REMOVE": 0.90}},
                "tennis": {"metrics": {"JM": 0.79, "JR": 0.99, "Q_REMOVE": 0.96}},
            },
            mask_stats={
                "wild": {"mean_mask_ratio": 0.01, "active_frame_ratio": 0.6},
                "bmx-trees": {"mean_mask_ratio": 0.04, "active_frame_ratio": 0.8},
                "tennis": {"mean_mask_ratio": 0.02, "active_frame_ratio": 0.7},
            },
            stage_mask_meta={},
            propainter_meta={},
        )

        best = select_best(
            stage="E4",
            entries=[hi, covered],
            score_datasets=["wild", "bmx-trees", "tennis"],
            coverage_constraints={"wild": {"min_mean_mask_ratio": 0.002, "min_active_frame_ratio": 0.25}},
            enforce_if_candidate_available=True,
        )
        self.assertEqual(best.spec.name, "covered")


if __name__ == "__main__":
    unittest.main()
