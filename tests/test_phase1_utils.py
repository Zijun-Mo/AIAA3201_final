import logging
import random
import unittest

import numpy as np

from src.part1.run_baseline import (
    build_wild_fallback_masks,
    classify_failure_case,
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
                psnr=10.0,
                edge_diff=0.9,
                texture_ratio=0.1,
                wild_fallback_applied=True,
            ),
            "missed_detection_then_fallback",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="bmx-trees",
                psnr=20.0,
                edge_diff=0.2,
                texture_ratio=0.9,
                wild_fallback_applied=False,
            ),
            "boundary_residue",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                psnr=20.0,
                edge_diff=0.01,
                texture_ratio=0.5,
                wild_fallback_applied=False,
            ),
            "texture_loss",
        )
        self.assertEqual(
            classify_failure_case(
                dataset="tennis",
                psnr=30.0,
                edge_diff=0.01,
                texture_ratio=1.2,
                wild_fallback_applied=False,
            ),
            "temporal_flicker_suspected",
        )


if __name__ == "__main__":
    unittest.main()

