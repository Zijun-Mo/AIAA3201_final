import json
import math
import tempfile
import unittest
from pathlib import Path

from src.part3.run_diffusion import metric_or_pos_inf, resolve_comparison_reference


class TestPhase5Utils(unittest.TestCase):
    def test_metric_or_pos_inf_handles_nan(self):
        self.assertTrue(math.isinf(metric_or_pos_inf({}, "TCF")))
        self.assertTrue(math.isinf(metric_or_pos_inf({"TCF": None}, "TCF")))
        self.assertTrue(math.isinf(metric_or_pos_inf({"TCF": float("nan")}, "TCF")))
        self.assertAlmostEqual(metric_or_pos_inf({"TCF": 0.95}, "TCF"), 0.95, places=6)

    def test_resolve_comparison_reference_prefers_phase2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            phase2_dir = root / "phase2_test"
            phase2_dir.mkdir(parents=True, exist_ok=True)
            with (phase2_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump({"aggregate": {"TCF": 0.91}}, f)

            label, exp_id, agg = resolve_comparison_reference(
                metrics_root=root,
                phase2_exp_id="phase2_test",
            )
            self.assertEqual(label, "B-best")
            self.assertEqual(exp_id, "phase2_test")
            self.assertAlmostEqual(float(agg["TCF"]), 0.91, places=6)

    def test_resolve_comparison_reference_raises_when_phase2_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(RuntimeError):
                resolve_comparison_reference(
                    metrics_root=root,
                    phase2_exp_id="phase2_missing",
                )


if __name__ == "__main__":
    unittest.main()
