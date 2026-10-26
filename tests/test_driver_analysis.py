"""Check joint permutations and pooled error arithmetic."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.driver_analysis import jointly_permute, pool_repeats


class DriverAnalysisTests(unittest.TestCase):
    def test_joint_permutation_preserves_group_relationships_and_other_columns(self):
        original = pd.DataFrame({"numeric": np.arange(20), "category": [f"P{i}" for i in range(20)],
                                 "other": np.arange(20) * 3}, index=np.arange(20) + 100)
        before = original.copy(deep=True)
        shuffled = jointly_permute(original, ["numeric", "category"], seed=42)
        self.assertEqual(set(zip(shuffled.numeric, shuffled.category)), set(zip(original.numeric, original.category)))
        self.assertFalse(shuffled.numeric.equals(original.numeric))
        pd.testing.assert_series_equal(shuffled.other, original.other)
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(shuffled, jointly_permute(original, ["numeric", "category"], seed=42))

    def test_pool_rmse_uses_weighted_squared_errors(self):
        detail = pd.DataFrame({
            "group": ["A", "A"], "repeat": [1, 1], "n_test": [2, 3],
            "baseline_mae_usd": [10., 20.], "permuted_mae_usd": [20., 30.],
            "baseline_log_rmse": [1., 3.], "permuted_log_rmse": [2., 4.],
        })
        pooled = pool_repeats(detail).iloc[0]
        self.assertEqual(pooled["n"], 5)
        self.assertAlmostEqual(pooled["baseline_mae_usd"], 16)
        self.assertAlmostEqual(pooled["delta_mae_usd"], 10)
        self.assertAlmostEqual(pooled["baseline_log_rmse"], np.sqrt(29 / 5))
        self.assertAlmostEqual(pooled["delta_log_rmse"], np.sqrt(56 / 5) - np.sqrt(29 / 5))


if __name__ == "__main__":
    unittest.main()
