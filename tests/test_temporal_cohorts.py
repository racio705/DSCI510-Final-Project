"""Check that temporal novelty and selection cannot use outcome eligibility."""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.temporal_cohorts import (
    read_raw_identity_sets, select_host_uniform, strict_novel_mask,
)


class TemporalCohortTests(unittest.TestCase):
    def test_prior_sets_include_ineligible_and_missing_price_records(self):
        rows = pd.DataFrame({"id": ["0001", "2", "3", None],
                             "host_id": ["h1", "h2", None, "h3"],
                             "price": [None, 0, -10, None],
                             "room_type": ["Hotel room"] * 4})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv.gz"
            rows.to_csv(path, index=False)
            listings, hosts = read_raw_identity_sets(path)
        self.assertEqual(listings, {"0001", "2", "3"})
        self.assertEqual(hosts, {"h1", "h2", "h3"})

    def test_novelty_requires_both_identities_and_rejects_unknown_hosts(self):
        frame = pd.DataFrame({"id": ["old-listing", "new-a", "new-b", "new-c", None]})
        hosts = pd.Series(["new-host-a", "old-host", "new-host-b", None, "new-host-c"])
        actual = strict_novel_mask(frame, hosts, {"old-listing"}, {"old-host"})
        np.testing.assert_array_equal(actual, [False, False, True, False, False])
        with self.assertRaises(ValueError):
            strict_novel_mask(frame, hosts.iloc[::-1], set(), set())

    def test_host_selection_is_hash_based_price_blind_and_order_stable(self):
        frame = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
                              "price": [1, 10000, 30, 40, 50]}, index=[9, 2, 8, 1, 4])
        hosts = pd.Series(["h1", "h1", "h2", "h2", "h3"], index=frame.index)
        selected = select_host_uniform(frame, hosts)
        expected = {
            min(group, key=lambda value: hashlib.sha256(("temporal-v1|" + value).encode()).hexdigest())
            for group in [["a", "b"], ["c", "d"], ["e"]]
        }
        self.assertEqual(set(frame.iloc[selected]["id"]), expected)
        self.assertTrue(np.all(selected[:-1] < selected[1:]))
        reversed_frame, reversed_hosts = frame.iloc[::-1].copy(), hosts.iloc[::-1]
        reversed_frame["price"] = [50000, 1, 1, 1, 1]
        selected_reversed = select_host_uniform(reversed_frame, reversed_hosts)
        self.assertEqual(set(reversed_frame.iloc[selected_reversed]["id"]), expected)
        with self.assertRaises(ValueError):
            select_host_uniform(frame, hosts.mask(hosts.eq("h1")))


if __name__ == "__main__":
    unittest.main()
