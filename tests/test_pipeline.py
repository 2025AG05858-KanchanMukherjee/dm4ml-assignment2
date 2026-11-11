"""Unit tests (plain `unittest`, runs with `python -m unittest discover -s tests` or pytest)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import metrics as M                       # noqa: E402
from src.features.store import FeatureStore                   # noqa: E402
from src.orchestration.pipeline import build_dag, topo_order  # noqa: E402
from src.transformation.split_derive import time_split        # noqa: E402
from src.utils import load_config                             # noqa: E402
from src.validation import validate as V                      # noqa: E402

CFG = load_config()
VAL = CFG["validation"]


class TestValidation(unittest.TestCase):
    def setUp(self):
        users = pd.DataFrame({"user_id": ["U1", "U2", "U2", ""], "age": ["30", "abc", "abc", "20"],
                              "gender": ["M", "F", "F", "M"], "country": ["IN", "", "", "US"],
                              "signup_date": ["2023-01-01", "2023-02-01", "2023-02-01", "2023-01-05"]})
        items = pd.DataFrame({"item_id": ["I1", "I2"], "title": ["a", "b"], "genres": ["Drama", "Comedy"],
                              "release_year": ["2000", "2099"]})
        self.tu, self.users_ok = V.validate_users(users, VAL)
        self.ti, self.items_ok = V.validate_items(items, VAL)

    def test_user_rules(self):
        rules = set(self.tu.issues_df().rule)
        self.assertTrue({"missing_user_id", "duplicate_row", "invalid_dtype_age", "missing_country"} <= rules)
        self.assertEqual(list(self.users_ok.user_id), ["U1", "U2"])      # blank id + exact dup quarantined
        self.assertTrue(self.users_ok.age.iloc[1] != self.users_ok.age.iloc[1])  # invalid age -> NaN (WARN)

    def test_interaction_rules_and_reconciliation(self):
        rows = [("U1", "I1", "5", "2023-06-01 10:00:00"),   # valid
                ("U1", "I2", "6", "2023-06-01 10:00:00"),   # rating out of range
                ("U1", "I2", "x", "2023-06-01 10:00:00"),   # non-numeric
                ("U1", "I1", "4", "2023-07-01 10:00:00"),   # same (user,item) later -> keeps latest, drops earlier
                ("U9", "I1", "4", "2023-06-01 10:00:00"),   # orphan user
                ("U1", "I9", "4", "2023-06-01 10:00:00"),   # orphan item
                ("U1", "I2", "4", "2030-01-01 00:00:00"),   # future
                ("U1", "I2", "4", "not-a-date"),            # unparsable
                ("U1", "I2", "4", "2022-01-01 00:00:00")]   # before signup
        df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
        t, ok = V.validate_interactions(df, VAL, self.users_ok, self.items_ok)
        rules = set(t.issues_df().rule)
        for r in ["rating_out_of_range", "invalid_dtype_rating", "fk_user_not_found", "fk_item_not_found",
                  "timestamp_in_future", "invalid_timestamp", "timestamp_before_signup", "duplicate_user_item"]:
            self.assertIn(r, rules)
        self.assertEqual(len(ok), 1)
        self.assertEqual(int(ok.rating.iloc[0]), 4)                      # the later interaction survived
        self.assertEqual(len(df), len(ok) + int(t.fatal.sum()))          # rows in = valid + quarantined


class TestSplitAndFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 400
        self.inter = pd.DataFrame({"user_id": rng.choice([f"U{i}" for i in range(20)], n),
                                   "item_id": rng.choice([f"I{i}" for i in range(15)], n),
                                   "rating": rng.integers(1, 6, n),
                                   "timestamp": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.uniform(0, 300, n), unit="D")})
        self.inter = self.inter.drop_duplicates(["user_id", "item_id"])
        self.users = pd.DataFrame({"user_id": [f"U{i}" for i in range(22)], "age": 30,
                                   "signup_date": pd.Timestamp("2023-12-01")})
        self.items = pd.DataFrame({"item_id": [f"I{i}" for i in range(15)], "release_year": 2010,
                                   "genres": ["Drama|Comedy" if i % 2 else "Action" for i in range(15)]})

    def test_time_split_has_no_leakage(self):
        tr, te, cut = time_split(self.inter, 0.2)
        self.assertLess(tr.timestamp.max(), te.timestamp.min())
        self.assertEqual(len(tr) + len(te), len(self.inter))

    def test_features_are_point_in_time_and_cold_start_safe(self):
        tr, te, _ = time_split(self.inter, 0.2)
        fs = FeatureStore.build(tr, self.users, self.items, CFG)
        self.assertEqual(fs.as_of, tr.timestamp.max())                   # nothing from the test period
        u = tr.user_id.iloc[0]
        self.assertAlmostEqual(fs.get_user_features([u]).user_avg_rating.iloc[0], tr[tr.user_id == u].rating.mean())
        cold = fs.get_user_features(["U_NEW"])
        self.assertEqual(cold.user_activity_count.iloc[0], 0)
        self.assertEqual(fs.affinity_scores(["U_NEW"]).shape, (1, 15))
        np.testing.assert_allclose(fs.affinity.sum(axis=1).values, 1.0)  # affinity rows sum to 1


class TestMetrics(unittest.TestCase):
    def test_precision_recall_ndcg_by_hand(self):
        recs = np.array([[0, 1, 2, 3, 4]])
        out = M.precision_recall_ndcg(recs, [{0, 2, 9}], k=5)
        self.assertAlmostEqual(out["precision@5"], 2 / 5)
        self.assertAlmostEqual(out["recall@5"], 2 / 3)
        dcg = 1 / np.log2(2) + 1 / np.log2(4)
        idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
        self.assertAlmostEqual(out["ndcg@5"], dcg / idcg)

    def test_coverage_diversity_errors_and_topk(self):
        recs = np.array([[0, 1], [1, 2]])
        self.assertAlmostEqual(M.coverage(recs, 6), 3 / 6)
        V_ = np.array([[1, 0], [1, 0], [0, 1]], float)                    # items 0,1 identical; 2 different
        self.assertAlmostEqual(M.intra_list_diversity(np.array([[0, 1]]), V_), 0.0)
        self.assertAlmostEqual(M.intra_list_diversity(np.array([[0, 2]]), V_), 1.0)
        e = M.rating_errors([3, 4], [4, 4])
        self.assertAlmostEqual(e["rmse"], np.sqrt(0.5)); self.assertAlmostEqual(e["mae"], 0.5)
        s = np.array([[0.1, 0.9, 0.5, 0.7]]); seen = np.array([[False, True, False, False]])
        self.assertEqual(list(M.top_k_indices(s, seen, 2)[0]), [3, 2])    # item 1 is masked as seen


class TestDag(unittest.TestCase):
    def test_dependencies_are_respected_and_acyclic(self):
        order = [t.name for t in topo_order(build_dag())]
        for t in build_dag():
            for d in t.deps:
                self.assertLess(order.index(d), order.index(t.name))
        self.assertEqual(order[0], "ingest_data"); self.assertEqual(order[-1], "evaluate_models")


if __name__ == "__main__":
    unittest.main()
