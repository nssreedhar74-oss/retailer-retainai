import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import storage
from src.data import (FEATURE_COLUMNS, build_training_set, current_snapshot, generate_demo_data,
                      is_churn_event, make_features, validate_data)
from src.engine import (add_observed_purchase_after_action, budget_plan, build_decision_table,
                        build_winback_table, recommend_action, risk_band)
from src.models import compute_metrics, train_and_score


class ChurnLabelTests(unittest.TestCase):
    def test_reaches_inactivity_threshold_within_horizon(self):
        dates = pd.Series(pd.to_datetime(["2026-01-01"]))
        result = is_churn_event("R1", pd.Timestamp("2026-01-31"), 30, 60,
                                pd.DataFrame(), {"R1": dates})
        self.assertEqual(result, 1)

    def test_purchase_before_threshold_prevents_label(self):
        dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-03-01"]))
        result = is_churn_event("R1", pd.Timestamp("2026-01-31"), 30, 60,
                                pd.DataFrame(), {"R1": dates})
        self.assertEqual(result, 0)

    def test_threshold_outside_horizon_is_negative(self):
        dates = pd.Series(pd.to_datetime(["2026-01-20"]))
        result = is_churn_event("R1", pd.Timestamp("2026-01-31"), 30, 60,
                                pd.DataFrame(), {"R1": dates})
        self.assertEqual(result, 0)


class LabelEdgeTests(unittest.TestCase):
    def test_purchase_exactly_on_threshold_date_prevents_churn(self):
        dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-03-02"]))  # threshold date = 2026-03-02
        self.assertEqual(is_churn_event("R1", pd.Timestamp("2026-01-31"), 30, 60, pd.DataFrame(), {"R1": dates}), 0)

    def test_purchase_after_threshold_date_still_counts_as_churn(self):
        dates = pd.Series(pd.to_datetime(["2026-01-01", "2026-03-03"]))
        self.assertEqual(is_churn_event("R1", pd.Timestamp("2026-01-31"), 30, 60, pd.DataFrame(), {"R1": dates}), 1)


class RetentionEconomicsTests(unittest.TestCase):
    def test_expected_net_benefit_formula(self):
        row = pd.DataFrame([{
            "retailer_id": "R1", "retailer_name": "Example Store", "territory": "South",
            "churn_probability": 0.60, "monthly_contribution_90d": 10000.0,
            "complaints_90d": 2, "stockouts_90d": 0, "sales_change_pct": 0.0,
            "avg_payment_delay_90d": 0.0, "visits_30d": 1, "days_since_last_purchase": 40,
            "orders_90d": 3,
        }])
        result = build_decision_table(row, future_months=3).iloc[0]
        # Complaint visit: 60% churn x 35% conditional uplift x ₹30,000 less ₹600 cost.
        self.assertAlmostEqual(result.expected_net_benefit, 5700.0, places=2)

    def test_budget_plan_does_not_exceed_budget(self):
        rows = pd.DataFrame([
            {"retailer_id":"R1", "intervention_cost":600, "expected_net_benefit":1200},
            {"retailer_id":"R2", "intervention_cost":500, "expected_net_benefit":900},
        ])
        selected, spent, _ = budget_plan(rows, 1000)
        self.assertLessEqual(spent, 1000)
        self.assertEqual(len(selected), 1)


    def test_uplift_changes_benefit_but_not_probability(self):
        row = pd.DataFrame([{"retailer_id": "R1", "retailer_name": "X", "territory": "South", "churn_probability": 0.5,
                             "monthly_contribution_90d": 10000.0, "complaints_90d": 2, "stockouts_90d": 0, "sales_change_pct": 0.0,
                             "avg_payment_delay_90d": 0.0, "visits_30d": 1, "days_since_last_purchase": 40, "orders_90d": 3}])
        lo = build_decision_table(row, uplift_overrides={"Complaint resolution visit": 0.1}).iloc[0]
        hi = build_decision_table(row, uplift_overrides={"Complaint resolution visit": 0.5}).iloc[0]
        self.assertEqual(lo.churn_probability, hi.churn_probability)
        self.assertLess(lo.expected_net_benefit, hi.expected_net_benefit)

    def test_budget_plan_skips_negative_benefit_and_ranks_by_value_per_cost(self):
        rows = pd.DataFrame([
            {"retailer_id": "A", "intervention_cost": 500, "expected_net_benefit": 600},   # ratio 1.2
            {"retailer_id": "B", "intervention_cost": 100, "expected_net_benefit": 300},   # ratio 3.0
            {"retailer_id": "C", "intervention_cost": 100, "expected_net_benefit": -50},   # negative: never selected
        ])
        selected, spent, benefit = budget_plan(rows, 550)
        self.assertEqual(set(selected.retailer_id), {"B"})  # A no longer fits after B (100 + 500 > 550)
        self.assertEqual(spent, 100.0)
        self.assertEqual(benefit, 300.0)


class RecommendationTests(unittest.TestCase):
    base = {"churn_probability": 0.4, "complaints_90d": 0, "stockouts_90d": 0, "sales_change_pct": 0.0,
            "avg_payment_delay_90d": 0.0, "visits_30d": 1}

    def act(self, **kw):
        return recommend_action({**self.base, **kw})[0]

    def test_each_action_is_triggered_by_its_evidence(self):
        self.assertEqual(self.act(complaints_90d=2), "Complaint resolution visit")
        self.assertEqual(self.act(stockouts_90d=2), "Stock replenishment review")
        self.assertEqual(self.act(avg_payment_delay_90d=20), "Credit and payment review")
        self.assertEqual(self.act(visits_30d=0), "Personal salesperson visit")
        self.assertEqual(self.act(), "Call / WhatsApp reactivation")
        self.assertEqual(self.act(churn_probability=0.02), "Routine relationship check")

    def test_sales_drop_alone_does_not_claim_a_stock_problem(self):
        self.assertNotEqual(self.act(sales_change_pct=-0.5), "Stock replenishment review")

    def test_risk_band_thresholds(self):
        self.assertEqual(risk_band(0.5), "High")
        self.assertEqual(risk_band(0.15), "Medium")
        self.assertEqual(risk_band(0.02), "Low")
        self.assertEqual(risk_band(0.5, high=0.6, medium=0.4), "Medium")


class ValidationTests(unittest.TestCase):
    def frames(self):
        p = pd.DataFrame({"retailer_id": ["R1", "R2", "R2"], "territory": ["N", "S", "S"]})
        t = pd.DataFrame({"Retailer ID": ["R1", "R1", "R2", "R2"], "order_id": [None, None, "O1", "O1"],
                          "order_date": ["2026-01-01", "2026-01-05", "2026-01-02", "2026-01-02"],
                          "order_value": [100, 200, 300, 300], "gross_contribution": [10, 20, 30, 30]})
        e = pd.DataFrame({"retailer_id": ["R1", "ZZ"], "event_date": ["2026-01-03", "2026-01-03"], "event_type": ["Complaint", "visit"]})
        return p, t, e

    def test_cleaning_and_report(self):
        p, t, e, rep = validate_data(*self.frames(), return_report=True)
        self.assertEqual(len(p), 2)
        self.assertEqual(len(t), 3)  # blank order_ids are NOT collapsed; the duplicated O1 is
        self.assertEqual(rep["duplicate_transactions_removed"], 1)
        self.assertEqual(rep["duplicate_retailers_removed"], 1)
        self.assertEqual(rep["events_for_unknown_retailers_removed"], 1)
        self.assertEqual(e.event_type.iloc[0], "complaint")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(t.order_date))

    def test_invalid_date_raises_clear_error(self):
        p, t, e = self.frames()
        t.loc[0, "order_date"] = "not a date"
        with self.assertRaisesRegex(ValueError, "invalid 'order_date'"):
            validate_data(p, t, e)

    def test_missing_column_and_negative_values(self):
        p, t, e = self.frames()
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_data(p, t.drop(columns=["gross_contribution"]), e)
        t.loc[1, "order_value"] = -5
        _, t2, _, rep = validate_data(p, t, e, return_report=True)
        self.assertEqual(rep["negative_value_transactions_removed"], 1)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.p, cls.t, cls.e = validate_data(*generate_demo_data(200, 42))
        cls.training, cls.cutoffs = build_training_set(cls.p, cls.t, cls.e)
        cls.asof = cls.t.order_date.max()
        cls.current = current_snapshot(cls.p, cls.t, cls.e, cls.asof)

    def test_multiple_cutoffs_and_full_label_window(self):
        self.assertGreaterEqual(len(self.cutoffs), 3)
        self.assertLessEqual(max(self.cutoffs) + pd.Timedelta(days=30), self.asof)
        self.assertTrue(set(self.training.churned_next_30d.unique()) <= {0, 1})

    def test_only_active_retailers_in_training_and_current(self):
        self.assertTrue((self.training.days_since_last_purchase < 60).all())
        self.assertTrue((self.current.days_since_last_purchase < 60).all())

    def test_features_ignore_data_after_cutoff(self):
        cutoff = self.cutoffs[2]
        rid = self.training[self.training.as_of_date == cutoff].retailer_id.iloc[0]
        before = make_features(rid, cutoff, self.p, self.t, self.e)
        # Remove everything after the cutoff: features must not change.
        t_cut = self.t[self.t.order_date <= cutoff]
        e_cut = self.e[self.e.event_date <= cutoff]
        after = make_features(rid, cutoff, self.p, t_cut, e_cut)
        for c in FEATURE_COLUMNS:
            self.assertEqual(before[c], after[c], c)
        # Adding a huge future order must not change them either.
        future = pd.DataFrame([{"retailer_id": rid, "order_date": cutoff + pd.Timedelta(days=5), "order_value": 1e6,
                                "gross_contribution": 1e5, "order_id": "FUT"}])
        polluted = make_features(rid, cutoff, self.p, pd.concat([self.t, future]), self.e)
        self.assertEqual(before["sales_30d"], polluted["sales_30d"])

    def test_time_based_split_and_honest_outputs(self):
        out = train_and_score(self.training, self.current)
        self.assertLess(out["test_start"], out["test_end"] + pd.Timedelta(days=1))
        train_dates = self.training[self.training.as_of_date < out["test_start"]].as_of_date
        self.assertTrue((train_dates < out["test_start"]).all())
        self.assertEqual(set(out["evaluations"]), {"Rule-based baseline", "Logistic Regression", "Random Forest"})
        probs = out["scores"].churn_probability
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())
        for m in out["evaluations"].values():
            for k in ("precision", "recall", "f1", "pr_auc"):
                self.assertTrue(0 <= m[k] <= 1)
        self.assertEqual(sum(sum(r) for r in out["evaluations"]["Logistic Regression"]["confusion_matrix"]), out["n_test"])

    def test_winback_is_disjoint_from_scored_retailers(self):
        wb = build_winback_table(self.p, self.t, self.e, self.asof, 60)
        self.assertTrue((wb.days_since_last_purchase >= 60).all())
        self.assertFalse(set(wb.retailer_id) & set(self.current.retailer_id))

    def test_metrics_function(self):
        m = compute_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), 0.5)
        self.assertEqual((m["precision"], m["recall"]), (1.0, 1.0))
        self.assertEqual(m["confusion_matrix"], [[2, 0], [0, 2]])


class TrackerTests(unittest.TestCase):
    def test_round_trip_and_observed_purchase(self):
        with tempfile.TemporaryDirectory() as d:
            original = storage.DB_PATH
            storage.DB_PATH = Path(d) / "t.db"
            try:
                storage.add_intervention("R1", "Routine relationship check", "Asha", "2026-01-10", "Planned", 50, "Pending", "note")
                rec = storage.get_interventions()
                self.assertEqual(len(rec), 1)
                storage.update_intervention(int(rec.id.iloc[0]), "Completed", 75, "Purchased again", "done")
                rec = storage.get_interventions()
                self.assertEqual((rec.status.iloc[0], rec.actual_cost.iloc[0], rec.outcome.iloc[0]), ("Completed", 75.0, "Purchased again"))
                tx = pd.DataFrame({"retailer_id": ["R1", "R1"], "order_date": pd.to_datetime(["2026-01-05", "2026-01-20"])})
                out = add_observed_purchase_after_action(rec, tx)
                self.assertEqual(out.first_purchase_after_action.iloc[0], "2026-01-20")
            finally:
                storage.DB_PATH = original


if __name__ == "__main__":
    unittest.main()
