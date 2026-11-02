from __future__ import annotations

import numpy as np
import pandas as pd

ACTION_ASSUMPTIONS = {
    "Complaint resolution visit": {"cost": 600.0, "uplift": 0.35, "description": "Resolve open service issues and agree a service recovery plan."},
    "Stock replenishment review": {"cost": 450.0, "uplift": 0.30, "description": "Check availability, correct stock-outs and recommend a practical replenishment order."},
    "Credit and payment review": {"cost": 200.0, "uplift": 0.22, "description": "Discuss overdue balances, reconcile disputes and agree a payment plan."},
    "Personal salesperson visit": {"cost": 500.0, "uplift": 0.28, "description": "Conduct a relationship visit, understand objections and review the retailer's assortment."},
    "Call / WhatsApp reactivation": {"cost": 75.0, "uplift": 0.18, "description": "Contact the retailer, understand the purchase gap and share a relevant offer or reorder reminder."},
    "Routine relationship check": {"cost": 50.0, "uplift": 0.10, "description": "Maintain contact and monitor the next expected order."},
}


# Operating thresholds for risk bands. Because the 30-day churn base rate is usually low
# (single-digit to low-double-digit %), bands are defined on the calibrated probability and
# can be changed in the sidebar. They are management choices, not statistical standards.
DEFAULT_HIGH_THRESHOLD = 0.30
DEFAULT_MEDIUM_THRESHOLD = 0.10


def risk_band(p, high=DEFAULT_HIGH_THRESHOLD, medium=DEFAULT_MEDIUM_THRESHOLD):
    if p >= high:
        return "High"
    if p >= medium:
        return "Medium"
    return "Low"


def recommend_action(row, medium=DEFAULT_MEDIUM_THRESHOLD):
    """Return (action, reason). Each action is triggered by observable evidence in the row.

    One action per retailer: rules are checked in this fixed order, so the first matching
    evidence wins. This is a simple, explainable heuristic, not an optimised treatment policy.
    """
    p = row.get("churn_probability", 0)
    if row.get("complaints_90d", 0) >= 2:
        return "Complaint resolution visit", f"{int(row['complaints_90d'])} complaints in the last 90 days"
    if row.get("stockouts_90d", 0) >= 2:
        return "Stock replenishment review", f"{int(row['stockouts_90d'])} stock-outs in the last 90 days"
    if row.get("avg_payment_delay_90d", 0) >= 15:
        return "Credit and payment review", f"Average payment delay of {row['avg_payment_delay_90d']:.0f} days"
    if p >= medium and (row.get("visits_30d", 0) == 0 or row.get("sales_change_pct", 0) <= -0.25):
        why = "No visit in the last 30 days" if row.get("visits_30d", 0) == 0 else f"Sales down {abs(row['sales_change_pct']):.0%} versus previous 30 days"
        return "Personal salesperson visit", why
    if p >= medium:
        return "Call / WhatsApp reactivation", "Elevated churn probability without a specific service or payment issue"
    return "Routine relationship check", "Low predicted risk and no strong adverse signal"


def risk_signals(row):
    signals = []
    if row.get("days_since_last_purchase", 0) >= 30:
        signals.append(f"No purchase for {int(row['days_since_last_purchase'])} days")
    if row.get("sales_change_pct", 0) <= -0.20:
        signals.append(f"Sales down {abs(row['sales_change_pct']):.0%} versus previous 30 days")
    if row.get("orders_90d", 0) <= 1:
        signals.append("Very low order frequency in the last 90 days")
    if row.get("complaints_90d", 0) >= 1:
        signals.append(f"{int(row['complaints_90d'])} complaint(s) in the last 90 days")
    if row.get("stockouts_90d", 0) >= 1:
        signals.append(f"{int(row['stockouts_90d'])} stock-out incident(s) in the last 90 days")
    if row.get("avg_payment_delay_90d", 0) >= 10:
        signals.append(f"Average payment delay of {row['avg_payment_delay_90d']:.0f} days")
    if row.get("visits_30d", 0) == 0:
        signals.append("No salesperson visit recorded in the last 30 days")
    return signals or ["No major adverse signal in the measured features; monitor purchase cadence."]


def build_decision_table(scores: pd.DataFrame, future_months=3, cost_overrides=None, uplift_overrides=None,
                         high=DEFAULT_HIGH_THRESHOLD, medium=DEFAULT_MEDIUM_THRESHOLD):
    if scores.empty:
        return scores.copy()
    out = scores.copy()
    out["risk_segment"] = out.churn_probability.map(lambda p: risk_band(p, high, medium))
    picked = out.apply(lambda r: recommend_action(r, medium), axis=1)
    out["recommended_action"] = picked.map(lambda x: x[0])
    out["action_reason"] = picked.map(lambda x: x[1])
    cost_overrides = cost_overrides or {}
    uplift_overrides = uplift_overrides or {}
    out["intervention_cost"] = out.recommended_action.map(lambda a: float(cost_overrides.get(a, ACTION_ASSUMPTIONS[a]["cost"])))
    out["conditional_uplift_assumption"] = out.recommended_action.map(lambda a: float(uplift_overrides.get(a, ACTION_ASSUMPTIONS[a]["uplift"])))
    out["future_contribution"] = out.monthly_contribution_90d.clip(lower=0) * float(future_months)
    # Conditional uplift is an assumption about effectiveness among at-risk retailers.
    # The incremental probability of saving = predicted churn probability x conditional uplift.
    out["incremental_save_probability"] = out.churn_probability * out.conditional_uplift_assumption
    out["expected_gross_benefit"] = out.incremental_save_probability * out.future_contribution
    out["expected_net_benefit"] = out.expected_gross_benefit - out.intervention_cost
    out["benefit_cost_ratio"] = np.where(out.intervention_cost > 0, out.expected_gross_benefit / out.intervention_cost, np.nan)
    out["risk_signals"] = out.apply(lambda r: " • ".join(risk_signals(r)), axis=1)
    out["priority_rank"] = out.expected_net_benefit.rank(method="first", ascending=False).astype(int)
    return out.sort_values(["expected_net_benefit", "churn_probability"], ascending=False).reset_index(drop=True)


def budget_plan(decisions: pd.DataFrame, budget: float):
    """Greedy, transparent budget plan: highest expected net benefit per rupee first.

    Walk down the list ranked by (net benefit / cost) and add each retailer whose action still
    fits in the remaining budget. Fast and easy to explain, but not guaranteed to be the best
    possible combination (that would be a knapsack problem).
    """
    if decisions.empty:
        return decisions.copy(), 0.0, 0.0
    candidates = decisions[decisions.expected_net_benefit > 0].copy()
    candidates["selection_ratio"] = candidates.expected_net_benefit / candidates.intervention_cost.clip(lower=1)
    candidates = candidates.sort_values(["selection_ratio", "expected_net_benefit"], ascending=False)
    chosen, spent, benefit = [], 0.0, 0.0
    for idx, row in candidates.iterrows():
        cost = float(row.intervention_cost)
        if spent + cost <= budget:
            chosen.append(idx)
            spent += cost
            benefit += float(row.expected_net_benefit)
    selected = decisions.loc[chosen].copy() if chosen else decisions.iloc[0:0].copy()
    selected["budget_selection"] = "Selected within budget"
    return selected.sort_values("expected_net_benefit", ascending=False), spent, benefit


def winback_action(row, churn_days=60):
    """Suggested first step for a retailer that has already crossed the inactivity threshold."""
    if row.get("complaints_90d", 0) >= 1:
        return "Complaint resolution visit"
    if row.get("stockouts_90d", 0) >= 1:
        return "Stock replenishment review"
    if row.get("avg_payment_delay_90d", 0) >= 15:
        return "Credit and payment review"
    if row.get("days_since_last_purchase", 0) >= 2 * churn_days:
        return "Personal salesperson visit"
    return "Call / WhatsApp reactivation"


def build_winback_table(profiles, transactions, events, as_of, churn_days=60):
    """Retailers already inactive for >= churn_days. They are NOT scored by the 30-day model.

    There is no win-back probability model here, so the list is ranked by the contribution the
    retailer was generating before it lapsed (monthly average over the 180 days up to its last purchase).
    """
    as_of = pd.Timestamp(as_of)
    last = transactions.groupby("retailer_id").order_date.max().rename("last_purchase_date").reset_index()
    t = transactions.merge(last, on="retailer_id")
    window = t[(t.order_date > t.last_purchase_date - pd.Timedelta(days=180))]
    pre = (window.groupby("retailer_id").gross_contribution.sum() / 6.0).rename("pre_lapse_monthly_contribution").reset_index()
    out = profiles.merge(last, on="retailer_id").merge(pre, on="retailer_id", how="left")
    out["days_since_last_purchase"] = (as_of - out.last_purchase_date).dt.days
    out = out[out.days_since_last_purchase >= churn_days].copy()
    if out.empty:
        return out
    ev = events[(events.event_date > as_of - pd.Timedelta(days=90)) & (events.event_date <= as_of)]
    piv = ev.groupby(["retailer_id", "event_type"]).size().unstack(fill_value=0)
    for col in ["complaint", "stockout"]:
        out[f"{col}s_90d"] = out.retailer_id.map(piv[col]).fillna(0).astype(int) if col in piv else 0
    pay = ev[ev.event_type == "payment"].groupby("retailer_id").payment_delay_days.mean()
    out["avg_payment_delay_90d"] = out.retailer_id.map(pay).fillna(0.0)
    out["suggested_first_action"] = out.apply(lambda r: winback_action(r, churn_days), axis=1)
    cols = ["retailer_id", "retailer_name", "territory", "channel_type", "last_purchase_date", "days_since_last_purchase",
            "pre_lapse_monthly_contribution", "complaints_90d", "stockouts_90d", "avg_payment_delay_90d", "suggested_first_action"]
    return out[cols].sort_values("pre_lapse_monthly_contribution", ascending=False).reset_index(drop=True)


def add_observed_purchase_after_action(records: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Add the first purchase on/after each intervention's planned date, taken from transaction data.

    This is an OBSERVED follow-up only. It does not show the intervention caused the purchase:
    the retailer might have bought anyway, and there is no control group.
    """
    out = records.copy()
    if out.empty:
        out["first_purchase_after_action"] = pd.Series(dtype="object")
        return out
    planned = pd.to_datetime(out["planned_date"], errors="coerce")
    firsts = []
    by_ret = {k: g.order_date.sort_values().to_numpy() for k, g in transactions.groupby("retailer_id")}
    for rid, d in zip(out.retailer_id.astype(str), planned):
        dates = by_ret.get(rid)
        if dates is None or pd.isna(d):
            firsts.append(None)
            continue
        pos = int(np.searchsorted(dates, np.datetime64(d), side="left"))
        firsts.append(str(pd.Timestamp(dates[pos]).date()) if pos < len(dates) else None)
    out["first_purchase_after_action"] = firsts
    return out
