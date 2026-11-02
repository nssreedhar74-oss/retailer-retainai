from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_TX = {"retailer_id", "order_date", "order_value", "gross_contribution"}
REQUIRED_RETAILERS = {"retailer_id", "territory"}
REQUIRED_EVENTS = {"retailer_id", "event_date", "event_type"}
KNOWN_EVENT_TYPES = {"complaint", "stockout", "payment", "visit"}
FEATURE_COLUMNS = [
    "days_since_last_purchase", "orders_30d", "orders_90d", "avg_order_value_90d",
    "sales_30d", "sales_prev_30d", "sales_change_pct", "complaints_90d",
    "stockouts_90d", "avg_payment_delay_90d", "visits_30d",
    "monthly_contribution_90d", "avg_gap_days_180d", "territory", "channel_type",
]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in {"territory", "channel_type"}]
CATEGORICAL_FEATURES = ["territory", "channel_type"]


def generate_demo_data(n_retailers: int = 500, seed: int = 42):
    """Create labelled synthetic retailer profiles, transactions and service events.

    The generator creates ~15 months of events: a 12-month operating history plus
    a follow-through period for historical 30-day labels. All records are synthetic.
    """
    rng = np.random.default_rng(seed)
    end = date(2026, 11, 2)
    start = end - timedelta(days=449)
    territories = ["North", "South", "East", "West", "Central"]
    channels = ["Kirana", "General Trade", "Mini Market", "Low-frequency Outlet"]
    retailers, tx_rows, event_rows = [], [], []

    for i in range(n_retailers):
        rid = f"R{i+1:04d}"
        channel = rng.choice(channels, p=[0.47, 0.28, 0.17, 0.08])
        tier = rng.choice(["Small", "Medium", "Large"], p=[0.50, 0.35, 0.15])
        base_gap = float({"Small": rng.uniform(14, 32), "Medium": rng.uniform(8, 22), "Large": rng.uniform(5, 15)}[tier])
        if channel == "Low-frequency Outlet":
            base_gap *= 1.65
        latent_risk = float(rng.beta(2.0, 4.0))
        contribution_target = float({"Small": rng.uniform(900, 4500), "Medium": rng.uniform(3500, 12000), "Large": rng.uniform(9000, 30000)}[tier])
        contribution_target *= float(rng.uniform(0.75, 1.25))
        territory = str(rng.choice(territories))
        retailer_name = f"{rng.choice(['Sri', 'New', 'Maha', 'Green', 'Anand', 'Royal', 'City', 'Fresh'])} {rng.choice(['Stores', 'Traders', 'Agency', 'Mart', 'Provisions', 'Supermarket'])} {i+1:03d}"
        # A proportion of outlets experience a sustained purchase stop during the period.
        stop_prob = 0.10 + 0.48 * latent_risk
        stop_day = int(rng.integers(230, 410)) if rng.random() < stop_prob else None
        trend_strength = float(rng.uniform(-0.30, 0.12) if latent_risk > 0.45 else rng.uniform(-0.08, 0.18))
        retailers.append({
            "retailer_id": rid, "retailer_name": retailer_name, "territory": territory,
            "channel_type": channel, "tier": tier,
            "monthly_contribution_target": round(contribution_target, 2),
        })

        day_idx = int(rng.integers(0, 12))
        order_no = 0
        while day_idx <= 449:
            if stop_day is not None and day_idx >= stop_day:
                break
            progress = day_idx / 449
            deterioration = max(0.0, progress - 0.42) * (0.5 + latent_risk * 2.0)
            gap = max(4.0, base_gap * (1.0 + deterioration + max(0.0, -trend_strength) * progress))
            sampled_gap = max(3, int(rng.lognormal(np.log(gap), 0.30)))
            if rng.random() < latent_risk * 0.10:
                sampled_gap += int(rng.integers(15, 36))
            day_idx += sampled_gap
            if day_idx > 449 or (stop_day is not None and day_idx >= stop_day):
                break
            order_no += 1
            order_date = start + timedelta(days=day_idx)
            seasonal = 1.0 + 0.12 * np.sin((order_date.month - 1) * 2 * np.pi / 12)
            order_value = contribution_target * float(rng.uniform(0.55, 1.55)) * seasonal * (1 + trend_strength * progress)
            order_value = max(350.0, order_value)
            margin_rate = float(rng.uniform(0.10, 0.23))
            tx_rows.append({
                "retailer_id": rid, "order_id": f"{rid}-O{order_no:04d}",
                "order_date": order_date.isoformat(), "order_value": round(order_value, 2),
                "gross_contribution": round(order_value * margin_rate, 2),
                "units": int(max(1, order_value / rng.uniform(80, 250))),
                "product_category": str(rng.choice(["Staples", "Snacks", "Beverages", "Home Care", "Personal Care"])),
            })

        # Service events are correlated with the latent risk and purchase deterioration.
        event_count = int(rng.poisson(2.0 + latent_risk * 4.0))
        for _ in range(event_count):
            d = int(rng.integers(0, 450))
            event_type = str(rng.choice(["complaint", "stockout", "payment", "visit"], p=[0.18 + latent_risk*0.12, 0.22, 0.22, 0.38 - latent_risk*0.12]))
            delay = int(max(0, rng.normal(5 + latent_risk * 17, 7))) if event_type == "payment" else 0
            event_rows.append({
                "retailer_id": rid, "event_date": (start + timedelta(days=d)).isoformat(),
                "event_type": event_type, "payment_delay_days": delay,
            })

    profiles = pd.DataFrame(retailers)
    transactions = pd.DataFrame(tx_rows)
    events = pd.DataFrame(event_rows)
    return profiles, transactions, events


def _parse_dates(df: pd.DataFrame, col: str, label: str = "data") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    bad = out[col].isna()
    if bad.any():
        rows = ", ".join(str(i + 2) for i in out.index[bad][:5])  # +2 = spreadsheet row incl. header
        raise ValueError(f"{label}: {int(bad.sum())} row(s) have a blank or invalid '{col}' "
                         f"(e.g. spreadsheet rows {rows}). Correct these dates and upload again.")
    return out


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def validate_data(profiles: pd.DataFrame, transactions: pd.DataFrame, events: pd.DataFrame,
                  return_report: bool = False):
    """Validate and clean the three input tables.

    Raises ValueError with a plain-English message for structural problems. Row-level
    problems (duplicates, negative or non-numeric values) are removed and counted in a
    report, so the user can see exactly what was dropped. Set return_report=True to get
    the report as a fourth return value.
    """
    profiles, transactions, events = _clean_columns(profiles), _clean_columns(transactions), _clean_columns(events)
    report = {}
    missing = REQUIRED_TX - set(transactions.columns)
    if missing:
        raise ValueError(f"Transactions are missing required columns: {', '.join(sorted(missing))}. "
                         f"Required: {', '.join(sorted(REQUIRED_TX))}.")
    if not REQUIRED_RETAILERS.issubset(profiles.columns):
        raise ValueError("Retailers table must contain retailer_id and territory columns.")
    if not REQUIRED_EVENTS.issubset(events.columns):
        raise ValueError("Service events table must contain retailer_id, event_date and event_type columns.")

    for frame in (profiles, transactions, events):
        frame["retailer_id"] = frame["retailer_id"].astype(str).str.strip()
    n = len(profiles)
    profiles = profiles.drop_duplicates("retailer_id", keep="last")
    report["duplicate_retailers_removed"] = n - len(profiles)

    # Duplicate orders: use order_id where it is filled in; otherwise fall back to
    # retailer + date + value. (Blank order_ids must not be treated as duplicates of each other.)
    n = len(transactions)
    if "order_id" in transactions.columns:
        has_id = transactions["order_id"].notna() & (transactions["order_id"].astype(str).str.strip() != "")
        with_id = transactions[has_id].drop_duplicates("order_id")
        no_id = transactions[~has_id].drop_duplicates(["retailer_id", "order_date", "order_value"])
        transactions = pd.concat([with_id, no_id])
    else:
        transactions = transactions.drop_duplicates(["retailer_id", "order_date", "order_value"])
    report["duplicate_transactions_removed"] = n - len(transactions)
    n = len(events)
    events = events.drop_duplicates()
    report["duplicate_events_removed"] = n - len(events)

    transactions = _parse_dates(transactions, "order_date", "Transactions")
    events = _parse_dates(events, "event_date", "Service events")
    for c in ["order_value", "gross_contribution"]:
        transactions[c] = pd.to_numeric(transactions[c], errors="coerce")
    n = len(transactions)
    transactions = transactions.dropna(subset=["retailer_id", "order_value", "gross_contribution"])
    report["non_numeric_transactions_removed"] = n - len(transactions)
    n = len(transactions)
    transactions = transactions[(transactions.order_value >= 0) & (transactions.gross_contribution >= 0)]
    report["negative_value_transactions_removed"] = n - len(transactions)
    if transactions.empty:
        raise ValueError("No valid transactions remain after validation. Check order_value and gross_contribution are numeric and non-negative.")

    if "payment_delay_days" not in events:
        events["payment_delay_days"] = 0
    events["payment_delay_days"] = pd.to_numeric(events.payment_delay_days, errors="coerce").fillna(0).clip(lower=0)
    events["event_type"] = events["event_type"].astype(str).str.lower().str.strip()
    report["unrecognised_event_types"] = int((~events.event_type.isin(KNOWN_EVENT_TYPES)).sum())
    profiles["territory"] = profiles["territory"].fillna("Unknown").astype(str)

    # Keep transaction retailers even when the uploaded master file is incomplete.
    missing_ids = sorted(set(transactions.retailer_id) - set(profiles.retailer_id))
    report["retailers_added_from_transactions"] = len(missing_ids)
    if missing_ids:
        additions = pd.DataFrame({"retailer_id": missing_ids, "territory": "Unknown"})
        profiles = pd.concat([profiles, additions], ignore_index=True)
    known = set(profiles.retailer_id)
    n = len(events)
    events = events[events.retailer_id.isin(known)]
    report["events_for_unknown_retailers_removed"] = n - len(events)

    for col, default in [("retailer_name", None), ("channel_type", "General Trade"), ("tier", "Unclassified")]:
        if col not in profiles:
            profiles[col] = profiles["retailer_id"] if default is None else default
        else:
            profiles[col] = profiles[col].fillna(profiles["retailer_id"] if default is None else default)
    out = (profiles.reset_index(drop=True), transactions.sort_values("order_date").reset_index(drop=True),
           events.sort_values("event_date").reset_index(drop=True))
    return (*out, report) if return_report else out


def load_excel_workbook(file_obj, return_report: bool = False):
    xls = pd.ExcelFile(file_obj)
    names = {s.strip().lower(): s for s in xls.sheet_names}
    required = {"retailers", "transactions", "service_events"}
    if not required.issubset(names):
        raise ValueError("Excel workbook must contain three sheets named: retailers, transactions, service_events "
                         f"(found: {', '.join(xls.sheet_names)}).")
    return validate_data(
        pd.read_excel(xls, names["retailers"]),
        pd.read_excel(xls, names["transactions"]),
        pd.read_excel(xls, names["service_events"]),
        return_report=return_report,
    )


def make_features(retailer_id: str, as_of: pd.Timestamp, profiles: pd.DataFrame,
                  transactions: pd.DataFrame, events: pd.DataFrame, tx_lookup=None,
                  events_lookup=None, profile_lookup=None) -> Optional[dict]:
    all_tx = tx_lookup.get(retailer_id) if tx_lookup is not None else transactions[transactions.retailer_id == retailer_id]
    if all_tx is None:
        return None
    tx = all_tx[all_tx.order_date <= as_of].sort_values("order_date")
    if tx.empty:
        return None
    last_date = tx.order_date.max()
    recency = int((as_of - last_date).days)
    recent90 = tx[tx.order_date > as_of - pd.Timedelta(days=90)]
    last30 = tx[tx.order_date > as_of - pd.Timedelta(days=30)]
    prev30 = tx[(tx.order_date > as_of - pd.Timedelta(days=60)) & (tx.order_date <= as_of - pd.Timedelta(days=30))]
    recent180 = tx[tx.order_date > as_of - pd.Timedelta(days=180)]
    sales30 = float(last30.order_value.sum())
    sales_prev = float(prev30.order_value.sum())
    sales_change = (sales30 - sales_prev) / max(sales_prev, 1.0)
    gaps = recent180.order_date.sort_values().diff().dt.days.dropna()
    if profile_lookup is not None:
        profile = profile_lookup[retailer_id]
    else:
        profile = profiles.loc[profiles.retailer_id == retailer_id].iloc[0]
    all_events = events_lookup.get(retailer_id) if events_lookup is not None else events[events.retailer_id == retailer_id]
    if all_events is None:
        all_events = events.iloc[0:0]
    ev = all_events[(all_events.event_date > as_of - pd.Timedelta(days=90)) & (all_events.event_date <= as_of)]
    return {
        "retailer_id": retailer_id, "as_of_date": pd.Timestamp(as_of),
        "days_since_last_purchase": recency,
        "orders_30d": int(len(last30)), "orders_90d": int(len(recent90)),
        "avg_order_value_90d": float(recent90.order_value.mean()) if len(recent90) else float(tx.order_value.tail(3).mean()),
        "sales_30d": sales30, "sales_prev_30d": sales_prev,
        "sales_change_pct": float(np.clip(sales_change, -1.0, 3.0)),
        "complaints_90d": int((ev.event_type == "complaint").sum()),
        "stockouts_90d": int((ev.event_type == "stockout").sum()),
        "avg_payment_delay_90d": float(ev.loc[ev.event_type == "payment", "payment_delay_days"].mean()) if (ev.event_type == "payment").any() else 0.0,
        "visits_30d": int(((all_events.event_type == "visit") & (all_events.event_date > as_of - pd.Timedelta(days=30)) & (all_events.event_date <= as_of)).sum()),
        "monthly_contribution_90d": float(recent90.gross_contribution.sum() / 3.0),
        "avg_gap_days_180d": float(gaps.mean()) if len(gaps) else 30.0,
        "territory": str(profile.territory), "channel_type": str(profile.channel_type),
        "retailer_name": str(profile.retailer_name),
    }


def is_churn_event(retailer_id: str, cutoff: pd.Timestamp, horizon_days: int,
                   churn_days: int, transactions: pd.DataFrame, date_lookup=None) -> int:
    """Label 1 if the retailer reaches the inactivity threshold during the horizon.

    Once the last purchase is known, the threshold date is last_purchase + churn_days.
    The retailer reaches the threshold only if that date is inside the horizon and no
    new purchase occurs on or before that threshold date.
    """
    dates = date_lookup.get(retailer_id) if date_lookup is not None else transactions.loc[transactions.retailer_id == retailer_id, "order_date"].sort_values()
    pos = int(dates.searchsorted(cutoff, side="right"))
    if pos == 0:
        return 0
    last_purchase = dates.iloc[pos - 1]
    threshold_date = last_purchase + pd.Timedelta(days=churn_days)
    if threshold_date <= cutoff or threshold_date > cutoff + pd.Timedelta(days=horizon_days):
        return 0
    next_purchase = dates.iloc[pos] if pos < len(dates) else pd.NaT
    return int(pd.isna(next_purchase) or next_purchase > threshold_date)


def build_training_set(profiles: pd.DataFrame, transactions: pd.DataFrame, events: pd.DataFrame,
                       churn_days: int = 60, horizon_days: int = 30, cutoff_step: int = 30):
    min_date, max_date = transactions.order_date.min(), transactions.order_date.max()
    first_cutoff = min_date + pd.Timedelta(days=180)
    last_cutoff = max_date - pd.Timedelta(days=horizon_days)
    cutoffs = list(pd.date_range(first_cutoff, last_cutoff, freq=f"{cutoff_step}D"))
    rows = []
    ids = profiles.retailer_id.astype(str).tolist()
    tx_lookup = {str(k): g.sort_values("order_date") for k, g in transactions.groupby("retailer_id", sort=False)}
    events_lookup = {str(k): g.sort_values("event_date") for k, g in events.groupby("retailer_id", sort=False)}
    profile_lookup = {str(r.retailer_id): r for r in profiles.itertuples(index=False)}
    date_lookup = {rid: g.order_date.sort_values() for rid, g in tx_lookup.items()}
    for cutoff in cutoffs:
        for rid in ids:
            feat = make_features(rid, cutoff, profiles, transactions, events, tx_lookup, events_lookup, profile_lookup)
            if feat is None or feat["days_since_last_purchase"] >= churn_days:
                continue
            feat["churned_next_30d"] = is_churn_event(rid, cutoff, horizon_days, churn_days, transactions, date_lookup)
            rows.append(feat)
    return pd.DataFrame(rows), cutoffs


def current_snapshot(profiles, transactions, events, as_of, churn_days=60):
    rows = []
    tx_lookup = {str(k): g.sort_values("order_date") for k, g in transactions.groupby("retailer_id", sort=False)}
    events_lookup = {str(k): g.sort_values("event_date") for k, g in events.groupby("retailer_id", sort=False)}
    profile_lookup = {str(r.retailer_id): r for r in profiles.itertuples(index=False)}
    for rid in profiles.retailer_id.astype(str):
        row = make_features(rid, pd.Timestamp(as_of), profiles, transactions, events, tx_lookup, events_lookup, profile_lookup)
        if row is not None and row["days_since_last_purchase"] < churn_days:
            rows.append(row)
    return pd.DataFrame(rows)
