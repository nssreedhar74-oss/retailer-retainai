from __future__ import annotations

import html
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import (build_training_set, current_snapshot, generate_demo_data,
                      load_excel_workbook, validate_data)
from src.models import compute_metrics, reliability_table, train_and_score
from src.engine import (ACTION_ASSUMPTIONS, DEFAULT_HIGH_THRESHOLD, DEFAULT_MEDIUM_THRESHOLD,
                        add_observed_purchase_after_action, budget_plan, build_decision_table,
                        build_winback_table, risk_signals)
from src.storage import add_intervention, get_interventions, update_intervention

st.set_page_config(page_title="Retailer RetainAI – Predict, Prioritise and Retain", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

# Streamlit renamed use_container_width -> width="stretch" (1.50+). Support both.
def _ver(v):
    return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())


W = {"width": "stretch"} if _ver(st.__version__) >= (1, 50) else {"use_container_width": True}

# ----------------------------- Visual system -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#15243A; --muted:#66758B; --accent:#2563EB; --line:#E5EAF1; --surface:#FFFFFF; --bg:#F5F7FB; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); }
[data-testid="stSidebar"] { background:#111D31; border-right:1px solid #23334C; }
[data-testid="stSidebar"] * { color:#E7EEF9 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#AEBBD0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="input"] * { color:#15243A !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] * { color:#FFFFFF !important; }
.block-container { padding-top:1.45rem; padding-bottom:2rem; max-width:1500px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-0.025em; }
h1 { font-size:2.0rem !important; font-weight:800 !important; }
h2 { font-size:1.35rem !important; font-weight:750 !important; }
h3 { font-size:1.04rem !important; font-weight:700 !important; }
[data-testid="stMetric"] { background:white; border:1px solid var(--line); padding:16px 18px; border-radius:15px; box-shadow:0 4px 16px rgba(25,45,75,.035); }
[data-testid="stMetricLabel"] { color:#67768C; font-size:.82rem; }
[data-testid="stMetricValue"] { color:#16263E; font-family:'Manrope',sans-serif; font-weight:800; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:650; min-height:2.6rem; }
.stButton>button[kind="primary"] { background:#245DE8; border-color:#245DE8; }
div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px; }
div[data-testid="stAlert"] { border-radius:12px; }
.hero { background:linear-gradient(120deg,#12213A 0%,#1D3B68 58%,#245DE8 100%); padding:25px 28px; border-radius:18px; color:white; margin-bottom:18px; }
.hero h1 { color:white !important; margin:0 0 3px 0; font-size:1.9rem !important; }
.hero p { color:#D9E7FF !important; margin:0; }
.eyebrow { color:#8EB6FF; text-transform:uppercase; letter-spacing:.13em; font-size:.68rem; font-weight:800; margin-bottom:7px; }
.section-note { color:#718096; font-size:.88rem; margin-top:-8px; margin-bottom:13px; }
.pill { display:inline-block; padding:4px 9px; border-radius:20px; font-size:.74rem; font-weight:700; background:#EAF1FF; color:#2456BF; }
.small-muted { color:#718096; font-size:.78rem; }
hr { border-color:#E6EBF2; }
@media (max-width: 768px) {
  .block-container { padding-left:.8rem; padding-right:.8rem; }
  .hero { padding:18px 16px; border-radius:14px; }
  .hero h1 { font-size:1.4rem !important; }
  [data-testid="stMetricValue"] { font-size:1.35rem; }
}
</style>
""", unsafe_allow_html=True)


def money(x):
    return f"₹{float(x):,.0f}"


def hero(title, subtitle, eyebrow="SALES INTELLIGENCE"):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def plot_theme(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=35, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="DM Sans", color="#53647C"),
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(gridcolor="#EDF1F6", zeroline=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    return fig


@st.cache_data(show_spinner=False)
def demo_data():
    # The generator writes dates as text; validate_data converts them to real dates.
    return validate_data(*generate_demo_data(500, 42))


def get_loaded_data():
    if "profiles" not in st.session_state:
        p, t, e = demo_data()
        st.session_state.profiles, st.session_state.transactions, st.session_state.events = p, t, e
        st.session_state.data_source = "Synthetic demonstration data"
    return st.session_state.profiles, st.session_state.transactions, st.session_state.events


@st.cache_data(show_spinner="Building historical examples and training models...")
def cached_model_run(profiles, transactions, events, churn_days):
    training, cutoffs = build_training_set(profiles, transactions, events, churn_days=churn_days, horizon_days=30)
    current = current_snapshot(profiles, transactions, events, transactions.order_date.max(), churn_days=churn_days)
    output = train_and_score(training, current, churn_days=churn_days)
    output["cutoffs"] = len(cutoffs)
    output["current_as_of"] = transactions.order_date.max()
    return output


def demo_workbook_bytes(p, t, e):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        p.to_excel(xw, sheet_name="retailers", index=False)
        t.to_excel(xw, sheet_name="transactions", index=False)
        e.to_excel(xw, sheet_name="service_events", index=False)
    return buf.getvalue()


def download_csv(df, label, filename):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", **W)


# ----------------------------- Sidebar -----------------------------------
profiles, transactions, events = get_loaded_data()
all_territories = sorted(profiles.territory.dropna().astype(str).unique().tolist())
with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-size:1.3rem;font-weight:800;color:#fff'>◈ Retain<span style='color:#79A7FF'>AI</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.70rem;letter-spacing:.15em;color:#91A2BD;margin:2px 0 18px'>PREDICT · PRIORITISE · RETAIN</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Executive overview", "Predict & prioritise", "Retailer 360", "Retention planner", "Intervention tracker", "Data & model"], label_visibility="visible")
    st.markdown("---")
    churn_days = st.selectbox("Inactivity definition (days)", [45, 60, 75, 90], index=1, help="A retailer is treated as inactive after this many consecutive days without a purchase. Changing it rebuilds labels and retrains the models.")
    st.caption("Prediction horizon: next 30 days")
    territory_scope = st.multiselect("Territory filter", all_territories, default=all_territories,
                                     help="Applies to the overview, priority lists, win-back list and planner.")
    if not territory_scope:
        st.warning("No territory selected - showing all territories.")
        territory_scope = all_territories
    with st.expander("Risk band thresholds"):
        high_t = st.slider("High risk from probability", 0.10, 0.90, DEFAULT_HIGH_THRESHOLD, 0.05)
        med_t = st.slider("Medium risk from probability", 0.01, float(high_t), min(DEFAULT_MEDIUM_THRESHOLD, float(high_t)), 0.01)
        st.caption("Operating choices, not statistical standards. 30-day churn is usually uncommon, so bands sit well below 50%.")
    st.markdown("---")
    st.caption(f"Data source: {st.session_state.get('data_source','Synthetic demonstration data')}")
    st.caption(f"Retailer records: {len(profiles):,}")

# ----------------------------- Model cache -------------------------------
model_error = None
try:
    result = cached_model_run(profiles, transactions, events, int(churn_days))
    scores = result["scores"].copy()
    decisions_all = build_decision_table(scores, high=float(high_t), medium=float(med_t))
except Exception as exc:
    result, scores, decisions_all = None, pd.DataFrame(), pd.DataFrame()
    model_error = str(exc)

decisions = decisions_all[decisions_all.territory.isin(territory_scope)].reset_index(drop=True) if not decisions_all.empty else decisions_all
winback_all = build_winback_table(profiles, transactions, events, transactions.order_date.max(), int(churn_days))
inactive = winback_all[winback_all.territory.isin(territory_scope)].reset_index(drop=True) if not winback_all.empty else winback_all
profiles_in_scope = profiles[profiles.territory.isin(territory_scope)]

if model_error:
    st.error(f"The churn model could not be trained: {model_error}")
    st.info("Use the included synthetic dataset or upload a longer transaction history containing both churn and non-churn examples. Roughly 8-9 months of history is the minimum for three or more historical cutoffs.")

# ----------------------------- Executive overview ------------------------
if page == "Executive overview":
    hero("Retailer RetainAI – Predict, Prioritise and Retain", "Know which retailers are at risk, understand why, and direct retention spend where it can create value.")
    total = profiles_in_scope.retailer_id.nunique()
    active = len(decisions)
    high = int((decisions.risk_segment == "High").sum()) if not decisions.empty else 0
    avg_risk = float(decisions.churn_probability.mean()) if not decisions.empty else 0
    contribution = float((decisions.churn_probability * decisions.monthly_contribution_90d * 3).sum()) if not decisions.empty else 0
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Retailers in scope", f"{total:,}")
    c2.metric("Eligible active", f"{active:,}")
    c3.metric("High risk", f"{high:,}", delta=f"{high/max(active,1):.1%} of eligible" if active else None, delta_color="inverse")
    c4.metric("Average churn probability", f"{avg_risk:.1%}")
    c5.metric("Expected contribution at risk · 3 mo", money(contribution))
    st.markdown("### Portfolio health")
    left, right = st.columns([1.05, 1])
    with left:
        if not decisions.empty:
            counts = decisions.risk_segment.value_counts().reindex(["High","Medium","Low"], fill_value=0).reset_index()
            counts.columns = ["Risk segment", "Retailers"]
            fig = px.pie(counts, names="Risk segment", values="Retailers", hole=.65,
                         color="Risk segment", color_discrete_map={"High":"#E45B68","Medium":"#E9AD4D","Low":"#38A887"})
            fig.update_traces(textposition="inside", textinfo="percent+label")
            plot_theme(fig, 315)
            st.plotly_chart(fig, **W)
        else:
            st.info("No eligible active retailers to score.")
    with right:
        if not decisions.empty:
            terr = decisions.groupby("territory", as_index=False).agg(retailers=("retailer_id","count"), avg_risk=("churn_probability","mean"), contribution=("monthly_contribution_90d","sum"))
            fig = px.bar(terr.sort_values("avg_risk", ascending=True), x="avg_risk", y="territory", orientation="h",
                         color="avg_risk", color_continuous_scale=["#75C5AA","#F0C36D","#E45B68"],
                         labels={"avg_risk":"Average churn probability", "territory":"Territory"}, text=terr.sort_values("avg_risk", ascending=True).avg_risk.map(lambda v:f"{v:.0%}"))
            fig.update_layout(coloraxis_showscale=False)
            plot_theme(fig, 315)
            st.plotly_chart(fig, **W)
        else:
            st.info("Territory chart appears when scored retailers are available.")
    st.markdown("### Retailers requiring attention")
    if not decisions.empty:
        display_cols = ["retailer_id", "retailer_name", "territory", "churn_probability", "risk_segment", "monthly_contribution_90d", "recommended_action", "expected_net_benefit"]
        top = decisions.sort_values("expected_net_benefit", ascending=False)[display_cols].head(8).copy()
        top["churn_probability"] = top.churn_probability.map(lambda x:f"{x:.1%}")
        top["monthly_contribution_90d"] = top.monthly_contribution_90d.map(money)
        top["expected_net_benefit"] = top.expected_net_benefit.map(money)
        st.dataframe(top, hide_index=True, **W)
    else:
        st.info("No retailer predictions available.")
    if len(inactive):
        st.info(f"Win-back population (separate from the 30-day model): {len(inactive)} retailers in scope have already gone {churn_days}+ days without a purchase. "
                "Open **Predict & prioritise → Win-back** for the list.")
    st.caption("Expected contribution at risk = Σ churn probability × 3 months of recent monthly contribution, assuming a churned retailer contributes nothing.")
    st.markdown("<div class='small-muted'>Demo values are generated synthetic data. Predicted probabilities and evaluation metrics are illustrative and are not evidence of real-world performance.</div>", unsafe_allow_html=True)

# ----------------------------- Predict & prioritise -----------------------
elif page == "Predict & prioritise":
    hero("Predict & prioritise", "Focus sales capacity on retailers with the strongest commercial case, and handle already-inactive retailers separately.")
    tab_risk, tab_win = st.tabs(["At-risk retailers (next 30 days)", "Win-back (already inactive)"])
    with tab_risk:
        if decisions.empty:
            st.warning("No predictions available for the current dataset and territory filter.")
        else:
            f1, f2 = st.columns([1.2, 1])
            risk_filter = f1.multiselect("Risk segment", ["High", "Medium", "Low"], default=["High", "Medium"])
            sort_by = f2.selectbox("Rank retailers by", ["Expected net benefit", "Churn probability", "Monthly contribution"])
            view = decisions.copy()
            if risk_filter:
                view = view[view.risk_segment.isin(risk_filter)]
            sort_col = {"Expected net benefit": "expected_net_benefit", "Churn probability": "churn_probability", "Monthly contribution": "monthly_contribution_90d"}[sort_by]
            view = view.sort_values(sort_col, ascending=False)
            a, b, c = st.columns(3)
            a.metric("Retailers in view", f"{len(view):,}")
            b.metric("Positive expected net benefit", f"{int((view.expected_net_benefit > 0).sum()):,}")
            c.metric("Combined expected net benefit (positive cases only)", money(view.expected_net_benefit.clip(lower=0).sum()))
            if view.empty:
                st.info("No retailers match these filters.")
            else:
                show = view[["retailer_id", "retailer_name", "territory", "churn_probability", "risk_segment", "days_since_last_purchase", "monthly_contribution_90d", "recommended_action", "action_reason", "intervention_cost", "expected_net_benefit"]].copy()
                show["churn_probability"] = show.churn_probability.map(lambda x: f"{x:.1%}")
                show["monthly_contribution_90d"] = show.monthly_contribution_90d.map(money)
                show["intervention_cost"] = show.intervention_cost.map(money)
                show["expected_net_benefit"] = show.expected_net_benefit.map(money)
                st.dataframe(show, hide_index=True, height=440, **W)
                download_csv(view, "Download filtered priority list (CSV)", "retainai_priority_list.csv")
            st.caption("Churn probability comes from the model. Expected net benefit also uses assumed intervention uplift and cost (not estimated causal effects). A negative value means the assumed action does not pay back for that retailer, even if risk is high.")
    with tab_win:
        st.markdown(f"Retailers with **{churn_days}+ days since last purchase**. They have already reached the inactivity definition, so the 30-day churn model does not score them.")
        if inactive.empty:
            st.info("No already-inactive retailers in the selected territories.")
        else:
            w1, w2 = st.columns(2)
            w1.metric("Inactive retailers", f"{len(inactive):,}")
            w2.metric("Pre-lapse monthly contribution (total)", money(inactive.pre_lapse_monthly_contribution.sum()))
            show = inactive.copy()
            show["last_purchase_date"] = show.last_purchase_date.dt.date
            show["pre_lapse_monthly_contribution"] = show.pre_lapse_monthly_contribution.map(money)
            show["avg_payment_delay_90d"] = show.avg_payment_delay_90d.round(0)
            st.dataframe(show, hide_index=True, height=420, **W)
            download_csv(inactive, "Download win-back list (CSV)", "retainai_winback_list.csv")
            st.caption("Ranked by contribution before the retailer lapsed. There is no win-back probability model, so no expected net benefit is shown. Check whether reactivation economics are positive before committing sales time.")

# ----------------------------- Retailer 360 -------------------------------
elif page == "Retailer 360":
    hero("Retailer 360°", "A single account view combining purchase behaviour, service history, churn risk and next-best action.")
    options = sorted(profiles.retailer_id.astype(str).tolist())
    selected = st.selectbox("Select retailer", options, format_func=lambda x: f"{x} · {profiles.loc[profiles.retailer_id==x,'retailer_name'].iloc[0]}")
    profile = profiles[profiles.retailer_id == selected].iloc[0]
    tx = transactions[transactions.retailer_id == selected].sort_values("order_date")
    ev = events[events.retailer_id == selected].sort_values("event_date")
    score_row = decisions_all[decisions_all.retailer_id == selected] if not decisions_all.empty else decisions_all
    latest_asof = transactions.order_date.max()
    last_purchase = tx.order_date.max() if not tx.empty else pd.NaT
    recency = int((latest_asof-last_purchase).days) if pd.notna(last_purchase) else None
    st.markdown(f"### {html.escape(str(profile.retailer_name))}  <span class='pill'>{html.escape(str(profile.territory))}</span>", unsafe_allow_html=True)
    if len(score_row):
        r = score_row.iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("30-day churn probability", f"{r.churn_probability:.1%}")
        b.metric("Risk segment", r.risk_segment)
        c.metric("Monthly contribution", money(r.monthly_contribution_90d))
        d.metric("Expected net benefit", money(r.expected_net_benefit))
        st.markdown("#### Why is this retailer at risk?")
        signals = risk_signals(r)
        for s in signals: st.markdown(f"- {s}")
        action = r.recommended_action
        st.success(f"Recommended action: **{action}** — {ACTION_ASSUMPTIONS[action]['description']}  \nTriggered by: {r.action_reason}.")
        st.caption("Churn probability (model) and intervention uplift (assumption) are separate inputs to the expected net benefit shown above.")
    else:
        if recency is None:
            st.warning("No transaction history was found for this retailer. Confirm the retailer ID and purchase-data coverage before making a retention decision.")
        else:
            st.warning(f"This retailer is already inactive: {recency} days since its last purchase. It is outside the prospective churn model and should be assessed through a separate win-back process.")
            wb = winback_all[winback_all.retailer_id == selected]
            if len(wb):
                st.info(f"Suggested win-back first step: **{wb.iloc[0].suggested_first_action}**. Pre-lapse monthly contribution was about {money(wb.iloc[0].pre_lapse_monthly_contribution)}. Assess whether reactivation economics are positive before committing sales time.")
    left,right = st.columns([1.25,1])
    with left:
        st.markdown("#### Purchase history")
        if not tx.empty:
            monthly = tx.assign(month=tx.order_date.dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False).agg(sales=("order_value","sum"), orders=("order_id","count"))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=monthly.month, y=monthly.sales, name="Sales", marker_color="#3977E8"))
            fig.add_trace(go.Scatter(x=monthly.month, y=monthly.orders, name="Orders", yaxis="y2", mode="lines+markers", line=dict(color="#E9A63A",width=3)))
            fig.update_layout(yaxis=dict(title="Sales (₹)", gridcolor="#EDF1F6"), yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False), barmode="group")
            plot_theme(fig, 330)
            st.plotly_chart(fig, **W)
        else:
            st.info("No purchase records for this retailer.")
    with right:
        st.markdown("#### Service and engagement")
        counts = ev.event_type.value_counts().reindex(["complaint","stockout","payment","visit"], fill_value=0).reset_index()
        counts.columns = ["Event", "Count"]
        fig = px.bar(counts, x="Event", y="Count", color="Event", color_discrete_sequence=["#E47770","#E9B45C","#7186B8","#48A58B"])
        fig.update_layout(showlegend=False)
        plot_theme(fig, 260)
        st.plotly_chart(fig, **W)
        st.caption(f"Last purchase: {last_purchase.date() if pd.notna(last_purchase) else 'No recorded purchase'} · {recency if recency is not None else '—'} days ago · {len(tx)} recorded orders")
    with st.expander("View transaction records"):
        st.dataframe(tx.sort_values("order_date", ascending=False), hide_index=True, **W)
    with st.expander("View service event records"):
        st.dataframe(ev.sort_values("event_date", ascending=False), hide_index=True, **W)

# ----------------------------- Retention planner --------------------------
elif page == "Retention planner":
    hero("Retention planner", "Turn churn probabilities into an affordable action plan using explicit, editable commercial assumptions.")
    if decisions.empty:
        st.warning("No predictions available for planning in the current dataset and territory filter.")
    else:
        x1, x2 = st.columns(2)
        budget = x1.number_input("Available retention budget (₹)", min_value=0, max_value=10_000_000, value=25000, step=1000)
        future_months = x2.slider("Contribution protection horizon (months)", 1, 12, 3)
        st.markdown("##### Assumptions (editable)")
        st.caption("Cost and uplift are **assumptions, not causal estimates**. Uplift = the share of an at-risk retailer's churn risk the action is assumed to remove. It is deliberately separate from the model's churn probability.")
        base_df = pd.DataFrame([{"Action": k, "Cost per retailer (₹)": v["cost"], "Assumed uplift (0-1)": v["uplift"]} for k, v in ACTION_ASSUMPTIONS.items()])
        edited = st.data_editor(base_df, hide_index=True, disabled=["Action"], key="assumption_editor",
                                column_config={"Cost per retailer (₹)": st.column_config.NumberColumn(min_value=0.0, step=25.0),
                                               "Assumed uplift (0-1)": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01)}, **W)
        scale = st.slider("Scenario uplift scaling", 0.50, 1.50, 1.00, 0.05, help="Multiplies every uplift assumption above, for quick what-if testing.")
        edited = edited.fillna(0)
        costs = dict(zip(edited["Action"], edited["Cost per retailer (₹)"].clip(lower=0)))
        uplifts = {a: float(np.clip(u * scale, 0.0, 1.0)) for a, u in zip(edited["Action"], edited["Assumed uplift (0-1)"])}
        plan_df = build_decision_table(scores[scores.territory.isin(territory_scope)], future_months=future_months,
                                       cost_overrides=costs, uplift_overrides=uplifts, high=float(high_t), medium=float(med_t))
        selected, spent, net = budget_plan(plan_df, float(budget))
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Budget", money(budget))
        p2.metric("Planned spend", money(spent))
        p3.metric("Retailers selected", f"{len(selected)}")
        p4.metric("Expected net benefit", money(net))
        st.markdown("#### Recommended portfolio")
        n_pos = int((plan_df.expected_net_benefit > 0).sum())
        if selected.empty:
            st.info("No intervention has positive expected net benefit within this budget under the current assumptions. Try a longer protection horizon, lower costs or higher assumed uplift, and see how sensitive the plan is.")
        else:
            table = selected[["retailer_id", "retailer_name", "territory", "churn_probability", "recommended_action", "action_reason", "incremental_save_probability", "future_contribution", "intervention_cost", "expected_net_benefit"]].copy()
            table["churn_probability"] = table.churn_probability.map(lambda x: f"{x:.1%}")
            table["incremental_save_probability"] = table.incremental_save_probability.map(lambda x: f"{x:.1%}")
            for col in ["future_contribution", "intervention_cost", "expected_net_benefit"]:
                table[col] = table[col].map(money)
            st.dataframe(table, hide_index=True, **W)
            download_csv(selected, "Download budget action plan (CSV)", "retainai_budget_plan.csv")
        if spent < float(budget) and n_pos <= len(selected):
            st.caption(f"{money(float(budget) - spent)} of the budget is unspent because only {n_pos} retailer(s) have positive expected net benefit. Spending more would, under these assumptions, destroy value.")
        with st.expander("How the plan is built (formula and heuristic)", expanded=True):
            st.markdown("**Expected net benefit = (churn probability × assumed uplift) × future contribution − intervention cost**")
            st.markdown("- *Churn probability* comes from the calibrated model.\n- *Assumed uplift* and *cost* come from the table above and are not learned from data.\n- *Future contribution* = recent monthly gross contribution × protection horizon, assuming a lost retailer would contribute nothing.")
            st.markdown("**Budget heuristic (greedy):** keep only retailers with positive net benefit, rank them by net benefit per rupee spent, then add them in that order while the action cost still fits in the remaining budget. It is quick and easy to audit, but it is not guaranteed to find the best possible combination, and it tends to favour cheap actions.")

# ----------------------------- Intervention tracker -----------------------
elif page == "Intervention tracker":
    hero("Intervention tracker", "Record sales actions and outcomes so the organisation can learn which interventions work over time.")
    STATUSES = ["Planned", "In progress", "Completed", "Cancelled"]
    OUTCOMES = ["Pending", "Purchased again", "No purchase yet", "Declined", "Unreachable"]
    available_ids = sorted(profiles.retailer_id.astype(str).tolist())
    default_actions = list(ACTION_ASSUMPTIONS.keys())
    st.markdown("### Log an intervention")
    c1, c2 = st.columns(2)
    rid = c1.selectbox("Retailer", available_ids)
    rec_row = decisions_all[decisions_all.retailer_id == rid] if not decisions_all.empty else decisions_all
    default_idx = default_actions.index(rec_row.iloc[0].recommended_action) if len(rec_row) else 0
    action = c2.selectbox("Action", default_actions, index=default_idx, help="Pre-selected to the model's recommendation when the retailer is scored.")
    with st.form("intervention_form", clear_on_submit=True):
        c3, c4, c5 = st.columns(3)
        owner = c3.text_input("Salesperson / owner", placeholder="Name or employee ID")
        planned = c4.date_input("Planned date", value=date.today())
        status = c5.selectbox("Status", STATUSES)
        c6, c7 = st.columns(2)
        actual_cost = c6.number_input("Actual cost (₹)", min_value=0.0, value=float(ACTION_ASSUMPTIONS[action]["cost"]), step=50.0)
        outcome = c7.selectbox("Outcome", OUTCOMES)
        notes = st.text_area("Visit / call notes", placeholder="Record what happened, objections, commitments and next steps.")
        submitted = st.form_submit_button("Save intervention", type="primary", **W)
        if submitted:
            add_intervention(rid, action, owner.strip(), planned, status, actual_cost, outcome, notes)
            st.success("Intervention saved to the local SQLite tracker.")
    records = add_observed_purchase_after_action(get_interventions(), transactions)
    st.markdown("### Recorded interventions")
    if records.empty:
        st.info("No interventions recorded yet. Use the form above to create the first record.")
    else:
        done = records[records.status == "Completed"]
        reactivated = records[records.outcome == "Purchased again"]
        a, b, c, d = st.columns(4)
        a.metric("Actions logged", len(records))
        b.metric("Completed", len(done))
        c.metric("Outcome logged: purchased again", len(reactivated))
        d.metric("Total actual cost", money(records.actual_cost.fillna(0).sum()))
        st.dataframe(records, hide_index=True, **W)
        download_csv(records, "Download tracker records (CSV)", "retainai_interventions.csv")
        st.caption("'first_purchase_after_action' is looked up from the loaded transaction data (first order on/after the planned date). It is observed follow-up only.")
        with st.expander("Update an intervention outcome"):
            record_ids = records.id.astype(int).tolist()
            chosen_id = st.selectbox("Intervention record", record_ids)
            rec = records[records.id == chosen_id].iloc[0]
            with st.form("update_form"):
                u1, u2 = st.columns(2)
                new_status = u1.selectbox("Status", STATUSES, index=STATUSES.index(rec.status) if rec.status in STATUSES else 0)
                new_outcome = u2.selectbox("Outcome", OUTCOMES, index=OUTCOMES.index(rec.outcome) if rec.outcome in OUTCOMES else 0)
                new_cost = st.number_input("Actual cost (₹)", min_value=0.0, value=float(rec.actual_cost or 0), step=50.0)
                new_notes = st.text_area("Notes", value=rec.notes or "")
                if st.form_submit_button("Update record", type="primary"):
                    update_intervention(chosen_id, new_status, new_cost, new_outcome, new_notes)
                    st.rerun()
        st.caption("A retailer purchasing again after an intervention is an observed outcome, not proof that the intervention caused the purchase. Causal uplift requires a suitable control group or experimental design.")

# ----------------------------- Data & model -------------------------------
elif page == "Data & model":
    hero("Data & model centre", "Manage input data, inspect the training design and understand model performance and limitations.")
    st.markdown("### 1 · Upload retailer data")
    st.markdown("Upload one Excel workbook with sheets named **retailers**, **transactions**, **service_events**, or upload the three CSV files separately.")
    excel_file = st.file_uploader("Excel workbook (.xlsx)", type=["xlsx"], key="excel_upload")
    u1,u2,u3 = st.columns(3)
    tx_file = u1.file_uploader("Transactions CSV", type=["csv"], key="tx_upload")
    profile_file = u2.file_uploader("Retailers CSV", type=["csv"], key="profile_upload")
    events_file = u3.file_uploader("Service events CSV", type=["csv"], key="events_upload")
    st.caption("If an Excel workbook is selected it is used and the CSV files are ignored. Only transactions are mandatory for CSV upload; missing retailer or event files are filled with defaults.")
    if st.button("Load uploaded dataset", type="primary", disabled=(excel_file is None and tx_file is None), **W):
        try:
            if excel_file is not None:
                p, t, e, report = load_excel_workbook(excel_file, return_report=True)
            else:
                t = pd.read_csv(tx_file)
                if profile_file is not None:
                    p = pd.read_csv(profile_file)
                else:
                    p = pd.DataFrame({"retailer_id": t.retailer_id.astype(str).unique(), "territory": "Unassigned"})
                e = pd.read_csv(events_file) if events_file is not None else pd.DataFrame(columns=["retailer_id", "event_date", "event_type", "payment_delay_days"])
                p, t, e, report = validate_data(p, t, e, return_report=True)
            st.session_state.profiles, st.session_state.transactions, st.session_state.events = p, t, e
            st.session_state.data_source = "Uploaded retailer dataset"
            st.session_state.validation_report = report
            cached_model_run.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Upload failed: {exc}")
    report = st.session_state.get("validation_report")
    if report:
        labels = {"duplicate_retailers_removed": "Duplicate retailers removed", "duplicate_transactions_removed": "Duplicate transactions removed",
                  "duplicate_events_removed": "Duplicate service events removed", "non_numeric_transactions_removed": "Transactions removed (non-numeric values)",
                  "negative_value_transactions_removed": "Transactions removed (negative values)", "unrecognised_event_types": "Events with unrecognised type (kept, unused by features)",
                  "retailers_added_from_transactions": "Retailers added from transactions (missing in master)", "events_for_unknown_retailers_removed": "Events removed (unknown retailer)"}
        issues = {labels[k]: v for k, v in report.items() if v}
        if issues:
            st.warning("Validation adjusted the uploaded data: " + "; ".join(f"{k}: {v}" for k, v in issues.items()))
        else:
            st.success("Validation passed with no rows removed or adjusted.")
    r1,r2 = st.columns(2)
    with r1:
        if st.button("Restore 500-retailer synthetic demo", **W):
            p,t,e = demo_data()
            st.session_state.profiles, st.session_state.transactions, st.session_state.events = p,t,e
            st.session_state.data_source = "Synthetic demonstration data"
            st.session_state.pop("validation_report", None)
            cached_model_run.clear()
            st.rerun()
    with r2:
        download_csv(profiles, "Download retailers CSV", "demo_retailers.csv")
        download_csv(transactions, "Download transactions CSV", "demo_transactions.csv")
        download_csv(events, "Download service events CSV", "demo_service_events.csv")
        st.download_button("Download all three as one Excel workbook", demo_workbook_bytes(profiles, transactions, events),
                           file_name="retainai_dataset.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **W)
    st.markdown("### 2 · Dataset health")
    h1,h2,h3,h4 = st.columns(4)
    h1.metric("Retailers", f"{profiles.retailer_id.nunique():,}")
    h2.metric("Transactions", f"{len(transactions):,}")
    h3.metric("Service events", f"{len(events):,}")
    h4.metric("History covered", f"{(transactions.order_date.max()-transactions.order_date.min()).days} days")
    with st.expander("Preview uploaded / active data"):
        st.markdown("**Retailers**")
        st.dataframe(profiles.head(10), hide_index=True, **W)
        st.markdown("**Transactions**")
        st.dataframe(transactions.head(10), hide_index=True, **W)
        st.markdown("**Service events**")
        st.dataframe(events.head(10), hide_index=True, **W)
    st.markdown("### 3 · Model evaluation")
    if result:
        st.success(f"Model used for current scoring: **{result['model_name']}** · {result['n_train']:,} training rows · {result['n_test']:,} time-held-out test rows")
        st.caption(f"Held-out period: {result['test_start'].date()} to {result['test_end'].date()} · {result['cutoffs']} historical cutoffs · Current scoring date: {result['current_as_of'].date()}")
        st.warning("Metrics below come from synthetic data (or your upload's history). They show how the pipeline behaves, not how it would perform on real retailers.")
        st.info(result["calibration_note"])
        if result.get("baseline_beats_ml"):
            st.info("On this held-out period the transparent rule-based baseline has a higher PR-AUC than the selected ML model, so the ML model adds ranking value only on some measures. Judge on your own data before preferring it.")
        threshold = st.slider("Decision threshold for precision / recall / F1 / confusion matrix", 0.05, 0.95, 0.50, 0.05,
                              help="A retailer is flagged as 'will churn' when its score is at or above this value. Lower values catch more churners but raise false alarms. PR-AUC and calibration do not depend on it.")
        y_test = result["test_labels"]
        evals = {name: compute_metrics(y_test, p, threshold) for name, p in result["test_probs"].items()}
        metric_rows = [{"Model": n, "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
                        "Brier score": m["brier_score"], "10-bin calibration error": m["calibration_error_10bin"]} for n, m in evals.items()]
        metrics_df = pd.DataFrame(metric_rows)
        st.dataframe(metrics_df.style.format({c: "{:.3f}" for c in metrics_df.columns if c != "Model"}), hide_index=True, **W)
        st.caption(f"Share of retailer-cutoff examples that churned in the held-out period: {evals['Rule-based baseline']['test_positive_rate']:.1%}. "
                   "The rule-based baseline is a transparent heuristic score, not a calibrated probability, so its threshold, Brier score and calibration error are not comparable with the ML models. "
                   "The model was chosen using this same held-out period, so its figures are somewhat optimistic.")
        cm_col, cal_col = st.columns(2)
        with cm_col:
            shown = st.selectbox("Confusion matrix for", list(evals.keys()), index=list(evals.keys()).index(result["model_name"]))
            cm = np.array(evals[shown]["confusion_matrix"])
            fig = go.Figure(data=go.Heatmap(z=cm, x=["Predicted active", "Predicted churn"], y=["Actual active", "Actual churn"],
                                            text=cm, texttemplate="%{text}", colorscale=[[0, "#EAF0FA"], [1, "#3E75D9"]], showscale=False))
            fig.update_layout(title=f"{shown} · threshold {threshold:.2f}", xaxis_title="Prediction", yaxis_title="Observed outcome")
            plot_theme(fig, 320)
            st.plotly_chart(fig, **W)
        with cal_col:
            cal_model = st.selectbox("Calibration diagnostic for", [n for n in evals if n != "Rule-based baseline"], index=[n for n in evals if n != "Rule-based baseline"].index(result["model_name"]))
            rel = reliability_table(y_test, result["test_probs"][cal_model])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(dash="dash", color="#9AA7BA")))
            fig.add_trace(go.Scatter(x=rel.mean_predicted, y=rel.observed_rate, mode="lines+markers", name=cal_model,
                                     marker=dict(size=np.clip(rel.n / rel.n.max() * 22, 6, 22)), line=dict(color="#245DE8"),
                                     customdata=rel.n, hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>n=%{customdata}<extra></extra>"))
            fig.update_layout(title="Predicted vs observed churn rate (held-out period)", xaxis_title="Mean predicted probability", yaxis_title="Observed churn rate")
            plot_theme(fig, 320)
            st.plotly_chart(fig, **W)
        st.markdown("### 4 · Modelling assumptions and responsible use")
        st.markdown(f"- **Target:** reaches {churn_days} consecutive days without a purchase at some point during the next 30 days, among retailers not already inactive at the cutoff. Already-inactive retailers form a separate win-back list.\n- **Leakage control:** input features use records on or before each historical cutoff; labels use only the following 30 days. Whole cutoff dates are assigned to train or test, with the latest dates held out.\n- **Models:** a rule-based baseline score, Logistic Regression and Random Forest are compared on the same held-out period. Only the two ML models can be used for scoring.\n- **Risk bands:** adjustable in the sidebar (currently High ≥ {high_t:.0%}, Medium ≥ {med_t:.0%}). They are operating thresholds, not statistical standards.\n- **Calibration:** probabilities are sigmoid-calibrated; re-check on new, real-world periods before any business use.\n- **Commercial uplift:** intervention effectiveness and costs are illustrative assumptions, not estimated causal effects.\n- **Low-frequency / seasonal outlets:** consider cohort-specific inactivity windows rather than applying one rule to every outlet.")
    else:
        st.warning("Model evaluation is not available until the model has enough labelled historical examples.")

st.markdown("<div style='text-align:center;color:#95A1B2;font-size:.72rem;padding:20px 0 4px'>RETAILER RETAINAI · MBA / PGDM BUILD WITH AI PROJECT · SYNTHETIC DEMO DATA IS CLEARLY LABELLED</div>", unsafe_allow_html=True)
