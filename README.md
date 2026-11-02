# Retailer RetainAI – Predict, Prioritise and Retain

A classroom demo that predicts which of 500 synthetic FMCG retailers are likely to stop buying in the next 30 days, ranks them by commercial value, and tracks the follow-up actions. Everything runs locally: no API keys, no paid services.

> **All demo data is synthetic. Model metrics describe how the pipeline behaves on that data. They are not evidence of real-world performance.**

## Run locally

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The first load builds the historical training examples and trains the models, which takes roughly 30-40 seconds; the result is then cached. Changing the inactivity definition or loading new data retrains.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover churn labels, leakage (features cannot see data after the cutoff), the time-based split, validation, recommendation rules, the win-back list, budget planning and the SQLite tracker.

## Input data

Upload either one Excel workbook (sheets `retailers`, `transactions`, `service_events`) or CSV files, on the **Data & model** page. The page also lets you download the demo data as CSV or as a ready-made workbook to see the expected format.

| Table | Required columns | Optional columns |
|---|---|---|
| retailers | `retailer_id`, `territory` | `retailer_name`, `channel_type`, `tier` |
| transactions | `retailer_id`, `order_date`, `order_value`, `gross_contribution` | `order_id`, `units`, `product_category` |
| service_events | `retailer_id`, `event_date`, `event_type` (`complaint`, `stockout`, `payment`, `visit`) | `payment_delay_days` |

Validation: column names are normalised; dates must parse (the error names the rows); duplicates, non-numeric and negative values are removed and **counted in a validation report** shown after upload; events for unknown retailers are removed; retailers missing from the master table are added with territory "Unknown". For CSV upload only transactions are mandatory.

Minimum history: features look back up to 180 days and every cutoff needs a full 30-day label window, so you need roughly 8-9 months of transactions to get three or more cutoffs.

## How it works

1. **Inactivity definition.** A retailer is *inactive* after 60 consecutive days without a purchase (sidebar: 45/60/75/90).
2. **Target.** Among retailers *not yet inactive* at a cutoff date, will they reach the inactivity threshold within the next 30 days? A purchase on or before the threshold date prevents the label.
3. **Win-back population.** Retailers already past the threshold are not scored. They appear in a separate list (**Predict & prioritise → Win-back**), ranked by contribution before they lapsed. No win-back probability model exists, so no expected benefit is shown for them.
4. **Historical examples.** Cutoffs are placed every 30 days. Features use records on or before the cutoff; the label uses only the following 30 days.
5. **Evaluation.** Whole cutoff dates go to training (earlier) or test (latest ~25%). Compared: a transparent rule-based score, Logistic Regression and Random Forest (sigmoid-calibrated). Shown: precision, recall, F1, PR-AUC, ROC-AUC, Brier score, a 10-bin calibration error, a reliability plot and confusion matrices at an adjustable threshold.
6. **Scoring.** The better ML model (by held-out PR-AUC) is refitted on all labelled history and scores currently active retailers. Risk bands High / Medium / Low use adjustable probability thresholds (defaults 30% / 10%).
7. **Actions.** One action per retailer from fixed, explainable rules, each with the evidence that triggered it: ≥2 complaints → complaint resolution; ≥2 stock-outs → stock replenishment; average payment delay ≥15 days → credit/payment review; elevated risk with no visit in 30 days or a ≥25% sales drop → salesperson visit; other elevated risk → reactivation call; otherwise routine check.
8. **Commercial prioritisation.**
   `expected net benefit = (churn probability × assumed uplift) × future contribution − intervention cost`.
   Churn probability is from the model. **Uplift and cost are editable assumptions, not causal estimates.** The budget planner uses a greedy heuristic: keep positive-net-benefit retailers, rank by net benefit per rupee, add while the cost fits. It is easy to audit but not guaranteed optimal and tends to favour cheap actions.
9. **Tracker.** SQLite (`data/retention.db`, created on first save) stores action, owner, date, status, actual cost, outcome and notes. The tracker also looks up each retailer's first purchase on/after the planned date from the loaded transactions.

## Exam demonstration flow (≈10 minutes)

1. **Executive overview:** state the churn definition and horizon; point to risk mix, territory chart and the win-back note. Say that data is synthetic.
2. **Data & model:** show the dataset health and validation report; explain cutoffs, leakage control and the time-based split. Compare the three models, move the threshold slider to show the precision/recall trade-off, and read the reliability plot. Note that the baseline can be competitive and that the model was chosen on the same held-out period, so metrics are optimistic.
3. **Predict & prioritise:** filter by territory (sidebar) and risk band; show the reason for each action; open the **Win-back** tab to show why already-inactive retailers are separate; export a CSV.
4. **Retailer 360:** pick a high-risk retailer and walk through signals, history and recommended action; then pick a win-back retailer.
5. **Retention planner:** set a budget; edit cost and uplift assumptions; show that high risk does not always mean positive net benefit and that budget can go unspent; explain the greedy heuristic and its limitation.
6. **Intervention tracker:** log an action, update its outcome, and stress that a later purchase is observed, not caused by the action.

## Known limitations

- Synthetic data only; with real data, validate on a later period before use. The model is chosen and evaluated on the same held-out period, and the final model is refitted on all history, so reported metrics are indicative only.
- Calibration folds inside the training period are stratified rather than chronological (chronological folds were tested but the earliest months contain almost no churn events).
- Uplift and costs are illustrative assumptions. Causal uplift needs a control group or experiment. The 30-day churn base rate in the demo is low, so most retailers are Low risk and few have positive expected net benefit at default costs; this is a result of the assumptions, not a bug.
- One recommended action per retailer from rules, not an optimised treatment policy.
- A single 60-day rule may be wrong for seasonal or low-frequency outlets; consider cohort-specific windows.
- Training takes about 20-30 seconds for 500 retailers because features are built row by row for clarity.
- The tracker is a local single-user SQLite file, not a multi-user system.
