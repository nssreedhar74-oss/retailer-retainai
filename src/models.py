from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])


def reliability_table(y_true, prob, bins=10):
    """Calibration diagnostic: mean predicted probability vs observed churn rate per bin."""
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)  # last bin includes 1.0
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}", "n": int(m.sum()),
                         "mean_predicted": float(prob[m].mean()), "observed_rate": float(y_true[m].mean())})
    return pd.DataFrame(rows)


def compute_metrics(y_true, prob, threshold=0.5):
    """Metrics for one model on the held-out period at a chosen decision threshold."""
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    both = len(np.unique(y_true)) > 1
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, prob)) if both else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
        "test_positive_rate": float(np.mean(y_true)),
        "brier_score": float(brier_score_loss(y_true, np.clip(prob, 0, 1))),
    }
    rel = reliability_table(y_true, prob)
    out["calibration_error_10bin"] = float((rel.n / rel.n.sum() * (rel.mean_predicted - rel.observed_rate).abs()).sum()) if len(rel) else float("nan")
    return out


def _metrics(y_true, prob, threshold=0.5):  # kept for backward compatibility
    return compute_metrics(y_true, prob, threshold)


def rule_score(frame, churn_days=60):
    """Transparent baseline: a weighted 0-1 score built only from observed behaviour.

    It is a ranking score, NOT a calibrated probability.
    """
    g = lambda c: frame[c].to_numpy(dtype=float)
    recency = np.clip(g("days_since_last_purchase") / float(churn_days), 0, 1)
    trend = np.clip(-g("sales_change_pct") / 0.75, 0, 1)
    frequency = np.clip(1 - g("orders_90d") / 6.0, 0, 1)
    complaints = np.clip(g("complaints_90d") / 3.0, 0, 1)
    stockouts = np.clip(g("stockouts_90d") / 3.0, 0, 1)
    payment = np.clip(g("avg_payment_delay_90d") / 30.0, 0, 1)
    no_visit = (g("visits_30d") == 0).astype(float)
    return (0.28 * recency + 0.20 * trend + 0.13 * frequency +
            0.12 * complaints + 0.10 * stockouts + 0.10 * payment + 0.07 * no_visit)


def train_and_score(training: pd.DataFrame, current: pd.DataFrame, threshold=0.5, churn_days=60):
    if training.empty or training.churned_next_30d.nunique() < 2:
        raise ValueError("Not enough labelled examples from both outcome classes. Add more transaction history or use the demo dataset.")
    training = training.sort_values("as_of_date").reset_index(drop=True)
    unique_dates = sorted(training.as_of_date.unique())
    if len(unique_dates) < 2:
        raise ValueError("At least two historical cutoff dates are needed for a time-based train/test split. Upload a longer transaction history.")
    # Whole cutoff dates go to either train or test (earlier = train, later = test).
    split_at = unique_dates[max(1, int(len(unique_dates) * 0.75))]
    train_mask = training.as_of_date < split_at
    test_mask = ~train_mask
    if training.loc[train_mask, "churned_next_30d"].nunique() < 2 or training.loc[test_mask, "churned_next_30d"].nunique() < 2:
        split_at = unique_dates[-2]
        train_mask = training.as_of_date < split_at
        test_mask = ~train_mask
    train_df, test_df = training.loc[train_mask], training.loc[test_mask]
    if len(test_df) == 0 or len(train_df) == 0 or train_df.churned_next_30d.nunique() < 2:
        raise ValueError("Unable to form a valid time-based train/test split. Upload a longer history with both churn and non-churn examples.")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df.churned_next_30d.astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df.churned_next_30d.astype(int)
    baseline_prob = rule_score(X_test, churn_days)
    test_probs = {"Rule-based baseline": baseline_prob}
    evaluations = {"Rule-based baseline": compute_metrics(y_test.to_numpy(), baseline_prob, threshold)}
    models = {
        "Logistic Regression": Pipeline([("prep", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", C=0.8, random_state=42))]),
        "Random Forest": Pipeline([("prep", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=140, max_depth=7, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
    }
    notes = []
    for name, model in models.items():
        try:
            # cv=3 uses stratified folds inside the training period only (never the test period).
            # Folds are not chronological: time-ordered folds were tested but the earliest periods hold
            # almost no churn events, which made the calibrators unreliable.
            cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cal.fit(X_train, y_train)
            p = cal.predict_proba(X_test)[:, 1]
        except Exception:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[:, 1]
            notes.append(f"{name}: calibration failed on this dataset, so raw probabilities are shown.")
        test_probs[name] = p
        evaluations[name] = compute_metrics(y_test.to_numpy(), p, threshold)

    # Choose between the two ML models by held-out PR-AUC (fallback to F1 where PR-AUC is undefined).
    # NOTE: the same held-out period is used to choose and to report, so the chosen model's
    # metrics are slightly optimistic. Treat them as indicative only.
    def selection_score(m):
        v = evaluations[m]["pr_auc"]
        return v if np.isfinite(v) else evaluations[m]["f1"]
    selected_name = max(models, key=selection_score)

    # Final calibrated model is refitted on all labelled historical examples after the
    # later-period evaluation is complete. The evaluation metrics above use calibrators
    # trained only on the earlier training period.
    final_base = models[selected_name]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=3)
            calibrated.fit(training[FEATURE_COLUMNS], training.churned_next_30d.astype(int))
        final_model = calibrated
        calibration_note = "Sigmoid calibration (3 stratified folds, training period only) was fitted for evaluation, then the chosen model was refitted on all labelled history for current scoring. The held-out period is therefore not independent of the final model."
    except Exception:
        final_base.fit(training[FEATURE_COLUMNS], training.churned_next_30d.astype(int))
        final_model = final_base
        calibration_note = "Calibration was not stable on this dataset; raw model probabilities are shown. Interpret probability estimates cautiously."

    current_scores = current.copy()
    if not current_scores.empty:
        current_scores["churn_probability"] = final_model.predict_proba(current_scores[FEATURE_COLUMNS])[:, 1]
    return {
        "model": final_model, "model_name": selected_name, "evaluations": evaluations,
        "test_start": pd.Timestamp(test_df.as_of_date.min()),
        "test_end": pd.Timestamp(test_df.as_of_date.max()),
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "calibration_note": calibration_note + (" " + " ".join(notes) if notes else ""),
        "scores": current_scores, "training": training,
        "test_labels": y_test.to_numpy(), "test_probs": test_probs,
        "baseline_beats_ml": bool(evaluations["Rule-based baseline"]["pr_auc"] > evaluations[selected_name]["pr_auc"]),
    }
