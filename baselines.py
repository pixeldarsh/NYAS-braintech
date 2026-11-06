import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("eeg_features.csv")

abs_cols = [c for c in df.columns if c.endswith("_absolute")]
rel_cols = [c for c in df.columns if c.endswith("_relative")]
all_cols = abs_cols + rel_cols

# Absolute power spans ~1e-12..1e-9, so log-transform it for linear models
X = df[all_cols].copy()
X[abs_cols] = np.log10(X[abs_cols] + 1e-20)
y = df["sleep_stage"].to_numpy()
groups = df["subject"].to_numpy()

stages = ["Wake", "N1", "N2", "N3", "REM"]


def models():
    return {
        "Majority class": (DummyClassifier(strategy="most_frequent"), all_cols),
        "Stratified random": (DummyClassifier(strategy="stratified", random_state=42), all_cols),
        "Logistic regression (all)": (
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")), all_cols),
        "Logistic regression (relative only)": (
            make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced")), rel_cols),
        "Random forest (relative only)": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1), rel_cols),
        "Random forest (all)": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1), all_cols),
    }


def uncertainty(proba):
    s = np.sort(proba, axis=1)
    return {
        "confidence": proba.max(axis=1),
        "margin": s[:, -1] - s[:, -2],
        "neg_entropy": xlogy(proba, proba).sum(axis=1),  # higher = more certain
    }


rows, per_subject = [], []
for name, (model, cols) in models().items():
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(stages)))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        model.fit(X.iloc[tr][cols], y[tr])
        pred[te] = model.predict(X.iloc[te][cols])
        p = model.predict_proba(X.iloc[te][cols])
        for k, cls in enumerate(model.classes_):
            proba[te, stages.index(cls)] = p[:, k]

    correct = pred == y
    f1_by = f1_score(y, pred, labels=stages, average=None, zero_division=0)
    row = {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=stages, average="macro", zero_division=0),
        **{f"f1_{s}": v for s, v in zip(stages, f1_by)},
    }

    # Error-detection AUC (can the model's own confidence flag its mistakes?)
    if len(np.unique(correct)) == 2 and proba.max() > 0:
        for k, v in uncertainty(proba).items():
            row[f"auc_{k}"] = roc_auc_score(correct, v)

    rows.append(row)

    for s in np.unique(groups):
        m = groups == s
        per_subject.append({
            "model": name, "subject": s,
            "accuracy": accuracy_score(y[m], pred[m]),
            "macro_f1": f1_score(y[m], pred[m], labels=stages, average="macro", zero_division=0),
        })

summary = pd.DataFrame(rows).round(4)
subj = pd.DataFrame(per_subject)
spread = subj.groupby("model")[["accuracy", "macro_f1"]].agg(["mean", "std", "min", "max"]).round(3)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n===== Leave-one-subject-out, pooled held-out epochs =====")
print(summary.to_string(index=False))
print("\n===== Per-subject spread (n=5 subjects) =====")
print(spread.to_string())

summary.to_csv("baseline_comparison.csv", index=False)
subj.to_csv("baseline_per_subject.csv", index=False)
print("\nSaved baseline_comparison.csv and baseline_per_subject.csv")
