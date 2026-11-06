"""Blind test of the frozen sleep-stage model on 3 unseen participants.

Fixed before the new recordings were scored:
- Participants: SC4051, SC4061, SC4071, the next three in numerical order.
- Model: trainmodel.py random forest settings, fitted on all 5 development participants.
  Saved to frozen_rf.joblib before any new epoch is scored. No retraining or tuning.
- Preprocessing and features: same as preprocesseeg.py and extractfeatures.py.
- Primary result: pooled and per-person accuracy and macro-F1, with all epochs kept
  (same as development). Compared with development LOSO: 0.765 accuracy, 0.551 macro-F1.
- Secondary, evaluation only: epochs within 30 min of sleep; a majority-class baseline;
  a logistic regression frozen the same way; error-detection AUC; 90% conformal coverage
  with the threshold taken from development out-of-fold predictions; and the N2 and N3
  gating policies from integration.py.

If these results are used to change the model, these people are no longer blind.
"""
import os

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = "sleep_edf_blind"
BLIND = ["SC4051E0", "SC4061E0", "SC4071E0"]
stages = ["Wake", "N1", "N2", "N3", "REM"]
stage_mapping = {
    "Sleep stage W": "Wake", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (13, 16), "beta": (16, 30)}
# Column order from trainmodel.py
cols = [f"{b}_{kind}" for b in bands for kind in ("absolute", "relative")]


def load_subject(subject):
    raw = mne.io.read_raw_edf(os.path.join(DATA_DIR, f"{subject}-PSG.edf"), preload=True, verbose=False)
    raw.pick(["EEG Fpz-Cz"])
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)
    annotations = mne.read_annotations(os.path.join(DATA_DIR, subject.replace("E0", "EC") + "-Hypnogram.edf"))

    epochs, labels = [], []
    for annotation in annotations:
        stage = stage_mapping.get(annotation["description"])
        if stage is None:
            continue
        for i in range(int(annotation["duration"] // 30)):
            start = annotation["onset"] + i * 30
            if start + 30 > raw.times[-1]:
                continue
            data, _ = raw[:, int(start * 100):int((start + 30) * 100)]
            if data.shape[1] != 3000:
                continue
            epochs.append(data[0])
            labels.append(stage)
    return np.array(epochs), np.array(labels)


def features(epoch):
    f, psd = welch(epoch, fs=100, nperseg=1000)
    tm = (f >= 0.5) & (f <= 30)
    total = np.trapezoid(psd[tm], f[tm])
    row = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        a = np.trapezoid(psd[m], f[m])
        row[f"{name}_absolute"] = a
        row[f"{name}_relative"] = a / total if total > 0 else 0
    return row


def full_proba(model, X):
    p = model.predict_proba(X)
    out = np.zeros((len(X), 5))
    for k, c in enumerate(model.classes_):
        out[:, stages.index(c)] = p[:, k]
    return out


def near_sleep(labels, minutes=30):
    sleep = np.where(labels != "Wake")[0]
    pad = minutes * 2
    keep = np.zeros(len(labels), dtype=bool)
    keep[max(0, sleep[0] - pad):sleep[-1] + pad + 1] = True
    return keep


def persistent(mask, k=3):
    return np.convolve(mask.astype(int), np.ones(k, dtype=int))[:len(mask)] >= k


def metrics(y, pred):
    return {
        "epochs": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=stages, average="macro", zero_division=0),
        **{f"f1_{s}": v for s, v in zip(stages, f1_score(y, pred, labels=stages, average=None, zero_division=0))},
    }


# 1. Freeze models on development data only
dev = pd.read_csv("eeg_features.csv")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(dev[cols], dev["sleep_stage"])
joblib.dump(rf, "frozen_rf.joblib")

logged = dev[cols].copy()
abs_cols = [c for c in cols if c.endswith("_absolute")]
logged[abs_cols] = np.log10(logged[abs_cols] + 1e-20)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
lr.fit(logged, dev["sleep_stage"])

# Conformal threshold from development out-of-fold predictions, so it never sees the new people
oof = pd.read_csv("oof_predictions.csv")
scores = 1 - oof[stages].to_numpy()[np.arange(len(oof)), [stages.index(s) for s in oof["actual_stage"]]]
n = len(scores)
q = np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher")
print(f"Models frozen. Conformal threshold from development data: {q:.3f}", flush=True)

# 2. Score unseen participants
rows, gating = [], []
pool = {"y": [], "proba": [], "lr": [], "near": []}
for s in BLIND:
    X_raw, y = load_subject(s)
    table = pd.DataFrame([features(e) for e in X_raw])[cols]
    proba = full_proba(rf, table)
    pred = np.array(stages)[proba.argmax(1)]

    tl = table.copy()
    tl[abs_cols] = np.log10(tl[abs_cols] + 1e-20)
    lr_pred = lr.predict(tl)
    near = near_sleep(y)

    print(f"{s}: {len(y)} epochs, " + ", ".join(f"{k} {(y == k).sum()}" for k in stages), flush=True)
    for name, yt, pt in [("RF", y, pred), ("LR", y, lr_pred), ("Majority", y, np.full(len(y), "Wake")),
                         ("RF, within 30 min of sleep", y[near], pred[near]),
                         ("LR, within 30 min of sleep", y[near], lr_pred[near]),
                         ("Majority, within 30 min of sleep", y[near], np.full(near.sum(), "Wake"))]:
        rows.append({"subject": s, "model": name, **metrics(yt, pt)})

    for target in ("N2", "N3"):
        conf = proba.max(1)
        for pname, mask in {
            "argmax": pred == target,
            "conf >= 0.7": (pred == target) & (conf >= 0.7),
            "conf >= 0.7, 3 consecutive": persistent((pred == target) & (conf >= 0.7)),
        }.items():
            gating.append({"subject": s, "target": target, "policy": pname, "triggers": int(mask.sum()),
                           "true_target": int((y[mask] == target).sum()), "in_wake": int((y[mask] == "Wake").sum()),
                           "target_epochs": int((y == target).sum())})

    pool["y"].append(y); pool["proba"].append(proba); pool["lr"].append(lr_pred); pool["near"].append(near)

y = np.concatenate(pool["y"])
proba = np.vstack(pool["proba"])
lr_pred = np.concatenate(pool["lr"])
near = np.concatenate(pool["near"])
pred = np.array(stages)[proba.argmax(1)]
for name, yt, pt in [("RF", y, pred), ("LR", y, lr_pred), ("Majority", y, np.full(len(y), "Wake")),
                     ("RF, within 30 min of sleep", y[near], pred[near]),
                     ("LR, within 30 min of sleep", y[near], lr_pred[near]),
                     ("Majority, within 30 min of sleep", y[near], np.full(near.sum(), "Wake"))]:
    rows.append({"subject": "POOLED", "model": name, **metrics(yt, pt)})

res = pd.DataFrame(rows)
res.to_csv("blind_test_results.csv", index=False)
gate = pd.DataFrame(gating)
gate.to_csv("blind_test_gating.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n===== Blind test =====")
print(res.round(3).to_string(index=False))

print("\nConfusion matrix, RF pooled (rows = true, columns = predicted):")
print(pd.DataFrame(confusion_matrix(y, pred, labels=stages), index=stages, columns=stages).to_string())

correct = pred == y
conf = proba.max(1)
print(f"\nError-detection AUC (confidence): pooled {roc_auc_score(correct, conf):.3f}")
idx = np.array([stages.index(v) for v in y])
sets = (1 - proba) <= q
covered = sets[np.arange(len(y)), idx]
print(f"Conformal: coverage {covered.mean():.3f} (target 0.90), mean set size {sets.sum(1).mean():.2f}")
offset = 0
for s, part in zip(BLIND, pool["y"]):
    print(f"  {s} coverage {covered[offset:offset + len(part)].mean():.3f}")
    offset += len(part)
print("  by stage: " + ", ".join(f"{k} {covered[idx == i].mean():.3f}" for i, k in enumerate(stages) if (idx == i).any()))

print("\nGating:")
print(gate.to_string(index=False))
