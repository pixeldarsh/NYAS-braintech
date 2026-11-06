"""Wake vs sleep performance of the frozen 5-stage model (evaluation only, no retraining).

P(sleep) = P(N1) + P(N2) + P(N3) + P(REM). Predict sleep when P(sleep) >= 0.5.
Confidence = the probability of the predicted class, max(P(sleep), P(Wake)).
Blind participants use frozen_rf.joblib; development participants use the saved
leave-one-participant-out predictions in oof_predictions.csv.
"""
import os

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import cohen_kappa_score, roc_auc_score

DATA_DIR = "sleep_edf_blind"
BLIND = ["SC4051E0", "SC4061E0", "SC4071E0"]
stages = ["Wake", "N1", "N2", "N3", "REM"]
stage_mapping = {
    "Sleep stage W": "Wake", "Sleep stage 1": "N1", "Sleep stage 2": "N2",
    "Sleep stage 3": "N3", "Sleep stage 4": "N3", "Sleep stage R": "REM",
}
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (13, 16), "beta": (16, 30)}
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
            if data.shape[1] == 3000:
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


def near_sleep(labels, minutes=30):
    sleep = np.where(labels != "Wake")[0]
    keep = np.zeros(len(labels), dtype=bool)
    keep[max(0, sleep[0] - minutes * 2):sleep[-1] + minutes * 2 + 1] = True
    return keep


def summarise(group, subject, scope, y, p_sleep):
    asleep = y != "Wake"
    pred = p_sleep >= 0.5
    conf = np.maximum(p_sleep, 1 - p_sleep)
    correct = pred == asleep
    edges = np.linspace(0.5, 1.0, 6)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 4)
    ece = sum((idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
              for b in range(5) if (idx == b).any())
    return {
        "group": group, "subject": subject, "scope": scope, "epochs": len(y),
        "accuracy": correct.mean(),
        "always_majority": max(asleep.mean(), 1 - asleep.mean()),
        "balanced_accuracy": (pred[asleep].mean() + (~pred[~asleep]).mean()) / 2,
        "sleep_detected": pred[asleep].mean(),
        "wake_detected": (~pred[~asleep]).mean(),
        "sleep_call_correct": asleep[pred].mean() if pred.any() else np.nan,
        "wake_call_correct": (~asleep[~pred]).mean() if (~pred).any() else np.nan,
        "kappa": cohen_kappa_score(asleep, pred),
        "auc": roc_auc_score(asleep, p_sleep),
        "mean_confidence": conf.mean(),
        "ece": ece,
    }


def confidence_bins(y, p_sleep):
    asleep = y != "Wake"
    pred = p_sleep >= 0.5
    conf = np.maximum(p_sleep, 1 - p_sleep)
    out = pd.DataFrame({"conf": conf, "correct": pred == asleep})
    out["bin"] = pd.cut(out["conf"], [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], include_lowest=True)
    t = out.groupby("bin", observed=False).agg(
        epochs=("correct", "size"), mean_confidence=("conf", "mean"), observed_accuracy=("correct", "mean"))
    t["share_of_epochs"] = t["epochs"] / len(out)
    return t


rf = joblib.load("frozen_rf.joblib")
rows, blind_y, blind_p, blind_near = [], [], [], []
for s in BLIND:
    X, y = load_subject(s)
    table = pd.DataFrame([features(e) for e in X])[cols]
    wake_col = list(rf.classes_).index("Wake")
    p_sleep = 1 - rf.predict_proba(table)[:, wake_col]
    near = near_sleep(y)
    rows.append(summarise("blind", s, "all epochs", y, p_sleep))
    rows.append(summarise("blind", s, "within 30 min of sleep", y[near], p_sleep[near]))
    blind_y.append(y); blind_p.append(p_sleep); blind_near.append(near)

y, p_sleep, near = np.concatenate(blind_y), np.concatenate(blind_p), np.concatenate(blind_near)
rows.append(summarise("blind", "POOLED", "all epochs", y, p_sleep))
rows.append(summarise("blind", "POOLED", "within 30 min of sleep", y[near], p_sleep[near]))

oof = pd.read_csv("oof_predictions.csv")
dev_y, dev_p, dev_near = [], [], []
for s, part in oof.groupby("subject", sort=True):
    yd = part["actual_stage"].to_numpy()
    pd_sleep = 1 - part["Wake"].to_numpy()
    nd = near_sleep(yd)
    rows.append(summarise("development", s, "all epochs", yd, pd_sleep))
    rows.append(summarise("development", s, "within 30 min of sleep", yd[nd], pd_sleep[nd]))
    dev_y.append(yd); dev_p.append(pd_sleep); dev_near.append(nd)
yd, pds, nd = np.concatenate(dev_y), np.concatenate(dev_p), np.concatenate(dev_near)
rows.append(summarise("development", "POOLED", "all epochs", yd, pds))
rows.append(summarise("development", "POOLED", "within 30 min of sleep", yd[nd], pds[nd]))

res = pd.DataFrame(rows)
res.to_csv("wake_sleep_results.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(res.round(3).to_string(index=False))

print("\nConfidence vs observed accuracy, blind, all epochs:")
print(confidence_bins(y, p_sleep).round(3).to_string())
print("\nConfidence vs observed accuracy, blind, within 30 min of sleep:")
print(confidence_bins(y[near], p_sleep[near]).round(3).to_string())

asleep = y[near] != "Wake"
pred = p_sleep[near] >= 0.5
print("\nBlind, within 30 min of sleep, accuracy by true stage:")
for k in stages:
    m = y[near] == k
    print(f"  {k}: {(pred[m] == asleep[m]).mean():.3f} ({m.sum()} epochs)")

# Cut-off chosen on development out-of-fold predictions only (best balanced accuracy),
# then applied unchanged. Added after seeing that 0.5 under-detects Wake on the blind
# participants, so the decision to adjust was informed by them; the value was not.
dev_sleep = yd != "Wake"
grid = np.arange(0.05, 0.96, 0.01)
bal = [((pds[dev_sleep] >= t).mean() + (pds[~dev_sleep] < t).mean()) / 2 for t in grid]
cut = float(grid[int(np.argmax(bal))])
print(f"\nCut-off from development data: P(sleep) >= {cut:.2f}")


def at_cut(yy, pp):
    a = yy != "Wake"
    pr = pp >= cut
    return {
        "epochs": len(yy), "accuracy": (pr == a).mean(), "always_majority": max(a.mean(), 1 - a.mean()),
        "balanced_accuracy": (pr[a].mean() + (~pr[~a]).mean()) / 2,
        "sleep_detected": pr[a].mean(), "wake_detected": (~pr[~a]).mean(),
        "sleep_call_correct": a[pr].mean(), "wake_call_correct": (~a[~pr]).mean(),
        "kappa": cohen_kappa_score(a, pr),
    }


out = []
for s, yy, pp, nn in zip(BLIND, blind_y, blind_p, blind_near):
    out.append({"subject": s, "scope": "all epochs", **at_cut(yy, pp)})
    out.append({"subject": s, "scope": "within 30 min of sleep", **at_cut(yy[nn], pp[nn])})
out.append({"subject": "POOLED", "scope": "all epochs", **at_cut(y, p_sleep)})
out.append({"subject": "POOLED", "scope": "within 30 min of sleep", **at_cut(y[near], p_sleep[near])})
cut_res = pd.DataFrame(out)
cut_res.to_csv("wake_sleep_cutoff_results.csv", index=False)
print(cut_res.round(3).to_string(index=False))

pr = p_sleep[near] >= cut
print("\nBlind, within 30 min of sleep, at this cut-off, accuracy by true stage:")
for k in stages:
    m = y[near] == k
    print(f"  {k}: {(pr[m] == (k != 'Wake')).mean():.3f} ({m.sum()} epochs)")
