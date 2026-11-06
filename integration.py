"""Software-only integration of the EEG classifier and the acoustic model.

A hypothetical closed-loop *gating* study: a trigger fires when the EEG classifier says a
chosen target stage is happening. We measure how often triggers would be mistimed (wrong
true stage) under different gating policies and EEG noise levels, and compare that
variability with the variability from acoustic tissue-property uncertainty.

Nothing here models a physiological effect of ultrasound, and the 'delivered index' is a
relative number in arbitrary units, not a dose or safety quantity. The choice of target
stage is a free parameter and is not supported by this work.
"""
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut

stages = ["Wake", "N1", "N2", "N3", "REM"]
bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (13, 16), "beta": (16, 30)}
cols = [f"{b}_absolute" for b in bands] + [f"{b}_relative" for b in bands]

epochs = np.load("eeg_epochs.npy")
feat = pd.read_csv("eeg_features.csv")
y = feat["sleep_stage"].to_numpy()
subj = feat["subject"].to_numpy()


def features(epoch):
    f, psd = welch(epoch, fs=100, nperseg=1000)
    tm = (f >= 0.5) & (f <= 30)
    total = np.trapezoid(psd[tm], f[tm])
    out = {}
    for n, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        a = np.trapezoid(psd[m], f[m])
        out[f"{n}_absolute"], out[f"{n}_relative"] = a, (a / total if total > 0 else 0)
    return out


def noisy(snr, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for e in epochs:
        p = np.mean(e ** 2)
        rows.append(features(e + rng.normal(0, np.sqrt(p / 10 ** (snr / 10)), len(e))))
    return pd.DataFrame(rows)[cols]


def loso_proba(test_table):
    """Train on clean features of other participants; score `test_table` on the held-out one."""
    proba = np.zeros((len(y), 5))
    for tr, te in LeaveOneGroupOut().split(feat, y, subj):
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(feat.iloc[tr][cols], y[tr])
        p = m.predict_proba(test_table.iloc[te][cols])
        for k, c in enumerate(m.classes_):
            proba[te, stages.index(c)] = p[:, k]
    return proba


def persistent(mask, k):
    """True where the previous k-1 epochs and this one all satisfy `mask`, within a participant."""
    out = np.zeros_like(mask)
    for s in np.unique(subj):
        idx = np.where(subj == s)[0]
        m = mask[idx].astype(int)
        run = np.convolve(m, np.ones(k, dtype=int))[:len(m)]
        out[idx] = run >= k
    return out


# Acoustic uncertainty: brain RMS^2 samples from the Monte Carlo study (relative units)
mc = pd.read_csv("uq_acoustic_0.125mm.csv")
rms2 = mc["brain_rms_pa"].to_numpy() ** 2
acoustic_cv = rms2.std(ddof=1) / rms2.mean()

conditions = {"Clean": loso_proba(feat[cols])}
for snr in (10, 5):
    conditions[f"{snr} dB noise"] = loso_proba(noisy(snr))
    print(f"scored {snr} dB", flush=True)

rows = []
for target in ("N2", "N3"):
    k_t = stages.index(target)
    for cname, proba in conditions.items():
        pred = np.array(stages)[proba.argmax(1)]
        conf = proba.max(1)
        for pname, mask in {
            "argmax = target": pred == target,
            "argmax = target, conf >= 0.7": (pred == target) & (conf >= 0.7),
            "argmax = target, conf >= 0.9": (pred == target) & (conf >= 0.9),
            "argmax = target, 3 consecutive epochs": persistent(pred == target, 3),
            "conf >= 0.7, 3 consecutive epochs": persistent((pred == target) & (conf >= 0.7), 3),
        }.items():
            n_trig = int(mask.sum())
            true_t = y == target
            per_sub = []
            for s in np.unique(subj):
                m = subj == s
                t = mask & m
                per_sub.append(true_t[t].mean() if t.any() else np.nan)
            per_sub = np.array(per_sub)
            rows.append({
                "target": target, "eeg_condition": cname, "policy": pname,
                "triggers": n_trig,
                "trigger_frac": n_trig / len(y),
                "precision": true_t[mask].mean() if n_trig else np.nan,
                "recall": mask[true_t].mean(),
                "mistimed_in_wake": (y[mask] == "Wake").mean() if n_trig else np.nan,
                "precision_worst_subject": np.nanmin(per_sub) if np.isfinite(per_sub).any() else np.nan,
                "precision_best_subject": np.nanmax(per_sub) if np.isfinite(per_sub).any() else np.nan,
                # spread of per-participant mistimed share vs spread from acoustics
                "mistimed_cv_across_subjects": (np.nanstd(1 - per_sub) / np.nanmean(1 - per_sub)
                                                if np.isfinite(per_sub).sum() > 1 and np.nanmean(1 - per_sub) > 0 else np.nan),
                "acoustic_index_cv": acoustic_cv,
            })

res = pd.DataFrame(rows)
res.to_csv("integration_results.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
show = ["target", "eeg_condition", "policy", "triggers", "trigger_frac", "precision", "recall",
        "mistimed_in_wake", "precision_worst_subject", "precision_best_subject"]
print(res[show].round(3).to_string(index=False))
print(f"\nAcoustic relative-index CV (from Monte Carlo): {acoustic_cv:.3f}")
print("Per-participant spread of mistimed share (CV), examples:")
print(res[res.policy == "argmax = target"][["target", "eeg_condition", "mistimed_cv_across_subjects"]].round(3).to_string(index=False))
