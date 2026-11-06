import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.special import xlogy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

epochs = np.load("eeg_epochs.npy")
feat = pd.read_csv("eeg_features.csv")
y = feat["sleep_stage"].to_numpy()
groups = feat["subject"].to_numpy()
assert len(epochs) == len(y)

bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "sigma": (13, 16), "beta": (16, 30)}
abs_cols = [f"{b}_absolute" for b in bands]
rel_cols = [f"{b}_relative" for b in bands]
stages = ["Wake", "N1", "N2", "N3", "REM"]
SNRS = [None, 20, 10, 5, 0]
SEEDS = [0, 1, 2]


def features(epoch):
    f, psd = welch(epoch, fs=100, nperseg=1000)
    # Same definition as extractfeatures.py (relative power is over 0.5-30 Hz)
    tot_mask = (f >= 0.5) & (f <= 30)
    total = np.trapezoid(psd[tot_mask], f[tot_mask])
    out = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f <= hi)
        a = np.trapezoid(psd[m], f[m])
        out[f"{name}_absolute"] = a
        out[f"{name}_relative"] = a / total if total > 0 else 0
    return out


def add_noise(sig, snr_db, rng):
    p = np.mean(sig ** 2)
    return sig + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(sig))


def prep(df):
    X = df[abs_cols + rel_cols].copy()
    X[abs_cols] = np.log10(X[abs_cols] + 1e-20)
    return X


# Noisy feature tables, computed once (noise does not depend on the fold)
tables = {(None, 0): feat[abs_cols + rel_cols]}
for snr in SNRS[1:]:
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tables[(snr, seed)] = pd.DataFrame([features(add_noise(e, snr, rng)) for e in epochs])
    print(f"noisy features done: {snr} dB", flush=True)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return sum(
        (idx == b).mean() * abs(conf[idx == b].mean() - correct[idx == b].mean())
        for b in range(bins) if (idx == b).any()
    )


def make(kind):
    if kind.startswith("RF"):
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


MODELS = {
    "RF (all)": ("RF", abs_cols + rel_cols),
    "RF (relative only)": ("RF", rel_cols),
    "LR (all)": ("LR", abs_cols + rel_cols),
    "LR (relative only)": ("LR", rel_cols),
}

# store[(model, snr, seed)] -> predictions / probabilities over all epochs
store = {}
for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(epochs, y, groups), 1):
    for mname, (kind, cols) in MODELS.items():
        model = make(kind).fit(prep(feat).iloc[tr][cols], y[tr])
        for key, tab in tables.items():
            Xte = prep(tab).iloc[te][cols]
            p = model.predict_proba(Xte)
            full = np.zeros((len(te), 5))
            for k, c in enumerate(model.classes_):
                full[:, stages.index(c)] = p[:, k]
            st = store.setdefault((mname, *key), {"idx": [], "proba": []})
            st["idx"].append(te)
            st["proba"].append(full)
    print(f"fold {fold} done", flush=True)

rows = []
for (mname, snr, seed), st in store.items():
    idx = np.concatenate(st["idx"])
    proba = np.vstack(st["proba"])
    order = np.argsort(idx)
    proba = proba[order]
    pred = np.array(stages)[proba.argmax(1)]
    correct = pred == y
    conf = proba.max(1)
    rows.append({
        "model": mname, "snr": "Clean" if snr is None else f"{snr} dB", "seed": seed,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=stages, average="macro", zero_division=0),
        "mean_conf": conf.mean(),
        "ece": ece(conf, correct),
        "auc_err_conf": roc_auc_score(correct, conf) if 0 < correct.mean() < 1 else np.nan,
        "acc_top50pct_conf": correct[conf >= np.quantile(conf, 0.5)].mean(),
    })

res = pd.DataFrame(rows)
res.to_csv("robustness_comparison_raw.csv", index=False)
order = ["Clean", "20 dB", "10 dB", "5 dB", "0 dB"]
g = res.groupby(["model", "snr"])[["accuracy", "macro_f1", "mean_conf", "ece", "auc_err_conf", "acc_top50pct_conf"]].mean().round(3)
g = g.reindex(pd.MultiIndex.from_product([MODELS, order], names=["model", "snr"]))
pd.set_option("display.width", 200)
print("\n===== Robustness to Gaussian noise (pooled LOSO, mean over 3 noise seeds) =====")
print(g.to_string())
g.to_csv("robustness_comparison.csv")
