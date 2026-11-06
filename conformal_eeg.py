import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Split conformal prediction with participant-wise calibration:
# for each held-out test participant, one *other* participant calibrates the threshold
# and the remaining three train the model. Coverage is then checked on the test participant.
ALPHA = 0.10
stages = ["Wake", "N1", "N2", "N3", "REM"]

df = pd.read_csv("eeg_features.csv")
abs_cols = [c for c in df.columns if c.endswith("_absolute")]
rel_cols = [c for c in df.columns if c.endswith("_relative")]
X = df[abs_cols + rel_cols].copy()
X[abs_cols] = np.log10(X[abs_cols] + 1e-20)
y = df["sleep_stage"].to_numpy()
g = df["subject"].to_numpy()
subjects = sorted(np.unique(g))


def make(kind):
    if kind == "RF":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))


def proba_full(model, Xs):
    p = model.predict_proba(Xs)
    out = np.zeros((len(Xs), 5))
    for k, c in enumerate(model.classes_):
        out[:, stages.index(c)] = p[:, k]
    return out


rows = []
for kind in ["RF", "LR"]:
    for test_s in subjects:
        cal_s = [s for s in subjects if s != test_s]
        # average over each choice of calibration participant for stability
        for cs in cal_s:
            train = (g != test_s) & (g != cs)
            cal, test = g == cs, g == test_s
            model = make(kind).fit(X[train], y[train])

            # nonconformity: 1 - probability of the true class
            pc = proba_full(model, X[cal])
            yc = np.array([stages.index(v) for v in y[cal]])
            scores = 1 - pc[np.arange(len(yc)), yc]
            nc = len(scores)
            q = np.quantile(scores, min(1.0, np.ceil((nc + 1) * (1 - ALPHA)) / nc), method="higher")

            pt = proba_full(model, X[test])
            sets = (1 - pt) <= q
            yt = np.array([stages.index(v) for v in y[test]])
            covered = sets[np.arange(len(yt)), yt]
            size = sets.sum(1)
            top1 = pt.argmax(1) == yt

            # Mondrian (class-conditional) variant: one threshold per true class
            qc = np.array([
                np.quantile(scores[yc == k],
                            min(1.0, np.ceil(((yc == k).sum() + 1) * (1 - ALPHA)) / max((yc == k).sum(), 1)),
                            method="higher") if (yc == k).any() else 1.0
                for k in range(5)])
            msets = (1 - pt) <= qc[None, :]
            mcov = msets[np.arange(len(yt)), yt]
            msize = msets.sum(1)

            rec = {"model": kind, "test": test_s, "cal": cs,
                   "m_coverage": mcov.mean(), "m_mean_set_size": msize.mean(),
                   "m_cov_N1": mcov[yt == 1].mean() if (yt == 1).any() else np.nan,
                   "m_cov_N3": mcov[yt == 3].mean() if (yt == 3).any() else np.nan,
                   "m_cov_REM": mcov[yt == 4].mean() if (yt == 4).any() else np.nan,
                   "coverage": covered.mean(), "mean_set_size": size.mean(),
                   "empty_sets": (size == 0).mean(), "singleton_acc": top1[size == 1].mean() if (size == 1).any() else np.nan,
                   "singleton_frac": (size == 1).mean()}
            for k, s in enumerate(stages):
                m = yt == k
                rec[f"cov_{s}"] = covered[m].mean() if m.any() else np.nan
                rec[f"size_{s}"] = size[m].mean() if m.any() else np.nan
            rows.append(rec)
    print(f"{kind} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("conformal_results_raw.csv", index=False)

pd.set_option("display.width", 220)
cols = ["coverage", "mean_set_size", "singleton_frac", "singleton_acc",
        *[f"cov_{s}" for s in stages], *[f"size_{s}" for s in stages]]
print(f"\n===== Conformal prediction, target coverage {1 - ALPHA:.0%} (mean over test/cal pairs) =====")
print(res.groupby("model")[cols].mean().round(3).T.to_string())

print("\n===== Class-conditional (Mondrian) conformal =====")
print(res.groupby("model")[["m_coverage", "m_mean_set_size", "m_cov_N1", "m_cov_N3", "m_cov_REM"]].mean().round(3).T.to_string())

print("\nCoverage by held-out participant (mean over calibration choices):")
print(res.pivot_table(index="test", columns="model", values="coverage").round(3).to_string())
print("\nWorst single test/cal pair coverage:", res.groupby("model").coverage.min().round(3).to_dict())
