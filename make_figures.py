"""Figures for the Milestone 3 report. All values come from this project's saved results."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 300, "figure.facecolor": "white",
})
stages = ["Wake", "N1", "N2", "N3", "REM"]

# Figure 1: EEG
fig, ax = plt.subplots(1, 2, figsize=(4.4, 2.0))

# (a) Blind-test confusion matrix, from the blind_test.py output (rows = true stage)
cm = np.array([[5059, 555, 279, 24, 365], [37, 69, 42, 0, 41], [10, 14, 944, 21, 38],
               [9, 0, 122, 302, 0], [13, 33, 72, 0, 250]])
norm = cm / cm.sum(1, keepdims=True)
a = ax[0]
a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        a.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
               color="white" if norm[i, j] > 0.55 else INK)
a.set_xticks(range(5), stages, fontsize=6)
a.set_yticks(range(5), stages, fontsize=6)
a.set_xlabel("Predicted stage")
a.set_ylabel("True stage")
a.tick_params(length=0)
for s in a.spines.values():
    s.set_visible(False)
a.set_title("a  Blind test (% of true stage)")

# (b) Noise: mean over 3 noise seeds, random forest with all features
rob = pd.read_csv("robustness_comparison.csv")
rf = rob[rob.model == "RF (all)"]
a = ax[1]
xs = range(len(rf))
a.plot(xs, rf.mean_conf, color=ORANGE, lw=2, marker="o", ms=4)
a.plot(xs, rf.macro_f1, color=BLUE, lw=2, marker="o", ms=4)
a.text(4.05, rf.mean_conf.iloc[-1] + 0.03, "Mean confidence", ha="right", va="bottom", fontsize=7, color=INK)
a.text(4.05, rf.macro_f1.iloc[-1] - 0.05, "Macro-F1", ha="right", va="top", fontsize=7, color=INK)
a.set_xticks(list(xs), [s.replace(" dB", "") if s != "Clean" else "Clean" for s in rf.snr])
a.set_xlabel("Added noise, SNR (dB)")
a.set_ylim(0, 1)
a.grid(axis="y", color=GRID, lw=0.6)
a.set_axisbelow(True)
a.set_title("b  Effect of EEG noise")

fig.tight_layout(pad=0.5, w_pad=1.2)
fig.savefig("fig_eeg_results.png")

# Figure 2: acoustics
fig, ax = plt.subplots(1, 2, figsize=(4.4, 1.85), gridspec_kw={"width_ratios": [1.65, 1]})

# (a) Simulated field, normalised to the water focal peak
d = np.load("lifu_fields.npz")
x = (d["x"] - d["apex"]) * 1e3
y = d["y"] * 1e3
water_peak = d["water_p"][x > 30, :].max()
field = d["layered_p"] / water_peak
a = ax[0]
im = a.imshow(field.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="Blues", vmin=0, vmax=0.6, aspect="equal")
th = np.linspace(0, 2 * np.pi, 800)
for r in (79, 75, 68.5):
    a.plot(105 + r * np.cos(th), r * np.sin(th), color=INK, lw=0.5)
phi = np.linspace(-np.arcsin(0.5), np.arcsin(0.5), 100)
a.plot(64 - 64 * np.cos(phi), 64 * np.sin(phi), color=ORANGE, lw=2)
a.plot([64], [0], marker="+", color=ORANGE, ms=7, mew=1.3)
a.text(12, 29.5, "Source", color=ORANGE, fontsize=6, fontweight="bold")
a.text(62, 3.5, "Target", color=ORANGE, fontsize=6, ha="right", fontweight="bold")
a.text(92, 27, "Brain", fontsize=6)
a.text(45, -31, "Skin + skull", fontsize=6)
a.set_xlim(x[0], x[-1])
a.set_ylim(y[0], y[-1])
a.set_xlabel("Distance from source (mm)")
a.set_ylabel("Lateral (mm)")
cb = fig.colorbar(im, ax=a, fraction=0.028, pad=0.02)
cb.set_label("Pressure / water focus", fontsize=6)
cb.ax.tick_params(labelsize=6)
cb.outline.set_visible(False)
a.set_title("a  Simulated beam through the head")

# (b) Focal pressure by model
s = pd.read_csv("lifu_scenarios.csv").set_index("name")
order = ["water", "brain only, no skull", "single-layer skull, no absorption", "single-layer skull, absorption",
         "3-layer skull, no absorption", "3-layer skull, absorption"]
labels = ["Water", "Brain only", "1-layer", "1-layer + abs.", "3-layer", "3-layer + abs."]
v = s.loc[order, "relative_to_water"].to_numpy()
a = ax[1]
ypos = np.arange(len(v))[::-1]
a.barh(ypos, v, height=0.68, color=[BLUE] * 5 + [ORANGE])
for yi, vi in zip(ypos, v):
    a.text(vi + 0.02, yi, f"{vi:.2f}", va="center", fontsize=7)
a.set_yticks(ypos, labels, fontsize=6.5)
a.set_xlim(0, 1.3)
a.set_xticks([0, 0.5, 1])
a.set_xlabel("Focal pressure / water")
a.tick_params(axis="y", length=0)
a.set_title("b  Focal pressure")

fig.tight_layout(pad=0.5, w_pad=1.0)
fig.savefig("fig_acoustic_results.png")
print("saved fig_eeg_results.png and fig_acoustic_results.png")
