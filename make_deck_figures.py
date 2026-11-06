"""Large-text versions of the result charts for the Milestone 4 slides."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INK, MUTED, GRID, BG = "#12233F", "#44546A", "#DEDDD6", "#F7F6F1"
BLUE, ORANGE = "#2A78D6", "#EB6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 16, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})
stages = ["Wake", "N1", "N2", "N3", "REM"]
os.makedirs("deck_figs", exist_ok=True)

# Blind-test confusion matrix, from the blind_test.py output (rows = true stage)
cm = np.array([[5059, 555, 279, 24, 365], [37, 69, 42, 0, 41], [10, 14, 944, 21, 38],
               [9, 0, 122, 302, 0], [13, 33, 72, 0, 250]])
norm = cm / cm.sum(1, keepdims=True)
fig, a = plt.subplots(figsize=(6.6, 5.6))
a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        a.text(j, i, f"{norm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=16,
               color="white" if norm[i, j] > 0.55 else INK)
a.set_xticks(range(5), stages)
a.set_yticks(range(5), stages)
a.set_xlabel("Predicted stage")
a.set_ylabel("True stage (expert label)")
a.tick_params(length=0)
for s in a.spines.values():
    s.set_visible(False)
fig.tight_layout(pad=0.4)
fig.savefig("deck_figs/confusion.png")

# Noise: random forest, mean of 3 noise seeds
rf = pd.read_csv("robustness_comparison.csv")
rf = rf[rf.model == "RF (all)"]
fig, a = plt.subplots(figsize=(8.8, 5.6))
xs = list(range(len(rf)))
a.plot(xs, rf.mean_conf, color=ORANGE, lw=4, marker="o", ms=10)
a.plot(xs, rf.macro_f1, color=BLUE, lw=4, marker="o", ms=10)
a.text(0, rf.mean_conf.iloc[0] + 0.05, f"Confidence: {rf.mean_conf.iloc[0]:.2f}", fontsize=17, fontweight="bold")
a.text(4.05, rf.mean_conf.iloc[-1] + 0.05, f"{rf.mean_conf.iloc[-1]:.2f}", fontsize=17, ha="right", fontweight="bold")
a.text(0, rf.macro_f1.iloc[0] - 0.09, f"Macro-F1: {rf.macro_f1.iloc[0]:.2f}", fontsize=17, fontweight="bold")
a.text(4.05, rf.macro_f1.iloc[-1] + 0.05, f"{rf.macro_f1.iloc[-1]:.2f}", fontsize=17, ha="right", fontweight="bold")
a.set_xticks(xs, ["Clean", "20 dB", "10 dB", "5 dB", "0 dB"])
a.set_xlabel("More noise added to the EEG →")
a.set_ylim(0, 1)
a.grid(axis="y", color=GRID, lw=0.8)
a.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fig.savefig("deck_figs/noise.png")

# Focal pressure by model
s = pd.read_csv("lifu_scenarios.csv").set_index("name")
order = ["water", "brain only, no skull", "single-layer skull, absorption", "3-layer skull, no absorption", "3-layer skull, absorption"]
labels = ["Water", "Brain only", "1-layer skull + abs.", "3-layer skull", "3-layer skull + abs."]
v = s.loc[order, "relative_to_water"].to_numpy()
fig, a = plt.subplots(figsize=(7.0, 4.2))
ypos = np.arange(len(v))[::-1]
a.barh(ypos, v, height=0.7, color=[BLUE] * 4 + [ORANGE])
for yi, vi in zip(ypos, v):
    a.text(vi + 0.02, yi, f"{vi:.2f}", va="center", fontsize=17, fontweight="bold")
a.set_yticks(ypos, labels)
a.set_xlim(0, 1.2)
a.set_xticks([0, 0.5, 1])
a.set_xlabel("Focal pressure relative to water")
a.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.4)
fig.savefig("deck_figs/focal_pressure.png")

# What drives the uncertainty
m = pd.read_csv("lifu_mc.csv")
inputs = {"bone_absorption_factor": "Bone absorption", "skull_thickness_factor": "Skull thickness",
          "bone_speed_factor": "Bone speed", "bone_density_factor": "Bone density", "brain_speed": "Brain speed"}
rho = {lab: spearmanr(m[k], m.relative_to_water)[0] for k, lab in inputs.items()}
fig, a = plt.subplots(figsize=(7.4, 4.8))
ypos = np.arange(len(rho))[::-1]
a.barh(ypos, list(rho.values()), height=0.7, color=[ORANGE, ORANGE, BLUE, BLUE, BLUE])
a.axvline(0, color=MUTED, lw=1)
for yi, r in zip(ypos, rho.values()):
    a.text(r - 0.03 if r < 0 else r + 0.03, yi, f"{r:+.2f}", va="center", ha="right" if r < 0 else "left", fontsize=17, fontweight="bold")
a.set_yticks(ypos, list(rho.keys()))
a.set_xlim(-1.15, 0.55)
a.set_xticks([-1, -0.5, 0, 0.5])
a.set_xlabel("Rank correlation with focal pressure")
a.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.4)
fig.savefig("deck_figs/drivers.png")
print("saved deck_figs")
