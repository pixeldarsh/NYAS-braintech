import csv
import sys

import numpy as np
from scipy.stats import qmc, spearmanr

from sensitivity import build_head, run

# Monte Carlo propagation of tissue-property uncertainty to brain-region RMS pressure.
# Latin hypercube sampling over plausible ranges (assumed bounds, not fitted to data).
RANGES = {
    "skull_speed": (2400.0, 3200.0),     # m/s
    "skull_density": (1700.0, 2100.0),   # kg/m^3
    "brain_speed": (1500.0, 1580.0),     # m/s
    "brain_density": (1020.0, 1060.0),   # kg/m^3
}

dx_mm = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 24
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 2
dx = dx_mm * 1e-3

n, c0, d0, skull, brain = build_head(dx)
keep = np.zeros((n, n), dtype=bool)
keep[::stride, ::stride] = True
brain_sensor = brain & keep   # same region, uniformly thinned

names = list(RANGES)
lo = np.array([RANGES[k][0] for k in names])
hi = np.array([RANGES[k][1] for k in names])
samples = qmc.scale(qmc.LatinHypercube(d=len(names), seed=0).random(n_samples), lo, hi)

rows = []
for i, s in enumerate(samples, 1):
    p = dict(zip(names, s))
    c, d = c0.copy(), d0.copy()
    c[skull], d[skull] = p["skull_speed"], p["skull_density"]
    c[brain], d[brain] = p["brain_speed"], p["brain_density"]
    rms, peak = run(dx, c, d, brain_sensor, n)
    rows.append([*s, rms, peak])
    print(f"[{i}/{n_samples}] " + " ".join(f"{k}={p[k]:.0f}" for k in names)
          + f" -> RMS {rms:.5f}", flush=True)

arr = np.array(rows)
rms = arr[:, len(names)]

with open(f"uq_acoustic_{dx_mm}mm.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([*names, "brain_rms_pa", "brain_peak_pa"])
    w.writerows(rows)

print(f"\n=== Brain RMS over {n_samples} samples at dx={dx_mm} mm ===")
print(f"mean {rms.mean():.5f}  sd {rms.std(ddof=1):.5f}  CV {rms.std(ddof=1) / rms.mean() * 100:.1f}%")
print(f"min {rms.min():.5f}  median {np.median(rms):.5f}  max {rms.max():.5f}")
print(f"5-95% interval: {np.percentile(rms, 5):.5f} - {np.percentile(rms, 95):.5f}")

print("\nSpearman rank correlation with RMS (parameter importance):")
for k, name in enumerate(names):
    rho, pval = spearmanr(arr[:, k], rms)
    print(f"  {name:14s} rho {rho:+.2f}  (p={pval:.3f})")

# Standardised linear regression: share of variance explained by each parameter
Z = (arr[:, :len(names)] - arr[:, :len(names)].mean(0)) / arr[:, :len(names)].std(0)
yz = (rms - rms.mean()) / rms.std()
beta, *_ = np.linalg.lstsq(Z, yz, rcond=None)
r2 = 1 - np.sum((yz - Z @ beta) ** 2) / np.sum(yz ** 2)
print(f"\nLinear fit R^2 = {r2:.2f}; standardised coefficients:")
for name, b in zip(names, beta):
    print(f"  {name:14s} {b:+.2f}")
