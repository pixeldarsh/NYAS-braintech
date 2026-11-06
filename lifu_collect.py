"""Collect every finished focused-beam simulation in lifu_cache/ into one CSV.

Safe to run at any time, including while simulations are still running.
"""
import csv
import glob
import json
import os

from scipy.stats import qmc

from lifu_experiments import RANGES

names = list(RANGES)
lo, hi = [RANGES[k][0] for k in names], [RANGES[k][1] for k in names]
samples = qmc.scale(qmc.LatinHypercube(d=len(names), seed=1).random(32), lo, hi)

results = {}
for path in glob.glob("lifu_cache/*.json"):
    with open(path) as f:
        results[os.path.basename(path)[:-5]] = json.load(f)

water = {k.rsplit("_", 1)[1]: v["peak"] for k, v in results.items() if k.startswith("run_water")}
rows = []
for key, m in sorted(results.items()):
    row = {"run": key, **m}
    if key.startswith("mc_"):
        row.update(zip(names, map(float, samples[int(key[3:])])))
        reference = water.get("0.25mm")
    else:
        reference = water.get(key.rsplit("_", 1)[1])
    row["relative_to_water"] = m["peak"] / reference if reference else ""
    rows.append(row)

fields = ["run", "relative_to_water", "peak", "focus_x_from_apex_mm", "focus_y_mm", "shift_from_target_mm",
          "axial_6db_mm", "lateral_6db_mm", *names]
tmp = "lifu_finished_so_far.csv.tmp"
with open(tmp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
os.replace(tmp, "lifu_finished_so_far.csv")
print(f"{len(rows)} finished simulations saved to lifu_finished_so_far.csv")
