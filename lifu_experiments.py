"""Focused-beam experiments on the 2D head model (simulation only, uncalibrated).

scenarios - effect of skull layers and absorption at 0.25 mm, plus a steady-state check
grid      - convergence at 0.5, 0.25 and 0.125 mm
mc        - Monte Carlo over tissue properties and skull thickness at 0.25 mm

Pressures are reported relative to the same source in water at the same grid spacing.
Monte Carlo ranges are assumed spreads around the published representative values; the
ITRUSST practical guide notes that skull attenuation uncertainty is considerable.
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc, spearmanr

from lifu_model import THICKNESS, TISSUES, focus_metrics, run

CACHE_DIR = "lifu_cache"
WORKERS = int(os.environ.get("LIFU_WORKERS", "4"))


def cached(key, compute):
    """Each finished simulation is saved on its own, so a restart repeats only unfinished ones."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    result = compute()
    with open(path + ".tmp", "w") as f:
        json.dump(result, f)
    os.replace(path + ".tmp", path)
    return result


def one(job):
    name, dx, kw = job
    key = f"run_{kw.get('scenario', 'layered')}_abs{kw.get('absorb', True)}_{dx * 1e3}mm"
    m = cached(key, lambda: focus_metrics(run(dx, **kw)))
    print(f"done: {name} dx={dx * 1e3} mm peak={m['peak']:.4g}", flush=True)
    return {"name": name, "dx_mm": dx * 1e3, **m}


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def show(rows, keys):
    for r in rows:
        print(" | ".join(f"{k}={r[k]:.3f}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in keys))


def scenarios():
    dx = 0.25e-3
    jobs = [
        ("water", dx, dict(scenario="water")),
        ("brain only, no skull", dx, dict(scenario="homogeneous_brain")),
        ("single-layer skull, no absorption", dx, dict(scenario="single_layer", absorb=False)),
        ("single-layer skull, absorption", dx, dict(scenario="single_layer")),
        ("3-layer skull, no absorption", dx, dict(scenario="layered", absorb=False)),
        ("3-layer skull, absorption", dx, dict(scenario="layered")),
    ]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, jobs))
    water = rows[0]["peak"]
    for r in rows:
        r["relative_to_water"] = r["peak"] / water
    write("lifu_scenarios.csv", rows)
    show(rows, ["name", "relative_to_water", "focus_x_from_apex_mm", "shift_from_target_mm", "axial_6db_mm", "lateral_6db_mm"])


def grid():
    jobs = [(f"{s} {d}", d * 1e-3, dict(scenario=s)) for d in (0.5, 0.25, 0.125) for s in ("water", "layered")]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(one, jobs))
    for r in rows:
        water = next(q["peak"] for q in rows if q["dx_mm"] == r["dx_mm"] and q["name"].startswith("water"))
        r["relative_to_water"] = r["peak"] / water
    write("lifu_grid.csv", rows)
    show(rows, ["name", "relative_to_water", "focus_x_from_apex_mm", "shift_from_target_mm", "axial_6db_mm", "lateral_6db_mm"])


RANGES = {
    "bone_speed_factor": (0.9, 1.1),
    "bone_density_factor": (0.9, 1.1),
    "bone_absorption_factor": (0.5, 2.0),
    "skull_thickness_factor": (0.75, 1.25),
    "brain_speed": (1513.0, 1607.0),
}


def mc_job(args):
    i, p = args
    t = dict(TISSUES)
    for bone in ("cortical", "trabecular"):
        c, r, a = TISSUES[bone]
        t[bone] = (c * p["bone_speed_factor"], r * p["bone_density_factor"], a * p["bone_absorption_factor"])
    c, r, a = TISSUES["brain"]
    t["brain"] = (p["brain_speed"], r, a)
    th = {k: (v * p["skull_thickness_factor"] if k != "skin" else v) for k, v in THICKNESS.items()}
    try:
        m = cached(f"mc_{i}", lambda: focus_metrics(run(0.25e-3, tissues=t, thickness=th, scenario="layered")))
    except FloatingPointError as e:
        print(f"sample {i} UNSTABLE: {e}", flush=True)
        return None
    print(f"sample {i} done", flush=True)
    return {**p, **m}


def mc(n=32):
    names = list(RANGES)
    lo, hi = np.array([RANGES[k][0] for k in names]), np.array([RANGES[k][1] for k in names])
    samples = qmc.scale(qmc.LatinHypercube(d=len(names), seed=1).random(n), lo, hi)
    jobs = [(i, dict(zip(names, map(float, s)))) for i, s in enumerate(samples)]
    water = one(("water", 0.25e-3, dict(scenario="water")))["peak"]
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(mc_job, jobs))
    print(f"\nUnstable samples excluded: {sum(r is None for r in rows)} of {len(rows)}")
    rows = [r for r in rows if r is not None]
    for r in rows:
        r["relative_to_water"] = r["peak"] / water
    write("lifu_mc.csv", rows)

    outputs = ["relative_to_water", "focus_x_from_apex_mm", "shift_from_target_mm", "axial_6db_mm", "lateral_6db_mm"]
    for out in outputs:
        v = np.array([r[out] for r in rows])
        print(f"\n{out}: mean {v.mean():.3f} sd {v.std(ddof=1):.3f} CV {v.std(ddof=1) / abs(v.mean()) * 100:.1f}% "
              f"min {v.min():.3f} max {v.max():.3f} 5-95% {np.percentile(v, 5):.3f} to {np.percentile(v, 95):.3f}")
        for k in names:
            rho, pv = spearmanr([r[k] for r in rows], v)
            print(f"  {k:24s} rho {rho:+.2f} (p={pv:.3f})")


if __name__ == "__main__":
    {"scenarios": scenarios, "grid": grid, "mc": mc}[sys.argv[1]]()
