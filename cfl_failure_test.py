import csv

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

from sensitivity import build_head

# Empirical stability test: same virtual head, same 10 us of simulated time,
# only the time step (and so the CFL number) changes.
DX = 0.25e-3
T_END = 10e-6
FREQ = 1e6

n, sound_speed, density, skull, brain = build_head(DX)
c_max = float(sound_speed.max())

rows = []
for cfl_target in [0.1, 0.224, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
    dt = cfl_target * DX / c_max
    nt = int(round(T_END / dt)) + 1
    t = np.arange(nt) * dt

    kgrid = kWaveGrid([n, n], [DX, DX])
    kgrid.setTime(nt, dt)

    source = kSource()
    source.p_mask = np.zeros((n, n), dtype=bool)
    i, points = round(4e-3 / DX), round(1e-3 / DX)
    j0 = round(16e-3 / DX) - points // 2
    source.p_mask[i, j0:j0 + points] = True
    source.p = np.sin(2 * np.pi * FREQ * t)

    sensor = kSensor()
    sensor.mask = brain

    try:
        data = kspaceFirstOrder(kgrid, kWaveMedium(sound_speed=sound_speed, density=density),
                                source, sensor, backend="python")
        p = np.asarray(data["p"])
        finite = bool(np.isfinite(p).all())
        peak = float(np.nanmax(np.abs(p))) if finite else float("inf")
        status = "ok" if finite and peak < 10 else ("blow-up" if finite else "non-finite")
    except Exception as e:  # k-Wave may refuse or overflow
        peak, status = float("nan"), f"error: {type(e).__name__}"

    print(f"CFL {c_max * dt / DX:.3f} (dt {dt * 1e9:6.2f} ns): peak |p| {peak:.4g} Pa -> {status}", flush=True)
    rows.append([c_max * dt / DX, dt, peak, status])

with open("cfl_failure_test.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cfl", "dt_s", "peak_abs_pressure_pa", "status"])
    w.writerows(rows)
