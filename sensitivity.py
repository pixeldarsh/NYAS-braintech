import csv
import sys

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder


# Same time step as grid.py, so results are comparable with the convergence study.
DT = 5e-9
CFL_LIMIT = 0.3
FREQUENCY = 1e6
T_END = 20e-6
WINDOW_START = 5e-6


def build_head(dx):
    """Virtual head on the fixed 32 x 32 mm domain used in grid.py."""
    n = int(round(32e-3 / dx))
    x = np.arange(n) * dx * 1000
    X, Y = np.meshgrid(x, x, indexing="ij")

    R = np.sqrt(((X - 19.0) / 11.0) ** 2 + ((Y - 16.0) / 13.0) ** 2)

    scalp = (R <= 1.0) & ~(R <= 0.95)
    skull = (R <= 0.95) & ~(R <= 0.80)
    csf = (R <= 0.80) & ~(R <= 0.75)
    brain = R <= 0.75

    sound_speed = np.full((n, n), 1480.0)
    density = np.full((n, n), 1000.0)
    sound_speed[scalp], density[scalp] = 1540, 1050
    sound_speed[skull], density[skull] = 2800, 1900
    sound_speed[csf], density[csf] = 1500, 1000
    sound_speed[brain], density[brain] = 1540, 1040

    return n, sound_speed, density, skull, brain


def run(dx, sound_speed, density, brain, n):
    """Simulate and return RMS / peak pressure over the brain, 5-20 us."""
    c_max = float(np.max(sound_speed))
    cfl = c_max * DT / dx
    if cfl > CFL_LIMIT:
        raise ValueError(f"CFL {cfl:.3f} exceeds {CFL_LIMIT} at dx={dx * 1000} mm")

    kgrid = kWaveGrid([n, n], [dx, dx])
    t_array = np.arange(0, T_END + DT / 2, DT)
    kgrid.setTime(len(t_array), DT)

    medium = kWaveMedium(sound_speed=sound_speed, density=density)

    # 1 mm source aperture centred at (4 mm, 16 mm)
    source = kSource()
    source.p_mask = np.zeros((n, n), dtype=bool)
    i = round(4e-3 / dx)
    points = round(1e-3 / dx)
    j0 = round(16e-3 / dx) - points // 2
    source.p_mask[i, j0:j0 + points] = True
    source.p = np.sin(2 * np.pi * FREQUENCY * t_array)

    # Record only inside the brain mask, not the whole grid
    sensor = kSensor()
    sensor.mask = brain

    data = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python")
    p = np.asarray(data["p"], dtype=np.float32)[:, int(WINDOW_START / DT):int(T_END / DT)]

    rms = float(np.sqrt(np.mean(p.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(p)))
    return rms, peak


def main(dx, stride=1, only=None):
    n, sound_speed, density, skull, brain = build_head(dx)
    if stride > 1:
        # Uniform subsample of the brain region (keeps memory manageable on fine grids)
        keep = np.zeros((n, n), dtype=bool)
        keep[::stride, ::stride] = True
        brain = brain & keep
    print(f"\n##### dx = {dx * 1000:.4f} mm, grid {n}x{n}, "
          f"brain points {int(brain.sum())}", flush=True)

    scenarios = []

    # Homogeneous control: nominal brain properties everywhere
    scenarios.append(("homogeneous", None, np.full((n, n), 1540.0), np.full((n, n), 1040.0)))
    scenarios.append(("virtual_head_baseline", None, sound_speed, density))

    for speed in (2400, 3200):
        c = sound_speed.copy()
        c[skull] = speed
        scenarios.append((f"skull_speed_{speed}", speed, c, density))

    for rho in (1700, 2100):
        d = density.copy()
        d[skull] = rho
        scenarios.append((f"skull_density_{rho}", rho, sound_speed, d))

    if only:
        scenarios = [s for s in scenarios if s[0] in only]

    results = {}
    for name, _, c, d in scenarios:
        rms, peak = run(dx, c, d, brain, n)
        results[name] = (rms, peak)
        print(f"{name:24s} RMS {rms:.6f} Pa | peak {peak:.6f} Pa", flush=True)

    base_rms, base_peak = results["virtual_head_baseline"]
    print("\n--- Relative to virtual-head baseline (skull 2800 m/s, 1900 kg/m^3) ---")
    rows = []
    for name, (rms, peak) in results.items():
        d_rms = (rms - base_rms) / base_rms * 100
        d_peak = (peak - base_peak) / base_peak * 100
        print(f"{name:24s} RMS {d_rms:+7.2f}% | peak {d_peak:+7.2f}%")
        rows.append([dx * 1000, name, rms, peak, d_rms, d_peak])
    return rows


if __name__ == "__main__":
    # Usage: sensitivity.py [dx_mm[:stride] ...] ; fine grids run a reduced scenario set
    args = sys.argv[1:] or ["0.25", "0.125"]
    all_rows = []
    out = "sensitivity_results.csv"
    for a in args:
        dx_mm, _, stride = a.partition(":")
        if stride:
            out = "sensitivity_results_fine.csv"
            all_rows += main(float(dx_mm) * 1e-3, int(stride),
                             only={"homogeneous", "virtual_head_baseline", "skull_speed_3200"})
        else:
            all_rows += main(float(dx_mm) * 1e-3)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx_mm", "scenario", "brain_rms_pa", "brain_peak_pa",
                    "rms_vs_baseline_pct", "peak_vs_baseline_pct"])
        w.writerows(all_rows)
    print(f"\nSaved {out}")
