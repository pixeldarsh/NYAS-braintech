"""Checks of the acoustic solver against references it does not depend on.

plate  - plane wave through bone layers, against an exact transfer-matrix solution
arc2d  - 2D focused field in water, against a numerically integrated analytic field
bowl3d - 3D bowl in water (Aubry et al. benchmark 1): -6 dB focal size against the published
         26.2 mm axial and 4.1 mm lateral, and the on-axis shape against O'Neil's solution
"""
import csv
import sys
import warnings

import numpy as np
from scipy.special import hankel1

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.mapgen import focused_bowl_oneil

from lifu_model import ALPHA_POWER, APERTURE, CFL, FREQ, ROC, TISSUES, alpha_coeff, focus_metrics, run, width_db6

warnings.simplefilter("ignore", FutureWarning)
OMEGA = 2 * np.pi * FREQ


def transfer_matrix(layers, c0, rho0, c_end, rho_end):
    """Exact pressure transmission amplitude for normal incidence through layers.

    layers: (thickness m, c, rho, alpha dB/cm at FREQ). Time dependence exp(+i w t).
    """
    k0 = OMEGA / c0
    Z0 = rho0 * c0
    M = np.eye(2, dtype=complex)
    for d, c, rho, a in layers:
        alpha_np = a * 100 / 8.685889638  # Np/m
        k = OMEGA / c - 1j * alpha_np
        Z = rho * OMEGA / k
        M = M @ np.array([[np.cos(k * d), 1j * Z * np.sin(k * d)],
                          [1j * np.sin(k * d) / Z, np.cos(k * d)]])
    Z_end = rho_end * c_end
    a, b, c_, d_ = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    T = 2 / (a + b / Z_end + Z0 * c_ + Z0 * d_ / Z_end)
    return abs(T)


def plate_sim(layers, c_end, rho_end, dx, f_ref=None):
    """1D plane-wave simulation; returns steady amplitude just past the layers."""
    total = sum(l[0] for l in layers)
    x_src, x_gap = 10e-3, 15e-3
    L = x_src + x_gap + total + 30e-3
    n = int(round(L / dx))
    n += n % 2
    c = np.full(n, 1500.0)
    rho = np.full(n, 1000.0)
    alpha = np.zeros(n)
    x = np.arange(n) * dx
    pos = x_src + x_gap
    for d, ci, ri, ai in layers:
        m = (x >= pos) & (x < pos + d)
        c[m], rho[m], alpha[m] = ci, ri, ai
        pos += d
    c[x >= pos], rho[x >= pos] = c_end, rho_end

    kgrid = kWaveGrid([n], [dx])
    dt = CFL * dx / c.max()
    # Long enough for layer reverberation to settle (absorption alone does not end it in water)
    nt = int(round(160e-6 / dt))
    kgrid.setTime(nt, dt)
    t = np.arange(nt) * dt

    source = kSource()
    mask = np.zeros(n, dtype=bool)
    mask[int(round(x_src / dx))] = True
    source.p_mask = mask
    source.p = (np.minimum(t * FREQ / 3, 1) * np.sin(OMEGA * t))[None, :]

    sens = np.zeros(n, dtype=bool)
    sens[int(round((pos + 10e-3) / dx))] = True
    sensor = kSensor(mask=sens)
    kwargs = dict(sound_speed=c, density=rho)
    if alpha.max() > 0:
        kwargs.update(alpha_coeff=alpha_coeff(alpha), alpha_power=ALPHA_POWER)
    out = kspaceFirstOrder(kgrid, kWaveMedium(**kwargs), source, sensor, backend="python", quiet=True)
    p = np.asarray(out["p"]).ravel()
    return np.abs(p[-int(round(4 / FREQ / dt)):]).max(), (c_end, rho_end, pos)


def check_plate():
    co, tr, sk, br = TISSUES["cortical"], TISSUES["trabecular"], TISSUES["skin"], TISSUES["brain"]
    cases = {
        "6.5 mm cortical, no absorption, water both sides": ([(6.5e-3, co[0], co[1], 0.0)], 1500.0, 1000.0),
        "6.5 mm cortical, absorbing, water both sides": ([(6.5e-3, co[0], co[1], co[2])], 1500.0, 1000.0),
        "skin + 3-layer skull, absorbing, into brain": (
            [(4e-3, sk[0], sk[1], sk[2]), (1.5e-3, co[0], co[1], co[2]), (4e-3, tr[0], tr[1], tr[2]),
             (1e-3, co[0], co[1], co[2])], br[0], br[1]),
    }
    rows = []
    for name, (layers, c_end, rho_end) in cases.items():
        analytic = transfer_matrix(layers, 1500.0, 1000.0, c_end, rho_end)
        for dx in (0.25e-3, 0.125e-3, 0.0625e-3):
            ref, _ = plate_sim([], c_end=1500.0, rho_end=1000.0, dx=dx)
            amp, _ = plate_sim(layers, c_end, rho_end, dx)
            sim = amp / ref
            err = (sim - analytic) / analytic * 100
            rows.append([name, dx * 1e3, analytic, sim, err])
            print(f"{name:52s} dx {dx * 1e3:.4f} mm  analytic {analytic:.4f}  simulated {sim:.4f}  error {err:+.2f}%", flush=True)
    with open("verify_plate.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "dx_mm", "analytic_transmission", "simulated_transmission", "error_pct"])
        w.writerows(rows)


def analytic_arc(xq, yq, x_apex):
    """2D line-source Green's function integrated over the arc (water)."""
    k = OMEGA / 1500.0
    phi_max = np.arcsin(APERTURE / (2 * ROC))
    phi = np.linspace(-phi_max, phi_max, 8001)
    fx = x_apex + ROC
    sx, sy = fx - ROC * np.cos(phi), ROC * np.sin(phi)
    d = np.sqrt((xq[..., None] - sx) ** 2 + (yq[..., None] - sy) ** 2)
    return np.abs(np.trapezoid(hankel1(0, k * d), phi, axis=-1))


def check_arc():
    rows = []
    for dx in (0.5e-3, 0.25e-3, 0.125e-3):
        res = run(dx, scenario="water")
        x, y, p = res["x"], res["y"], res["p"]
        X_apex = res["x_apex"]
        region = np.zeros_like(p, dtype=bool)
        region[(x > X_apex + 30e-3)[:, None] & (np.abs(y) < 25e-3)[None, :]] = True
        m = focus_metrics(res, region)

        # Analytic reference evaluated on the same axes
        j0 = np.argmin(np.abs(y))
        ax = analytic_arc(x, np.zeros_like(x), X_apex)
        i_a = int(np.argmax(np.where(x > X_apex + 30e-3, ax, -1)))
        lat = analytic_arc(np.full_like(y, x[i_a]), y, X_apex)
        j_a = int(np.argmax(lat))
        an = {"focus": (x[i_a] - X_apex) * 1e3,
              "axial": width_db6(ax, i_a, dx) * 1e3,
              "lateral": width_db6(lat, j_a, dx) * 1e3}

        # Field shape error across the focal region, each normalised to its own peak
        keep = (x > X_apex + 30e-3) & (x < x.max() - 5e-3)
        rms_err = np.sqrt(np.mean((p[keep, j0] / p[keep, j0].max() - ax[keep] / ax[keep].max()) ** 2)) * 100
        print(f"dx {dx * 1e3:.3f} mm | focus {m['focus_x_from_apex_mm']:.2f} vs analytic {an['focus']:.2f} mm | "
              f"axial {m['axial_6db_mm']:.2f} vs {an['axial']:.2f} mm | lateral {m['lateral_6db_mm']:.2f} vs {an['lateral']:.2f} mm | "
              f"on-axis shape RMS error {rms_err:.2f}%", flush=True)
        rows.append([dx * 1e3, m["focus_x_from_apex_mm"], an["focus"], m["axial_6db_mm"], an["axial"],
                     m["lateral_6db_mm"], an["lateral"], rms_err])
    with open("verify_arc2d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx_mm", "focus_sim_mm", "focus_analytic_mm", "axial_sim_mm", "axial_analytic_mm",
                    "lateral_sim_mm", "lateral_analytic_mm", "onaxis_shape_rms_error_pct"])
        w.writerows(rows)


def check_bowl(dx=0.5e-3):
    """Benchmark 1 geometry in 3D water, using a symmetric lateral domain of +/- 25 mm."""
    Lx, Ly = 110e-3, 50e-3
    nx, ny = int(round(Lx / dx)), int(round(Ly / dx))
    kgrid = kWaveGrid([nx, ny, ny], [dx, dx, dx])
    dt = 0.3 * dx / 1500.0
    nt = int(round(95e-6 / dt))
    kgrid.setTime(nt, dt)
    t = np.arange(nt) * dt
    x = np.asarray(kgrid.x_vec).ravel()
    y = np.asarray(kgrid.y_vec).ravel()
    apex = x.min() + 2e-3

    arr = kWaveArray()
    arr.add_bowl_element((apex, 0.0, 0.0), ROC, APERTURE, (apex + ROC, 0.0, 0.0))
    source = kSource()
    source.p_mask = arr.get_array_binary_mask(kgrid)
    source.p = arr.get_distributed_source_signal(
        kgrid, (np.minimum(t * FREQ / 3, 1) * np.sin(OMEGA * t))[None, :], order="C").astype(np.float32)
    print("3D source points", int(source.p_mask.sum()), "steps", nt, flush=True)

    # Record only the axis and one lateral plane to save memory
    jc = int(np.argmin(np.abs(y)))
    sens = np.zeros((nx, ny, ny), dtype=bool)
    sens[:, :, jc] = True
    sensor = kSensor(mask=sens, record=["p_max"])
    sensor.record_start_index = max(1, nt - int(round(2 / FREQ / dt)))
    out = kspaceFirstOrder(kgrid, kWaveMedium(sound_speed=1500.0, density=1000.0), source, sensor,
                           backend="python", dtype=np.float32, quiet=True)
    plane = np.asarray(out["p_max"]).reshape((nx, ny), order="C")
    axis = plane[:, jc]
    i = int(np.argmax(np.where(x > apex + 30e-3, axis, -1)))
    ax_w = width_db6(axis, i, dx) * 1e3
    lat_w = width_db6(plane[i, :], jc, dx) * 1e3

    # O'Neil on-axis solution; x measured from the bowl apex along the axis
    q = x - apex
    keep = (q > 10e-3) & (q < x.max() - apex - 3e-3)
    res = focused_bowl_oneil(ROC, APERTURE, 0.04, FREQ, 1500.0, 1000.0, axial_positions=q[keep])
    onaxis = np.abs(np.asarray(res[0] if isinstance(res, tuple) else res)).ravel()
    i_on = int(np.argmax(onaxis))
    on_w = width_db6(onaxis, i_on, dx) * 1e3
    sim_shape = axis[keep] / axis[keep].max()
    rms = np.sqrt(np.mean((sim_shape - onaxis / onaxis.max()) ** 2)) * 100

    print(f"3D bowl dx {dx * 1e3:.2f} mm: focus at {q[i] * 1e3:.2f} mm (O'Neil {q[keep][i_on] * 1e3:.2f} mm)")
    print(f"  axial -6 dB {ax_w:.2f} mm (published 26.2; O'Neil {on_w:.2f})")
    print(f"  lateral -6 dB {lat_w:.2f} mm (published 4.1)")
    print(f"  on-axis shape RMS error vs O'Neil {rms:.2f}%", flush=True)
    with open("verify_bowl3d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dx_mm", "focus_sim_mm", "focus_oneil_mm", "axial_sim_mm", "axial_oneil_mm", "axial_published_mm",
                    "lateral_sim_mm", "lateral_published_mm", "onaxis_shape_rms_error_pct"])
        w.writerow([dx * 1e3, q[i] * 1e3, q[keep][i_on] * 1e3, ax_w, on_w, 26.2, lat_w, 4.1, rms])


if __name__ == "__main__":
    {"plate": check_plate, "arc2d": check_arc, "bowl3d": check_bowl}[sys.argv[1]]()
