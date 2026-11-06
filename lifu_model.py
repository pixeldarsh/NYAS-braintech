"""2D focused-ultrasound head model with absorption (simulation only, uncalibrated).

Tissue values are Table I of Aubry et al. 2022 (JASA 152:1003), which the paper calls
representative rather than definitive. Absorption is 500 kHz values applied with a
frequency-squared power law, as the paper's benchmarks do. Geometry follows its curved
layered benchmark in 2D: a 64 mm arc source, skull 30 mm from the source, and
skin 4 / cortical 1.5 / trabecular 4 / cortical 1 mm layers around brain.

Pressures are relative to the same source in water. They are not calibrated pressures
and are not safety or dosing values. 2D gives a line focus, not a 3D bowl focus.
"""
import warnings

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.checks import check_stability
from kwave.utils.kwave_array import kWaveArray

warnings.simplefilter("ignore", FutureWarning)

FREQ = 500e3
LX, LY = 120e-3, 70e-3
APEX_X = 2e-3            # source apex, measured from the x = 0 edge of the domain
ROC = APERTURE = 64e-3
SKULL_DISTANCE = 30e-3   # source apex to outer skull
CFL = 0.2
T_END = 80e-6
ALPHA_POWER = 2.0
SAFETY_CFL_LIMIT = 0.3
STABILITY_MARGIN = 0.9
SENSOR_ORDER = "C"  # k-Wave 0.6.2 returns C order; tested with an off-centre source on a non-square grid

# c (m/s), rho (kg/m^3), alpha (dB/cm at 500 kHz)
TISSUES = {
    "water": (1500.0, 1000.0, 0.0),
    "skin": (1610.0, 1090.0, 0.2),
    "brain": (1560.0, 1040.0, 0.3),
    "cortical": (2800.0, 1850.0, 4.0),
    "trabecular": (2300.0, 1700.0, 8.0),
}
# Layer thicknesses in mm, outside to inside
THICKNESS = {"skin": 4.0, "outer": 1.5, "trabecular": 4.0, "inner": 1.0}


def alpha_coeff(db_per_cm_at_500khz):
    """Convert to dB/(MHz^2 cm) for the frequency-squared power law."""
    return db_per_cm_at_500khz / 0.5 ** 2


def coords(dx):
    nx, ny = int(round(LX / dx)), int(round(LY / dx))
    kgrid = kWaveGrid([nx, ny], [dx, dx])
    x = np.asarray(kgrid.x_vec).ravel()
    y = np.asarray(kgrid.y_vec).ravel()
    return kgrid, nx, ny, x, y


def maps(dx, tissues=None, thickness=None, scenario="layered"):
    """Return c, rho, alpha and the brain mask.

    scenario: 'water', 'homogeneous_brain' (brain everywhere past the skin surface),
    'layered', or 'single_layer' (one cortical skull with the same total thickness).
    """
    t = {**TISSUES, **(tissues or {})}
    th = {**THICKNESS, **(thickness or {})}
    kgrid, nx, ny, x, y = coords(dx)
    X, Y = np.meshgrid(x, y, indexing="ij")

    skull_total = th["outer"] + th["trabecular"] + th["inner"]
    r_skull_out = 75.0e-3
    # Head centre stays put when thickness changes: the outer skull is fixed and
    # the layers grow inward, so the focus depth inside the brain changes with thickness.
    x_apex = x.min() + APEX_X
    centre_x = x_apex + SKULL_DISTANCE + r_skull_out
    r = np.sqrt((X - centre_x) ** 2 + Y ** 2)

    r_skin = r_skull_out + th["skin"] * 1e-3
    r_trab_out = r_skull_out - th["outer"] * 1e-3
    r_trab_in = r_trab_out - th["trabecular"] * 1e-3
    r_brain = r_skull_out - skull_total * 1e-3

    label = np.full((nx, ny), "water", dtype=object)
    if scenario == "homogeneous_brain":
        label[r <= r_skin] = "brain"
    elif scenario != "water":
        label[r <= r_skin] = "skin"
        if scenario == "single_layer":
            label[r <= r_skull_out] = "cortical"
        else:
            label[r <= r_skull_out] = "cortical"
            label[r <= r_trab_out] = "trabecular"
            label[r <= r_trab_in] = "cortical"
        label[r <= r_brain] = "brain"

    c = np.zeros((nx, ny))
    rho = np.zeros((nx, ny))
    alpha = np.zeros((nx, ny))
    for name, (ci, ri, ai) in t.items():
        m = label == name
        c[m], rho[m], alpha[m] = ci, ri, ai

    # Search region for the focus: inside the brain, 3 mm clear of the inner skull
    roi = r <= r_brain - 3e-3
    return kgrid, x, y, c, rho, alpha, roi


def run(dx, tissues=None, thickness=None, scenario="layered", absorb=True,
        t_end=T_END, dtype=np.float32, quiet=True):
    """Simulate continuous-wave focusing and return the peak-pressure map."""
    kgrid, x, y, c, rho, alpha, roi = maps(dx, tissues, thickness, scenario)
    kwargs = dict(sound_speed=c, density=rho)
    if absorb and alpha.max() > 0:
        kwargs.update(alpha_coeff=alpha_coeff(alpha), alpha_power=ALPHA_POWER)
    medium = kWaveMedium(**kwargs)

    # Absorption tightens the stable time step as the grid gets finer, so CFL alone is not enough
    dt = CFL * dx / float(c.max())
    limit = float(check_stability(kgrid, medium))
    if np.isfinite(limit):
        dt = min(dt, STABILITY_MARGIN * limit)
    if float(c.max()) * dt / dx > SAFETY_CFL_LIMIT:
        raise ValueError("CFL limit exceeded")
    nt = int(round(t_end / dt))
    kgrid.setTime(nt, dt)
    t = np.arange(nt) * dt

    x_apex = x.min() + APEX_X
    arr = kWaveArray()
    arr.add_arc_element((x_apex, 0.0), ROC, APERTURE, (x_apex + ROC, 0.0))

    # Ramp over 3 cycles to limit broadband start-up transients
    ramp = np.minimum(t * FREQ / 3.0, 1.0)
    signal = (ramp * np.sin(2 * np.pi * FREQ * t))[None, :]
    source = kSource()
    source.p_mask = arr.get_array_binary_mask(kgrid)
    source.p = arr.get_distributed_source_signal(kgrid, signal, order="C")

    sensor = kSensor(mask=np.ones(c.shape, dtype=bool), record=["p_max"])
    # Peak over the last 2 cycles, once the field has reached a steady state
    sensor.record_start_index = max(1, nt - int(round(2 / FREQ / dt)))

    out = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", dtype=dtype, quiet=quiet)
    pmax = np.asarray(out["p_max"])
    if pmax.ndim == 1:
        pmax = pmax.reshape(c.shape, order=SENSOR_ORDER)
    if not np.isfinite(pmax).all():
        raise FloatingPointError(f"Non-finite pressure at dx={dx * 1e3} mm, CFL={CFL}, absorb={absorb}")
    return {"p": pmax, "x": x, "y": y, "roi": roi, "x_apex": x_apex, "dt": dt, "nt": nt}


def width_db6(profile, peak_i, spacing):
    """Full width at half the peak pressure (-6 dB), with linear interpolation."""
    half = profile[peak_i] / 2
    i = peak_i
    while i > 0 and profile[i] >= half:
        i -= 1
    j = peak_i
    while j < len(profile) - 1 and profile[j] >= half:
        j += 1
    if profile[i] >= half or profile[j] >= half:
        return np.nan
    left = i + (half - profile[i]) / (profile[i + 1] - profile[i])
    right = j - 1 + (profile[j - 1] - half) / (profile[j - 1] - profile[j])
    return (right - left) * spacing


def focus_metrics(res, region=None):
    """Peak pressure, position and -6 dB size within `region` (default: the brain region)."""
    p, x, y = res["p"], res["x"], res["y"]
    region = res["roi"] if region is None else region
    masked = np.where(region, p, -np.inf)
    i, j = np.unravel_index(np.argmax(masked), p.shape)
    dx = x[1] - x[0]

    # Quadratic refinement so the position is not limited to whole grid points
    def refine(a, b, c_):
        d = a - 2 * b + c_
        return 0.5 * (a - c_) / d if d != 0 else 0.0

    fx = x[i] + dx * refine(p[i - 1, j], p[i, j], p[i + 1, j])
    fy = y[j] + dx * refine(p[i, j - 1], p[i, j], p[i, j + 1])
    target_x = res["x_apex"] + ROC
    return {
        "peak": float(p[i, j]),
        "focus_x_from_apex_mm": (fx - res["x_apex"]) * 1e3,
        "focus_y_mm": fy * 1e3,
        "shift_from_target_mm": float(np.hypot(fx - target_x, fy)) * 1e3,
        "axial_6db_mm": width_db6(p[:, j], i, dx) * 1e3,
        "lateral_6db_mm": width_db6(p[i, :], j, dx) * 1e3,
    }
