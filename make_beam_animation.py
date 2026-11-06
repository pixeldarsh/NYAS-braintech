"""Animated comparison of the simulated beam in water and through the head (2D, 0.5 mm).

Simulation only. Pressure is shown relative to the peak focal pressure in water.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder
from kwave.utils.checks import check_stability
from kwave.utils.kwave_array import kWaveArray

from lifu_model import ALPHA_POWER, APERTURE, APEX_X, CFL, FREQ, ROC, STABILITY_MARGIN, alpha_coeff, maps

warnings.simplefilter("ignore")
DX = 0.5e-3
T_END = 64e-6
FRAME_EVERY = 0.4e-6


def simulate(scenario):
    kgrid, x, y, c, rho, alpha, roi = maps(DX, scenario=scenario)
    kwargs = dict(sound_speed=c, density=rho)
    if alpha.max() > 0:
        kwargs.update(alpha_coeff=alpha_coeff(alpha), alpha_power=ALPHA_POWER)
    medium = kWaveMedium(**kwargs)
    dt = CFL * DX / float(c.max())
    limit = float(check_stability(kgrid, medium))
    if np.isfinite(limit):
        dt = min(dt, STABILITY_MARGIN * limit)
    # Use a whole number of steps per frame, so frames are evenly spaced
    per_frame = max(1, int(round(FRAME_EVERY / dt)))
    dt = FRAME_EVERY / per_frame
    nt = int(round(T_END / dt))
    kgrid.setTime(nt, dt)
    t = np.arange(nt) * dt

    apex = x.min() + APEX_X
    arr = kWaveArray()
    arr.add_arc_element((apex, 0.0), ROC, APERTURE, (apex + ROC, 0.0))
    source = kSource()
    source.p_mask = arr.get_array_binary_mask(kgrid)
    ramp = np.minimum(t * FREQ / 3.0, 1.0)
    source.p = arr.get_distributed_source_signal(kgrid, (ramp * np.sin(2 * np.pi * FREQ * t))[None, :], order="C")

    sensor = kSensor(mask=np.ones(c.shape, dtype=bool), record=["p"])
    out = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", dtype=np.float32, quiet=True)
    p = np.asarray(out["p"]).reshape(c.shape[0], c.shape[1], -1, order="C")
    frames = p[:, :, ::per_frame]
    if not np.isfinite(frames).all():
        raise FloatingPointError(scenario)
    return frames.astype(np.float32), x, y, apex


if __name__ == "__main__":
    with ProcessPoolExecutor(2) as ex:
        (water, x, y, apex), (head, _, _, _) = list(ex.map(simulate, ["water", "layered"]))
    xmm, ymm = (x - apex) * 1e3, y * 1e3
    n = min(water.shape[2], head.shape[2])
    # Reference: water focal amplitude, from the last frames
    scale = np.abs(water[xmm > 30, :, n - 12:n]).max()
    print("frames", n, "water scale", scale, "head peak in brain region", np.abs(head[xmm > 45, :, n - 12:n]).max() / scale, flush=True)

    cmap = LinearSegmentedColormap.from_list("bo", ["#1F5FAD", "#F7F6F1", "#EB6834"])
    th = np.linspace(0, 2 * np.pi, 500)
    phi = np.linspace(-np.arcsin(0.5), np.arcsin(0.5), 80)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    images = []
    for k in range(0, n):
        fig, ax = plt.subplots(1, 2, figsize=(11.6, 3.5), dpi=100)
        for a, f, title in [(ax[0], water, "In water"), (ax[1], head, "Through skin, skull and brain")]:
            a.imshow(f[:, :, k].T / scale, origin="lower", extent=[xmm[0], xmm[-1], ymm[0], ymm[-1]],
                     cmap=cmap, vmin=-0.7, vmax=0.7, aspect="equal", interpolation="bilinear")
            a.plot(64 - 64 * np.cos(phi), 64 * np.sin(phi), color="#12233F", lw=2.5)
            a.plot([64], [0], marker="+", color="#12233F", ms=9, mew=1.5)
            if a is ax[1]:
                for r in (79, 75, 68.5):
                    a.plot(105 + r * np.cos(th), r * np.sin(th), color="#12233F", lw=0.7)
            a.set_xlim(xmm[0], xmm[-1])
            a.set_ylim(ymm[0], ymm[-1])
            a.set_title(title, loc="left", fontweight="bold", color="#12233F")
            a.set_xlabel("Distance from source (mm)", color="#44546A")
            a.tick_params(colors="#44546A")
            for s in a.spines.values():
                s.set_visible(False)
        fig.tight_layout(pad=0.6)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(rgb).quantize(colors=96, method=Image.Quantize.MEDIANCUT))
        plt.close(fig)
        if k == n - 1:
            Image.fromarray(rgb).save("beam_last_frame.png")
    images[0].save("beam_animation.gif", save_all=True, append_images=images[1:], duration=70, loop=0, optimize=True, disposal=1)
    print("saved beam_animation.gif with", len(images), "frames", flush=True)
