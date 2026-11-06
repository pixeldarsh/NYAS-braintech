

import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder


# One time step shared by every resolution, so differences between grids
# are spatial rather than temporal. Sized for the finest grid:
# CFL = c_max * DT / dx = 2800 * 5e-9 / 0.0625e-3 = 0.224
DT = 5e-9
CFL_LIMIT = 0.3


def run_simulation(dx):

    print(f"\nRunning grid spacing: {dx * 1000:.4f} mm")

    # 1. Fixed physical domain: 32 x 32 mm
    Nx = int(round(32e-3 / dx))
    Ny = int(round(32e-3 / dx))

    kgrid = kWaveGrid([Nx, Ny], [dx, dx])

    # 2. Physical coordinates in mm
    x = np.arange(Nx) * dx * 1000
    y = np.arange(Ny) * dx * 1000
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 3. Virtual head geometry
    cx, cy = 19.0, 16.0

    R = np.sqrt(
        ((X - cx) / 11.0) ** 2 +
        ((Y - cy) / 13.0) ** 2
    )

    outer_head = R <= 1.0
    scalp_inner = R <= 0.95
    skull_inner = R <= 0.80
    csf_inner = R <= 0.75

    scalp = outer_head & ~scalp_inner
    skull = scalp_inner & ~skull_inner
    csf = skull_inner & ~csf_inner
    brain = csf_inner

    # 4. Acoustic properties
    sound_speed = np.full((Nx, Ny), 1480.0)
    density = np.full((Nx, Ny), 1000.0)

    sound_speed[scalp] = 1540
    density[scalp] = 1050

    sound_speed[skull] = 2800
    density[skull] = 1900

    sound_speed[csf] = 1500
    density[csf] = 1000

    sound_speed[brain] = 1540
    density[brain] = 1040

    medium = kWaveMedium(
        sound_speed=sound_speed,
        density=density
    )

    # 5. Fixed 1 mm source aperture
    source = kSource()
    source.p_mask = np.zeros((Nx, Ny), dtype=bool)

    # Source centre: (4 mm, 16 mm)
    source_i = round(4e-3 / dx)
    source_j = round(16e-3 / dx)

    number_of_points = round(1e-3 / dx)

    start_j = source_j - number_of_points // 2
    end_j = start_j + number_of_points

    source.p_mask[source_i, start_j:end_j] = True

    print(f"Source points: {number_of_points}")
    print(f"Physical aperture: {number_of_points * dx * 1000:.4f} mm")

    # 6. Time array and source signal
    frequency = 1e6
    dt = DT

    c_max = float(np.max(sound_speed))
    cfl = c_max * dt / dx
    print(f"CFL number: {cfl:.3f} (limit {CFL_LIMIT})")
    if cfl > CFL_LIMIT:
        raise ValueError(
            f"CFL {cfl:.3f} exceeds {CFL_LIMIT} at dx={dx * 1000:.4f} mm; "
            f"reduce DT (max {CFL_LIMIT * dx / c_max:.2e} s)"
        )

    t_array = np.arange(0, 20e-6 + dt / 2, dt)
    kgrid.setTime(len(t_array), dt)

    source.p = np.sin(2 * np.pi * frequency * t_array)

    # 7. Sensor at exactly (19 mm, 16 mm)
    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Ny), dtype=bool)

    target_i = round(19e-3 / dx)
    target_j = round(16e-3 / dx)

    sensor.mask[target_i, target_j] = True

    print(
        f"Sensor coordinate: "
        f"({x[target_i]:.4f}, {y[target_j]:.4f}) mm"
    )

    # 8. Run simulation
    sensor_data = kspaceFirstOrder(
        kgrid,
        medium,
        source,
        sensor,
        backend="python"
    )

    pressure = sensor_data["p"][0, :]

    # 9. Analyse 5–20 microseconds
    start_index = int(5e-6 / dt)
    end_index = int(20e-6 / dt)

    pressure_window = pressure[start_index:end_index]
    time_window = t_array[start_index:end_index] * 1e6

    rms = np.sqrt(np.mean(pressure_window ** 2))
    peak = np.max(np.abs(pressure_window))

    print(f"Point RMS pressure: {rms:.8f} Pa")
    print(f"Point peak pressure: {peak:.8f} Pa")

    del sensor_data
    del pressure

    return time_window, pressure_window, rms, peak


# 10. Run four resolutions
grid_spacings = [
    0.5e-3,
    0.25e-3,
    0.125e-3,
    0.0625e-3
]

results = {}

for spacing in grid_spacings:
    results[spacing] = run_simulation(spacing)


# 11. Plot waveforms
plt.figure(figsize=(10, 5))

for spacing in grid_spacings:

    time, pressure, rms, peak = results[spacing]

    plt.plot(
        time,
        pressure,
        label=f"{spacing * 1000:.4f} mm"
    )

plt.xlabel("Time (microseconds)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure waveform at (19 mm, 16 mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 12. Compare against finest grid
print("\n========== GRID CONVERGENCE RESULTS ==========")

reference_rms = results[0.0625e-3][2]
reference_peak = results[0.0625e-3][3]

for spacing in grid_spacings:

    rms = results[spacing][2]
    peak = results[spacing][3]

    rms_difference = (
        (rms - reference_rms) / reference_rms
    ) * 100

    peak_difference = (
        (peak - reference_peak) / reference_peak
    ) * 100

    print(
        f"{spacing * 1000:.4f} mm | "
        f"RMS: {rms:.8f} Pa | "
        f"Peak: {peak:.8f} Pa | "
        f"RMS difference: {rms_difference:+.2f}% | "
        f"Peak difference: {peak_difference:+.2f}%"
    )

print("\nGrid comparison completed.")

