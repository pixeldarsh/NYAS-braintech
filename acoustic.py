
import numpy as np
import matplotlib.pyplot as plt

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder


# 1. Computational grid
Nx, Ny = 128, 128
dx = 0.25e-3
dy = 0.25e-3

kgrid = kWaveGrid([Nx, Ny], [dx, dy])


# 2. Homogeneous medium
# 2. Virtual head geometry

x = np.arange(Nx) * dx * 1000  # mm
y = np.arange(Ny) * dy * 1000  # mm

X, Y = np.meshgrid(x, y, indexing="ij")

# Centre of virtual head
cx = 19.0
cy = 16.0

# Elliptical distance from head centre
R = np.sqrt(
    ((X - cx) / 11.0) ** 2 +
    ((Y - cy) / 13.0) ** 2
)

# Define tissue regions
outer_head = R <= 1.0
scalp_inner = R <= 0.95
skull_inner = R <= 0.80
csf_inner = R <= 0.75

scalp = outer_head & ~scalp_inner
skull = scalp_inner & ~skull_inner
csf = skull_inner & ~csf_inner
brain = csf_inner

# Initialise acoustic property maps
sound_speed = np.full((Nx, Ny), 1480.0)
density = np.full((Nx, Ny), 1000.0)

# Assign tissue properties
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

print("Virtual head medium created.")
print("Tissue regions:")
print("Scalp:", np.sum(scalp))
print("Skull:", np.sum(skull))
print("CSF:", np.sum(csf))
print("Brain:", np.sum(brain))
# Visualise virtual head tissue distribution

tissue_map = np.zeros((Nx, Ny))

tissue_map[scalp] = 1
tissue_map[skull] = 2
tissue_map[csf] = 3
tissue_map[brain] = 4

plt.figure(figsize=(7, 6))

plt.imshow(
    tissue_map.T,
    origin="lower",
    extent=[0, Nx * dx * 1000, 0, Ny * dy * 1000],
    cmap="viridis",
    interpolation="nearest",
    aspect="equal"
)

plt.colorbar(
    ticks=[0, 1, 2, 3, 4],
    label="Tissue region",
)

plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.title("Virtual Head Tissue Map")
plt.tight_layout()
plt.savefig("virtual_head_tissue_map.png", dpi=300)
plt.show()


# 3. Acoustic source
source = kSource()

source.p_mask = np.zeros((Nx, Ny), dtype=bool)
source.p_mask[16, Ny // 2] = True

frequency = 1e6

dt = 2e-8
t_array = np.arange(0, 20e-6, dt)

kgrid.setTime(len(t_array), dt)

source.p = np.sin(2 * np.pi * frequency * t_array)


# 4. Full-grid sensor
sensor = kSensor()
sensor.mask = np.ones((Nx, Ny), dtype=bool)


# 5. Run simulation
print("Starting acoustic simulation...")

sensor_data = kspaceFirstOrder(
    kgrid,
    medium,
    source,
    sensor,
    backend="python"
)

print("\nSimulation completed.")


# 6. Extract results
pressure_data = sensor_data["p"]
final_pressure = sensor_data["p_final"]

print("\nAvailable outputs:", sensor_data.keys())
print("Pressure data shape:", pressure_data.shape)
print("Final pressure field shape:", final_pressure.shape)


# 7. Numerical analysis
print("\n--- Pressure Statistics ---")

print(
    "Maximum absolute pressure:",
    np.max(np.abs(final_pressure)),
    "Pa"
)

print(
    "Maximum pressure:",
    np.max(final_pressure),
    "Pa"
)

print(
    "Minimum pressure:",
    np.min(final_pressure),
    "Pa"
)

print(
    "Maximum absolute pressure location:",
    np.unravel_index(
        np.argmax(np.abs(final_pressure)),
        final_pressure.shape
    )
)


# 8. Plot final pressure field
plt.figure(figsize=(7, 6))

plt.imshow(
    final_pressure.T,
    origin="lower",
    cmap="RdBu_r",
    extent=[
        0, Nx * dx * 1000,
        0, Ny * dy * 1000
    ],
    aspect="equal"
)

plt.colorbar(label="Pressure (Pa)")
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")
plt.title("Final Acoustic Pressure Field")

plt.tight_layout()
plt.savefig("pressure_field_high_resolution.png", dpi=300)
plt.show()


# 9. Reconstruct an earlier pressure snapshot
time_index = 150

pressure_snapshot = pressure_data[:, time_index].reshape(
    (Nx, Ny),
    order="F"
)

plt.figure(figsize=(7, 6))

plt.imshow(
    pressure_snapshot.T,
    origin="lower",
    cmap="RdBu_r",
    extent=[
        0, Nx * dx * 1000,
        0, Ny * dy * 1000
    ],
    aspect="equal"
)

plt.colorbar(label="Pressure (Pa)")
plt.xlabel("X position (mm)")
plt.ylabel("Y position (mm)")

plt.title(
    f"Acoustic Pressure Field at "
    f"{t_array[time_index] * 1e6:.2f} µs"
)

plt.tight_layout()
plt.savefig("pressure_snapshot.png", dpi=300)
plt.show()


print("\nBoth pressure-field plots saved successfully.")
# 10. Calculate pressure statistics by tissue region

tissue_regions = {
    "Scalp": scalp,
    "Skull": skull,
    "CSF": csf,
    "Brain": brain
}

print("\n--- Tissue-wise Pressure Analysis ---")

for name, mask in tissue_regions.items():

    # Convert 2D mask to the sensor's flattened ordering
    region_indices = np.flatnonzero(mask.flatten(order="F"))

    # Extract pressure over time for this tissue
    region_pressure = pressure_data[region_indices, :]

    peak_pressure = np.max(np.abs(region_pressure))
    rms_pressure = np.sqrt(np.mean(region_pressure ** 2))

    print(f"\n{name}:")
    print(f"Peak absolute pressure: {peak_pressure:.6f} Pa")
    print(f"RMS pressure: {rms_pressure:.6f} Pa")

# 11. Save virtual-head brain measurements

brain_indices = np.flatnonzero(brain.flatten(order="F"))

virtual_brain_pressure = pressure_data[brain_indices, :]

virtual_brain_rms = np.sqrt(
    np.mean(virtual_brain_pressure ** 2)
)

virtual_brain_peak = np.max(
    np.abs(virtual_brain_pressure)
)

print("\n--- Virtual Head Brain Results ---")
print(f"Brain RMS pressure: {virtual_brain_rms:.6f} Pa")
print(f"Brain peak pressure: {virtual_brain_peak:.6f} Pa")


# 12. Run homogeneous control simulation
# Uniform properties throughout the entire domain.
# These match the nominal brain properties.

print("\nStarting homogeneous control simulation...")

homogeneous_medium = kWaveMedium(
    sound_speed=1540,
    density=1040
)

homogeneous_data = kspaceFirstOrder(
    kgrid,
    homogeneous_medium,
    source,
    sensor,
    backend="python"
)

print("Homogeneous control completed.")


# 13. Measure pressure in the same brain-shaped region

homogeneous_pressure = homogeneous_data["p"]

homogeneous_brain_pressure = homogeneous_pressure[
    brain_indices, :
]

homogeneous_brain_rms = np.sqrt(
    np.mean(homogeneous_brain_pressure ** 2)
)

homogeneous_brain_peak = np.max(
    np.abs(homogeneous_brain_pressure)
)


# 14. Compare both simulations

print("\n========== CONTROL COMPARISON ==========")

print("\nVirtual head:")
print(f"Brain RMS pressure: {virtual_brain_rms:.6f} Pa")
print(f"Brain peak pressure: {virtual_brain_peak:.6f} Pa")

print("\nHomogeneous control:")
print(f"Brain RMS pressure: {homogeneous_brain_rms:.6f} Pa")
print(f"Brain peak pressure: {homogeneous_brain_peak:.6f} Pa")

rms_difference = (
    (virtual_brain_rms - homogeneous_brain_rms)
    / homogeneous_brain_rms
) * 100

print(
    f"\nRelative RMS difference: {rms_difference:.2f}%"
)

print("\n========================================")


# 15. Visual comparison

labels = ["Virtual head", "Homogeneous control"]

rms_values = [
    virtual_brain_rms,
    homogeneous_brain_rms
]

plt.figure(figsize=(7, 5))

plt.bar(labels, rms_values)

plt.ylabel("Brain-region RMS pressure (Pa)")
plt.title("Heterogeneous vs Homogeneous Medium")
plt.tight_layout()

plt.savefig("brain_pressure_comparison.png", dpi=300)
plt.show()

# 16. Skull sound-speed sensitivity experiment

print("\n========== SKULL SOUND-SPEED SENSITIVITY ==========")

skull_speeds = [2400, 2800, 3200]

sensitivity_results = []

for skull_speed in skull_speeds:

    print(f"\nRunning skull sound speed: {skull_speed} m/s")

    # Copy original acoustic property maps
    scenario_sound_speed = sound_speed.copy()

    # Change only the skull sound speed
    scenario_sound_speed[skull] = skull_speed

    scenario_medium = kWaveMedium(
        sound_speed=scenario_sound_speed,
        density=density.copy()
    )

    # Run simulation
    scenario_data = kspaceFirstOrder(
        kgrid,
        scenario_medium,
        source,
        sensor,
        backend="python"
    )

    # Extract pressure in the brain region
    scenario_pressure = scenario_data["p"][brain_indices, :]

    # Analyse the 5–20 microsecond interval
    start_index = int(5e-6 / dt)
    end_index = int(20e-6 / dt)

    arrival_window = scenario_pressure[:, start_index:end_index]

    brain_rms = np.sqrt(np.mean(arrival_window ** 2))
    brain_peak = np.max(np.abs(arrival_window))

    sensitivity_results.append(
        (skull_speed, brain_rms, brain_peak)
    )

    print(f"Brain RMS pressure: {brain_rms:.6f} Pa")
    print(f"Brain peak pressure: {brain_peak:.6f} Pa")


# Display comparison
print("\n========== FINAL RESULTS ==========")

baseline_rms = next(
    result[1]
    for result in sensitivity_results
    if result[0] == 2800
)

for skull_speed, brain_rms, brain_peak in sensitivity_results:

    difference = (
        (brain_rms - baseline_rms) / baseline_rms
    ) * 100

    print(
        f"{skull_speed} m/s | "
        f"RMS: {brain_rms:.6f} Pa | "
        f"Peak: {brain_peak:.6f} Pa | "
        f"Difference from baseline: {difference:+.2f}%"
    )

print("\n===================================")
# 17. Skull density sensitivity experiment

print("\n========== SKULL DENSITY SENSITIVITY ==========")

skull_densities = [1700, 1900, 2100]
density_results = []

for skull_density in skull_densities:

    print(f"\nRunning skull density: {skull_density} kg/m^3")

    scenario_density = density.copy()
    scenario_density[skull] = skull_density

    scenario_medium = kWaveMedium(
        sound_speed=sound_speed.copy(),
        density=scenario_density
    )

    scenario_data = kspaceFirstOrder(
        kgrid, scenario_medium, source, sensor, backend="python"
    )

    scenario_pressure = scenario_data["p"][brain_indices, :]

# Use the same 5–20 microsecond window
    start_index = int(5e-6 / dt)
    end_index = int(20e-6 / dt)

    arrival_window = scenario_pressure[:, start_index:end_index]

    brain_rms = np.sqrt(np.mean(arrival_window ** 2))
    brain_peak = np.max(np.abs(arrival_window))

    density_results.append(
        (skull_density, brain_rms, brain_peak)
    )

    print(f"Brain RMS pressure: {brain_rms:.6f} Pa")
    print(f"Brain peak pressure: {brain_peak:.6f} Pa")


baseline_rms = next(
    result[1] for result in density_results
    if result[0] == 1900
)

print("\n========== FINAL RESULTS ==========")

for skull_density, brain_rms, brain_peak in density_results:

    difference = (
        (brain_rms - baseline_rms) / baseline_rms
    ) * 100

    print(
        f"{skull_density} kg/m^3 | "
        f"RMS: {brain_rms:.6f} Pa | "
        f"Peak: {brain_peak:.6f} Pa | "
        f"Difference from baseline: {difference:+.2f}%"
    )


