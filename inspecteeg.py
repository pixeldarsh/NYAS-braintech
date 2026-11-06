import mne

psg_path = "/Users/darsh/mne_data/physionet-sleep-data/SC4001E0-PSG.edf"
hyp_path = "/Users/darsh/mne_data/physionet-sleep-data/SC4001EC-Hypnogram.edf"

# Load EEG recording
raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)

print("\n--- EEG RECORDING ---")
print("Channels:", raw.ch_names)
print("Sampling frequency:", raw.info["sfreq"], "Hz")
print("Duration:", raw.times[-1] / 3600, "hours")

# Load sleep-stage annotations
annotations = mne.read_annotations(hyp_path)

print("\n--- SLEEP ANNOTATIONS ---")
print("Number of annotations:", len(annotations))
print("First 10 annotations:")
print(annotations[:10])
from collections import Counter

print("\n--- SLEEP STAGE DISTRIBUTION ---")

stage_counts = Counter(annotations.description)

for stage, count in sorted(stage_counts.items()):
    print(f"{stage}: {count}")
print("\n--- ANNOTATION DETAILS ---")

for i in range(min(15, len(annotations))):
    print(
        f"Onset: {annotations.onset[i]:.1f}s | "
        f"Duration: {annotations.duration[i]:.1f}s | "
        f"Stage: {annotations.description[i]}"
    )
print("\n--- RECORDING AND ANNOTATION ALIGNMENT ---")

print("EEG start:", raw.info["meas_date"])
print("EEG duration:", raw.times[-1])
print("EEG samples:", raw.n_times)

print("Annotation onset:", annotations.onset[0])
print("Annotation end:",
      annotations.onset[-1] + annotations.duration[-1])

print("First 5 EEG annotation onsets:")
print(annotations.onset[:5])
print("\n--- FINAL ANNOTATION CHECK ---")

for stage in sorted(set(annotations.description)):
    mask = annotations.description == stage
    total_duration = annotations.duration[mask].sum()

    print(
        f"{stage}: "
        f"{total_duration / 3600:.2f} hours"
    )

print("\nEEG recording end:", raw.times[-1])
print("Hypnogram end:",
      annotations.onset[-1] + annotations.duration[-1])
print("\n--- UNscored ANNOTATIONS ---")

for onset, duration, description in zip(
    annotations.onset,
    annotations.duration,
    annotations.description
):
    if description == "Sleep stage ?":
        print(
            f"Start: {onset:.1f}s | "
            f"End: {onset + duration:.1f}s | "
            f"Duration: {duration / 3600:.2f} hours"
        )

print("\n--- LAST 10 ANNOTATIONS ---")

for onset, duration, description in zip(
    annotations.onset[-10:],
    annotations.duration[-10:],
    annotations.description[-10:]
):
    print(onset, duration, description)