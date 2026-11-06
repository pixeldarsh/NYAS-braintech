import os
import numpy as np
import pandas as pd
import mne

data_dir = "sleep_edf"

all_epochs = []
all_labels = []
all_subjects = []

stage_mapping = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM"
}

for filename in sorted(os.listdir(data_dir)):

    if not filename.endswith("-PSG.edf"):
        continue

    subject = filename.replace("-PSG.edf", "")
    hypnogram_file = subject.replace("E0", "EC") + "-Hypnogram.edf"

    # Handle the different hypnogram naming patterns
    hypnogram_file = {
        "SC4011": "SC4011EH-Hypnogram.edf",
        "SC4021": "SC4021EH-Hypnogram.edf",
        "SC4031": "SC4031EC-Hypnogram.edf",
        "SC4041": "SC4041EC-Hypnogram.edf"
    }.get(subject[:6], hypnogram_file)

    psg_path = os.path.join(data_dir, filename)
    hyp_path = os.path.join(data_dir, hypnogram_file)

    print(f"\nProcessing {subject}")

    raw = mne.io.read_raw_edf(
        psg_path, preload=True, verbose=False
    )

    raw.pick(["EEG Fpz-Cz"])
    raw.filter(l_freq=0.3, h_freq=35.0, verbose=False)

    annotations = mne.read_annotations(hyp_path)

    epochs = []
    labels = []

    for annotation in annotations:

        stage = stage_mapping.get(annotation["description"])

        if stage is None:
            continue

        start = annotation["onset"]
        duration = annotation["duration"]

        # Create 30-second epochs
        n_epochs = int(duration // 30)

        for i in range(n_epochs):

            epoch_start = start + i * 30
            epoch_end = epoch_start + 30

            if epoch_end > raw.times[-1]:
                continue

            data, _ = raw[
                :, int(epoch_start * 100):int(epoch_end * 100)
            ]

            if data.shape[1] != 3000:
                continue

            epochs.append(data[0])
            labels.append(stage)

    print("Epochs:", len(epochs))

    all_epochs.extend(epochs)
    all_labels.extend(labels)
    all_subjects.extend([subject] * len(labels))


X = np.array(all_epochs)
y = np.array(all_labels)
groups = np.array(all_subjects)

np.save("eeg_epochs.npy", X)

df = pd.DataFrame({
    "sleep_stage": y,
    "subject": groups
})

df.to_csv("eeg_labels.csv", index=False)

print("\nPREPROCESSING COMPLETE")
print("Epoch array shape:", X.shape)
print("\nStage distribution:")
print(df["sleep_stage"].value_counts())

print("\nParticipant distribution:")
print(df["subject"].value_counts())

