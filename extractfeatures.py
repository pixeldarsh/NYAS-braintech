import numpy as np
import pandas as pd
from scipy.signal import welch

# Load preprocessed EEG
X = np.load("eeg_epochs.npy")
labels = pd.read_csv("eeg_labels.csv")

bands = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (13, 16),
    "beta": (16, 30)
}

features = []

for i, epoch in enumerate(X):

    frequencies, psd = welch(
        epoch,
        fs=100,
        nperseg=1000
    )

    total_power = np.trapezoid(
        psd[(frequencies >= 0.5) & (frequencies <= 30)],
        frequencies[(frequencies >= 0.5) & (frequencies <= 30)]
    )

    row = {}

    for band, (low, high) in bands.items():

        mask = (frequencies >= low) & (frequencies <= high)

        absolute_power = np.trapezoid(
            psd[mask],
            frequencies[mask]
        )

        relative_power = (
            absolute_power / total_power
            if total_power > 0 else 0
        )

        row[f"{band}_absolute"] = absolute_power
        row[f"{band}_relative"] = relative_power

    features.append(row)

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(X)} epochs")

feature_df = pd.DataFrame(features)

feature_df["sleep_stage"] = labels["sleep_stage"]
feature_df["subject"] = labels["subject"]

feature_df.to_csv("eeg_features.csv", index=False)

print("\nFEATURE EXTRACTION COMPLETE")
print("Shape:", feature_df.shape)
print(feature_df.head())
