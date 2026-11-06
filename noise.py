
import numpy as np
import pandas as pd

from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

# Load data
epochs = np.load("eeg_epochs.npy")
labels_df = pd.read_csv("eeg_labels.csv")
features_df = pd.read_csv("eeg_features.csv")

y = labels_df["sleep_stage"].values
subjects = labels_df["subject"].values

feature_columns = [
    "delta_absolute", "theta_absolute", "alpha_absolute",
    "sigma_absolute", "beta_absolute",
    "delta_relative", "theta_relative", "alpha_relative",
    "sigma_relative", "beta_relative"
]

# Confirm row alignment
assert len(epochs) == len(y) == len(subjects) == len(features_df)

# Extract the same ten features from noisy epochs
def extract_features(epoch):
    frequencies, psd = welch(epoch, fs=100, nperseg=1000)

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "sigma": (13, 16),
        "beta": (16, 30)
    }

    absolute = {}

    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies <= high)
        absolute[name] = np.trapezoid(psd[mask], frequencies[mask])

    total = sum(absolute.values())

    result = []

    for name in bands:
        result.append(absolute[name])
    for name in bands:
        result.append(absolute[name] / total if total > 0 else 0)

    return result


# Add Gaussian noise at a specified SNR
def add_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)

    if signal_power == 0:
        return signal.copy()

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise


snr_levels = [None, 20, 10, 5, 0]
results = []

logo = LeaveOneGroupOut()

for fold, (train_idx, test_idx) in enumerate(
    logo.split(epochs, y, groups=subjects), 1
):

    print(f"\nParticipant fold {fold}: {subjects[test_idx][0]}")

    X_train = features_df.iloc[train_idx][feature_columns].values
    y_train = y[train_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_epochs = epochs[test_idx]
    y_test = y[test_idx]

    for snr in snr_levels:

        if snr is None:
            X_test = features_df.iloc[test_idx][feature_columns].values
            condition = "Clean"

        else:
            rng = np.random.default_rng(42)

            noisy_epochs = np.array([
                add_noise(epoch, snr, rng)
                for epoch in test_epochs
            ])

            X_test = np.array([
                extract_features(epoch)
                for epoch in noisy_epochs
            ])

            condition = f"{snr} dB"

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        confidence = probabilities.max(axis=1)
        entropy = -np.sum(
            np.where(
                probabilities > 0,
                probabilities * np.log(
                    np.maximum(probabilities, 1e-15)
                ),
                0
            ),
            axis=1
        )

        results.append({
            "subject": subjects[test_idx][0],
            "condition": condition,
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(
                y_test, predictions,
                average="macro",
                zero_division=0
            ),
            "mean_confidence": confidence.mean(),
            "mean_entropy": entropy.mean()
        })

results_df = pd.DataFrame(results)
results_df.to_csv("noise_robustness_results.csv", index=False)

print("\n========== NOISE ROBUSTNESS RESULTS ==========")

print(
    results_df.groupby("condition")[
        ["accuracy", "macro_f1", "mean_confidence", "mean_entropy"]
    ].mean().round(4)
)

print("\nDetailed participant results:")
print(results_df.round(4).to_string(index=False))

print("\nSaved to noise_robustness_results.csv")
