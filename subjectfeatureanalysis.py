
import pandas as pd

df = pd.read_csv("eeg_features.csv")

features = [
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "sigma_relative",
    "beta_relative"
]

print("\nMEDIAN FEATURES BY PARTICIPANT")
print(
    df.groupby("subject")[features]
    .median()
    .round(4)
)

print("\nMEDIAN FEATURES BY PARTICIPANT AND STAGE")
print(
    df.groupby(["subject", "sleep_stage"])[features]
    .median()
    .round(4)
)
