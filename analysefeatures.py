import pandas as pd
import numpy as np

df = pd.read_csv("eeg_features.csv")

bands = [
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "sigma_relative",
    "beta_relative"
]

# Median relative power for each sleep stage
print("\nMEDIAN RELATIVE BAND POWER BY STAGE")
print(df.groupby("sleep_stage")[bands].median().round(3))

# Mean relative power
print("\nMEAN RELATIVE BAND POWER BY STAGE")
print(df.groupby("sleep_stage")[bands].mean().round(3))

# Check whether relative powers approximately sum to 1
df["relative_power_sum"] = df[bands].sum(axis=1)

print("\nRELATIVE POWER SUM")
print(df["relative_power_sum"].describe().round(4))

# Check feature distributions across participants
print("\nEPOCHS PER PARTICIPANT")
print(df.groupby("subject").size())

# Check for missing or invalid values
print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nNON-FINITE FEATURE VALUES")
print(np.isfinite(df[bands].to_numpy()).all())
import matplotlib.pyplot as plt
import seaborn as sns

for band in bands:
    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="sleep_stage",
        y=band,
        order=["Wake", "N1", "N2", "N3", "REM"]
    )

    plt.title(f"{band} across sleep stages")
    plt.xlabel("Sleep stage")
    plt.ylabel("Relative power")
    plt.tight_layout()
    plt.savefig(f"{band}_distribution.png", dpi=200)
    plt.show()
    print("\nPERCENTAGE OF EPOCHS WITH DELTA > 90%")

df["high_delta"] = df["delta_relative"] > 0.90

print(
    df.groupby("sleep_stage")["high_delta"]
    .mean()
    .mul(100)
    .round(2)
)
feature_cols = [
    "delta_absolute", "delta_relative",
    "theta_absolute", "theta_relative",
    "alpha_absolute", "alpha_relative",
    "sigma_absolute", "sigma_relative",
    "beta_absolute", "beta_relative"
]

import numpy as np

print("\nNON-FINITE VALUES PER FEATURE")
print(
    (~np.isfinite(df[feature_cols]))
    .sum()
)

print("\nROWS CONTAINING NON-FINITE VALUES")
bad_rows = ~np.isfinite(df[feature_cols]).all(axis=1)
print(df.loc[bad_rows].head(10))

print("\nTOTAL NON-FINITE ROWS:", bad_rows.sum())