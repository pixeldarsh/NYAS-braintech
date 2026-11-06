
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("eeg_features.csv")

# Plot relative band power by sleep stage
bands = [
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "sigma_relative",
    "beta_relative"
]

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

    plt.savefig(f"{band}.png", dpi=150)
    plt.show()

print("All five plots generated.")
 