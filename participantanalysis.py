
import pandas as pd

df = pd.read_csv("eeg_features.csv")

print("\nEPOCH COUNTS BY PARTICIPANT AND STAGE")
counts = pd.crosstab(
    df["subject"],
    df["sleep_stage"]
)
print(counts)

print("\nSTAGE DISTRIBUTION BY PARTICIPANT (%)")
distribution = pd.crosstab(
    df["subject"],
    df["sleep_stage"],
    normalize="index"
) * 100

print(distribution.round(2))

print("\nALWAYS-WAKE BASELINE BY PARTICIPANT")
for subject, group in df.groupby("subject"):
    baseline = (group["sleep_stage"] == "Wake").mean()
    print(subject, round(baseline * 100, 2), "%")
