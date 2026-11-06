
import pandas as pd

df = pd.read_csv("oof_predictions.csv")

print("\nCONFIDENCE AND ACCURACY BY ACTUAL STAGE")

for stage, group in df.groupby("actual_stage"):
    print(f"\n{stage}")
    print(
        group.groupby(
            pd.cut(
                group["confidence"],
                bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001],
                include_lowest=True
            ),
            observed=False
        )["correct"]
        .agg(["count", "mean"])
        .round(3)
    )

print("\nOVERALL ERROR RATE")
print(round((~df["correct"]).mean(), 4))

print("\nMEAN CONFIDENCE: CORRECT VS INCORRECT")
print(
    df.groupby("correct")["confidence"]
    .agg(["count", "mean", "median"])
    .round(4)
)

print("\nENTROPY: CORRECT VS INCORRECT")
print(
    df.groupby("correct")["entropy"]
    .agg(["mean", "median"])
    .round(4)
)
