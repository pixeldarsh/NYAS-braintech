import pandas as pd

df = pd.read_csv("oof_predictions.csv")

print("\nN1 PREDICTION ANALYSIS")

n1 = df[df["actual_stage"] == "N1"]

print("\nActual N1 predictions:")
print(n1["predicted_stage"].value_counts())

print("\nN1 accuracy by participant:")
print(
    n1.groupby("subject")["correct"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nN1 confidence by correctness:")
print(
    n1.groupby("correct")["confidence"]
    .agg(["count", "mean", "median"])
    .round(4)
)

print("\nN1 entropy by correctness:")
print(
    n1.groupby("correct")["entropy"]
    .agg(["mean", "median"])
    .round(4)
)