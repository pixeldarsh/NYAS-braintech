import pandas as pd

df = pd.read_csv("oof_predictions.csv")

scores = {
    "Confidence": df["confidence"],
    "Margin": df["margin"],
    "Entropy": -df["entropy"]
}

print("\nSTAGE-WISE SELECTIVE PREDICTION")

for stage in df["actual_stage"].unique():

    subset = df[df["actual_stage"] == stage].copy()

    print(f"\n{'=' * 40}")
    print(f"STAGE: {stage}")
    print(f"{'=' * 40}")

    baseline = subset["correct"].mean()
    print(f"Baseline accuracy: {baseline:.4f}")

    for name, score in scores.items():

        subset["score"] = score.loc[subset.index]

        threshold = subset["score"].quantile(0.20)

        retained = subset[subset["score"] >= threshold]

        print(
            f"{name}: "
            f"Coverage = {len(retained)/len(subset):.1%}, "
            f"Accuracy = {retained['correct'].mean():.4f}"
        )