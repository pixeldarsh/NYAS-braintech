import pandas as pd

df = pd.read_csv("oof_predictions.csv")

# Higher score = more certainty
scores = {
    "Confidence": df["confidence"],
    "Margin": df["margin"],
    "Entropy": -df["entropy"]
}

coverages = [1.0, 0.9, 0.8, 0.7, 0.5]

print("\nSELECTIVE PREDICTION RESULTS")

for name, score in scores.items():
    print(f"\n{name}")

    results = df[["correct"]].copy()
    results["score"] = score

    for coverage in coverages:
        threshold = results["score"].quantile(1 - coverage)

        retained = results[results["score"] >= threshold]

        actual_coverage = len(retained) / len(df)
        accuracy = retained["correct"].mean()

        print(
            f"Coverage: {actual_coverage:.1%} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Errors: {(~retained['correct']).sum()}"
        )