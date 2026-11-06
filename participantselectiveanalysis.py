import pandas as pd

df = pd.read_csv("oof_predictions.csv")

uncertainty_scores = {
    "Confidence": df["confidence"],
    "Margin": df["margin"],
    "Entropy": -df["entropy"]
}

print("\nPARTICIPANT-WISE SELECTIVE PREDICTION")

for subject in df["subject"].unique():

    participant = df[df["subject"] == subject].copy()

    print(f"\n{'=' * 45}")
    print(f"PARTICIPANT: {subject}")
    print(f"{'=' * 45}")

    baseline = participant["correct"].mean()
    print(f"Baseline accuracy: {baseline:.4f}")

    for name, scores in uncertainty_scores.items():

        participant["score"] = scores.loc[participant.index]

        threshold = participant["score"].quantile(0.20)

        retained = participant[
            participant["score"] >= threshold
        ]

        print(
            f"{name}: "
            f"Coverage = {len(retained)/len(participant):.1%}, "
            f"Accuracy = {retained['correct'].mean():.4f}"
        )