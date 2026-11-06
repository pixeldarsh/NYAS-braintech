
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score

# Load features
df = pd.read_csv("eeg_features.csv")

feature_cols = [
    "delta_absolute", "delta_relative",
    "theta_absolute", "theta_relative",
    "alpha_absolute", "alpha_relative",
    "sigma_absolute", "sigma_relative",
    "beta_absolute", "beta_relative"
]

X = df[feature_cols]
y = df["sleep_stage"]
groups = df["subject"]

stage_order = ["Wake", "N1", "N2", "N3", "REM"]

logo = LeaveOneGroupOut()
all_results = []

for fold, (train_idx, test_idx) in enumerate(
    logo.split(X, y, groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    test_subjects = groups.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Map probabilities to their corresponding sleep stages
    probability_df = pd.DataFrame(
        probabilities,
        columns=model.classes_,
        index=X_test.index
    ).reindex(columns=stage_order, fill_value=0)

    # Highest predicted probability
    max_probability = probability_df.max(axis=1)

    # Difference between the two highest probabilities
    sorted_probabilities = np.sort(
        probability_df.to_numpy(), axis=1
    )

    probability_margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    # Entropy: higher values indicate a more spread-out
    # probability distribution
    p = probability_df.to_numpy()
    from scipy.special import xlogy

    entropy = -np.sum(xlogy(p, p), axis=1)

    fold_results = probability_df.copy()

    fold_results["subject"] = test_subjects.values
    fold_results["actual_stage"] = y_test.values
    fold_results["predicted_stage"] = predictions
    fold_results["confidence"] = max_probability.values
    fold_results["margin"] = probability_margin
    fold_results["entropy"] = entropy
    fold_results["correct"] = (
        y_test.values == predictions
    )

    all_results.append(fold_results)

    print(f"\nFold {fold}: {test_subjects.iloc[0]}")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Macro F1:", round(
        f1_score(y_test, predictions, average="macro"), 4
    ))

# Combine all held-out predictions
results = pd.concat(all_results).sort_index()

results.to_csv("oof_predictions.csv", index=False)

print("\n========== OVERALL RESULTS ==========")
print("Accuracy:", round(results["correct"].mean(), 4))

print("Macro F1:", round(
    f1_score(
        results["actual_stage"],
        results["predicted_stage"],
        average="macro"
    ), 4
))

print("\nPREDICTIONS SAVED TO oof_predictions.csv")

print("\nAVERAGE CONFIDENCE")
print(round(results["confidence"].mean(), 4))

print("\nACCURACY BY CONFIDENCE LEVEL")
results["confidence_group"] = pd.cut(
    results["confidence"],
    bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.001],
    include_lowest=True
)

print(
    results.groupby("confidence_group", observed=False)
    ["correct"]
    .agg(["count", "mean"])
    .round(4)
)

print("\nACCURACY BY SLEEP STAGE")
print(
    results.groupby("actual_stage")["correct"]
    .agg(["count", "mean"])
    .round(4)
)
print("All entropy values finite:",
      np.isfinite(entropy).all())
