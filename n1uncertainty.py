import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("oof_predictions.csv")

n1 = df[df["actual_stage"] == "N1"].copy()

# Incorrect N1 predictions are positive
y_error = (~n1["correct"].astype(bool)).astype(int)

scores = {
    "Confidence": -n1["confidence"],
    "Margin": -n1["margin"],
    "Entropy": n1["entropy"]
}

print("\nN1 ERROR DETECTION ROC-AUC")

for name, score in scores.items():
    auc = roc_auc_score(y_error, score)
    print(f"{name}: {auc:.4f}")