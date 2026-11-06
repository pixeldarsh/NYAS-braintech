import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("oof_predictions.csv")

# Incorrect predictions are the positive class
y_error = (~df["correct"].astype(bool)).astype(int)

# Higher values should indicate greater uncertainty
scores = {
    "Confidence": -df["confidence"],
    "Margin": -df["margin"],
    "Entropy": df["entropy"]
}

print("\nERROR DETECTION PERFORMANCE")

for name, score in scores.items():
    auc = roc_auc_score(y_error, score)
    print(f"{name}: ROC-AUC = {auc:.4f}")