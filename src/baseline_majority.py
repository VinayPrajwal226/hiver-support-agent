import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_PATH = "data/golden/golden_set.csv"

df = pd.read_csv(GOLDEN_PATH)

y_true = df["intent"].astype(str)

majority_intent = y_true.value_counts().idxmax()
y_pred = [majority_intent] * len(y_true)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

print("MAJORITY CLASS BASELINE")
print("=======================")
print("Majority intent:", majority_intent)
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print()
print(classification_report(y_true, y_pred, zero_division=0))