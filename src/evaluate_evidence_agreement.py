import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PATH = "data/golden/evidence_llm_judgment_50.csv"

df = pd.read_csv(PATH)

human = df["human_evidence_supported"].astype(str).str.lower().str.strip()
llm = df["llm_evidence_supported"].astype(str).str.lower().str.strip()

agreement = accuracy_score(human, llm)

print("HUMAN vs LLM EVIDENCE JUDGE")
print("===========================")
print()
print("Examples:", len(df))
print("Agreement:", round(agreement, 4))
print("Agreement %:", round(agreement * 100, 2))
print()

print("CONFUSION MATRIX")
print("----------------")
print(
    confusion_matrix(
        human,
        llm,
        labels=["yes", "no"]
    )
)

print()
print("CLASSIFICATION REPORT")
print("---------------------")
print(
    classification_report(
        human,
        llm,
        labels=["yes", "no"],
        zero_division=0
    )
)

df["agreement"] = human == llm

print()
print("AGREEMENTS:", df["agreement"].sum())
print("DISAGREEMENTS:", (~df["agreement"]).sum())