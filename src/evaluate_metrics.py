import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

PATH = "data/golden/agent_evaluation_leakage_safe.csv"

df = pd.read_csv(PATH)

intent_df = df.dropna(
    subset=["predicted_intent"]
)

intent_accuracy = accuracy_score(
    intent_df["human_intent"],
    intent_df["predicted_intent"]
)

intent_macro_f1 = f1_score(
    intent_df["human_intent"],
    intent_df["predicted_intent"],
    average="macro",
    zero_division=0
)

escalation_df = df.dropna(
    subset=["predicted_escalation"]
).copy()

escalation_df["human_escalation"] = (
    escalation_df["human_escalation"]
    .astype(str)
    .str.lower()
    .map({
        "yes": "yes",
        "no": "no",
        "true": "yes",
        "false": "no"
    })
)

escalation_df["predicted_escalation"] = (
    escalation_df["predicted_escalation"]
    .astype(str)
    .str.lower()
    .map({
        "yes": "yes",
        "no": "no",
        "true": "yes",
        "false": "no"
    })
)

escalation_accuracy = accuracy_score(
    escalation_df["human_escalation"],
    escalation_df["predicted_escalation"]
)

escalation_macro_f1 = f1_score(
    escalation_df["human_escalation"],
    escalation_df["predicted_escalation"],
    average="macro",
    zero_division=0
)

print("LEAKAGE-SAFE AGENT EVALUATION")
print("============================")

print()
print("Examples:", len(df))

print()
print("INTENT")
print("------")
print("Accuracy:", round(intent_accuracy, 4))
print("Macro F1:", round(intent_macro_f1, 4))

print()
print(classification_report(
    intent_df["human_intent"],
    intent_df["predicted_intent"],
    zero_division=0
))

print()
print("ESCALATION")
print("----------")
print("Accuracy:", round(escalation_accuracy, 4))
print("Macro F1:", round(escalation_macro_f1, 4))

print()
print(classification_report(
    escalation_df["human_escalation"],
    escalation_df["predicted_escalation"],
    zero_division=0
))

print()
print("RETRIEVAL")
print("---------")
print(
    "Average top-1 similarity:",
    round(df["top_similarity"].mean(), 4)
)