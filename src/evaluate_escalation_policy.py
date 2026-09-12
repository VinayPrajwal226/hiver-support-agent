import pandas as pd
from escalation_policy import should_escalate
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_PATH = "data/golden/golden_set.csv"
EVAL_PATH = "data/golden/agent_evaluation.csv"

golden_df = pd.read_csv(GOLDEN_PATH)
eval_df = pd.read_csv(EVAL_PATH)

df = golden_df.merge(
    eval_df[["tweet_id", "top_similarity"]],
    on="tweet_id",
    how="left"
)

predictions = []

for _, row in df.iterrows():
    escalate, reason = should_escalate(
        row["customer_text"],
        row["intent"],
        row["top_similarity"]
    )

    predictions.append(escalate)

y_true = df["escalation"].astype(str).str.lower()

y_pred = pd.Series(predictions).map({
    True: "yes",
    False: "no"
})

print("ESCALATION POLICY EVALUATION")
print("============================")
print("Examples:", len(df))
print()
print("Accuracy:", round(
    accuracy_score(y_true, y_pred), 4
))
print("Macro F1:", round(
    f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    ), 4
))
print()
print(classification_report(
    y_true,
    y_pred,
    zero_division=0
))