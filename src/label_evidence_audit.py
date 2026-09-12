import pandas as pd
import os

INPUT_PATH = "data/golden/evidence_human_audit_50.csv"

df = pd.read_csv(INPUT_PATH)

df["human_evidence_supported"] = df["human_evidence_supported"].fillna("").astype(str)
df["human_reason"] = df["human_reason"].fillna("").astype(str)

for i in range(len(df)):
    if df.loc[i, "human_evidence_supported"].strip().lower() in ["yes", "no"]:
        continue

    print()
    print("=" * 80)
    print(f"CASE {i + 1} / {len(df)}")
    print("=" * 80)

    print()
    print("CUSTOMER MESSAGE:")
    print(df.loc[i, "golden_text"])

    print()
    print("HUMAN INTENT:")
    print(df.loc[i, "golden_intent"])

    print()
    print("PREDICTED INTENT:")
    print(df.loc[i, "predicted_intent"])

    print()
    print("PREDICTED ESCALATION:")
    print(df.loc[i, "predicted_escalation"])

    print()
    print("RETRIEVED HISTORICAL CUSTOMER:")
    print(df.loc[i, "historical_customer_text"])

    print()
    print("RETRIEVED HISTORICAL AMAZON RESPONSE:")
    print(df.loc[i, "historical_brand_reply"])

    print()
    print("SIMILARITY:")
    print(df.loc[i, "similarity"])

    print()
    print("AGENT DRAFT REPLY:")
    print(df.loc[i, "draft_reply"])

    print()
    print("AGENT REASON:")
    print(df.loc[i, "reason"])

    while True:
        answer = input(
            "\nDoes the historical evidence sufficiently support "
            "the agent's handling? (y/n): "
        ).strip().lower()

        if answer in ["y", "n"]:
            break

        print("Please enter y or n.")

    if answer == "y":
        df.loc[i, "human_evidence_supported"] = "yes"
    else:
        df.loc[i, "human_evidence_supported"] = "no"

    reason = input(
        "Short reason (optional): "
    ).strip()

    df.loc[i, "human_reason"] = reason

    df.to_csv(INPUT_PATH, index=False)

    print()
    print("Saved.")

print()
print("=" * 80)
print("EVIDENCE HUMAN AUDIT COMPLETE")
print("=" * 80)

print(
    df["human_evidence_supported"]
    .value_counts()
)