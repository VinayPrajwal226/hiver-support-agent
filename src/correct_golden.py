import pandas as pd
from pathlib import Path

GOLDEN_PATH = Path("data/golden/golden_set.csv")
CORRECTIONS_PATH = Path("data/golden/golden_corrections.csv")

df = pd.read_csv(GOLDEN_PATH)

corrections = [
    {
        "tweet_id": "1394458",
        "old_intent": "delivery_issue",
        "new_intent": "non_action_or_context",
        "reason": "Customer is answering a clarification question about a season; there is no delivery issue."
    },
    {
        "tweet_id": "256686",
        "old_intent": "account_login",
        "new_intent": "payment_issue",
        "reason": "Customer is asking about cashback transfer/payment."
    },
    {
        "tweet_id": "788630",
        "old_intent": "product_issue",
        "new_intent": "non_action_or_context",
        "reason": "Customer is expressing anticipation about an Echo shipment, not reporting a product defect."
    },
    {
        "tweet_id": "13144",
        "old_intent": "delivery_issue",
        "new_intent": "non_action_or_context",
        "reason": "Customer is complaining about repeated template replies; the current message does not describe a delivery problem."
    },
]

corrections_df = pd.DataFrame(corrections)

print("Proposed corrections:")
print(corrections_df.to_string(index=False))

print("\nChecking current values...")

for _, row in corrections_df.iterrows():
    matches = df[df["tweet_id"].astype(str) == row["tweet_id"]]

    if len(matches) != 1:
        print(f"WARNING: tweet_id {row['tweet_id']} found {len(matches)} times")
        continue

    current = matches.iloc[0]["intent"]

    if current != row["old_intent"]:
        print(
            f"WARNING: {row['tweet_id']} currently has "
            f"'{current}', expected '{row['old_intent']}'"
        )
    else:
        print(
            f"OK: {row['tweet_id']} "
            f"{row['old_intent']} -> {row['new_intent']}"
        )

corrections_df.to_csv(CORRECTIONS_PATH, index=False)

print(f"\nSaved proposed corrections to: {CORRECTIONS_PATH}")
print("\nNO changes were made to golden_set.csv.")