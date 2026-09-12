import pandas as pd
from pathlib import Path
from datetime import datetime

GOLDEN_PATH = Path("data/golden/golden_set.csv")

df = pd.read_csv(GOLDEN_PATH)

backup = Path(
    "data/golden/golden_set_before_final_audit_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)
df.to_csv(backup, index=False)

intent_corrections = {'2600217': 'delivery_issue', '473363': 'delivery_issue', '1047278': 'order_status', '379933': 'delivery_issue', '2766377': 'delivery_issue', '1373383': 'delivery_issue', '816326': 'delivery_issue', '1530923': 'product_issue', '2448297': 'delivery_issue', '2433635': 'delivery_issue', '1103631': 'payment_issue', '343850': 'delivery_issue', '2242570': 'delivery_issue', '1395566': 'order_status', '1110383': 'refund_issue', '2343971': 'delivery_issue', '1091280': 'delivery_issue', '1986995': 'delivery_issue', '963233': 'delivery_issue', '153602': 'delivery_issue', '958773': 'account_login', '2508105': 'delivery_issue', '2205209': 'order_status', '962050': 'delivery_issue', '741555': 'delivery_issue', '581173': 'order_status', '1714198': 'non_action_or_context', '1338511': 'account_login', '2850759': 'non_action_or_context', '1013777': 'payment_issue', '1615564': 'refund_issue', '2200651': 'delivery_issue', '2879492': 'account_login', '500676': 'payment_issue'}

escalation_no = {'2616435': 'No action required for acknowledgement or feedback', '1317506': 'No action required; customer confirms resolution', '425081': 'No action required; customer confirms issue is being addressed', '409343': 'No action required; customer confirms resolution', '442841': 'No action required; customer confirms claims are settled', '2850759': 'No action required; customer confirms refund was processed', '2909202': 'No action required; customer confirms delivery was resolved', '1539867': 'No action required; customer confirms issue is solved', '312132': 'No action required; customer confirms issue was resolved', '1522574': 'No action required; customer confirms they followed up by email', '2393517': 'No action required; customer confirms they contacted the seller', '2501916': 'No action required; customer is thanking support', '229969': 'No action required; customer is thanking support', '2115518': 'No action required; customer is thanking support', '38561': 'No action required; conversational acknowledgement', '2441720': 'No action required; customer confirms they will follow the instructions'}

for tweet_id, new_intent in intent_corrections.items():
    mask = df["tweet_id"].astype(str) == tweet_id
    if mask.sum() != 1:
        raise ValueError(f"Expected one row for {tweet_id}, found {mask.sum()}")

    old = df.loc[mask, "intent"].iloc[0]
    if old != new_intent:
        df.loc[mask, "intent"] = new_intent
        old_notes = df.loc[mask, "notes"].iloc[0]
        note = f"Final semantic audit: intent {old} -> {new_intent}"
        df.loc[mask, "notes"] = note if pd.isna(old_notes) or not str(old_notes).strip() else str(old_notes) + " | " + note

for tweet_id, reason in escalation_no.items():
    mask = df["tweet_id"].astype(str) == tweet_id
    if mask.sum() != 1:
        raise ValueError(f"Expected one row for {tweet_id}, found {mask.sum()}")

    df.loc[mask, "escalation"] = "no"
    df.loc[mask, "escalation_reason"] = reason

    old_notes = df.loc[mask, "notes"].iloc[0]
    note = f"Final semantic audit: escalation -> no. {reason}"
    df.loc[mask, "notes"] = note if pd.isna(old_notes) or not str(old_notes).strip() else str(old_notes) + " | " + note

df.to_csv(GOLDEN_PATH, index=False)

assert len(df) == 250
assert df["tweet_id"].astype(str).is_unique
assert (df["label_status"] == "human_reviewed").sum() == 250
assert set(df["intent"].dropna()).issubset({
    "order_status", "delivery_issue", "refund_issue",
    "return_replacement", "cancel_order", "account_login",
    "payment_issue", "product_issue", "non_action_or_context"
})
assert set(df["escalation"].dropna()).issubset({"yes", "no"})

print("FINAL GOLDEN SET AUDIT")
print("=" * 70)
print("Backup:", backup)
print("Rows:", len(df))
print("Human reviewed:", (df["label_status"] == "human_reviewed").sum())
print("Unique tweet IDs:", df["tweet_id"].astype(str).nunique())
print("\nIntent distribution:")
print(df["intent"].value_counts().to_string())
print("\nEscalation distribution:")
print(df["escalation"].value_counts().to_string())
print("\nValidation: PASSED")
