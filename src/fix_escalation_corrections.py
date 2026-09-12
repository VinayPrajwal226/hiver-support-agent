import pandas as pd
from pathlib import Path
from datetime import datetime

GOLDEN_PATH = Path("data/golden/golden_set.csv")

df = pd.read_csv(GOLDEN_PATH)

backup_path = Path(
    "data/golden/golden_set_before_escalation_corrections_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

df.to_csv(backup_path, index=False)

corrections = {
    "256686": (
        "yes",
        "Payment or billing investigation required"
    ),
    "13144": (
        "yes",
        "Insufficient context to safely resolve"
    ),
    "788630": (
        "no",
        "No action required for general response"
    ),
    "1394458": (
        "no",
        "No action required for general response"
    ),
}

for tweet_id, (escalation, reason) in corrections.items():
    mask = df["tweet_id"].astype(str) == tweet_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for {tweet_id}, found {mask.sum()}"
        )

    df.loc[mask, "escalation"] = escalation
    df.loc[mask, "escalation_reason"] = reason

    old_notes = df.loc[mask, "notes"].iloc[0]

    note = (
        f"Manual audit correction: escalation -> {escalation}. "
        f"Reason: {reason}"
    )

    if pd.isna(old_notes) or str(old_notes).strip() == "":
        df.loc[mask, "notes"] = note
    else:
        df.loc[mask, "notes"] = str(old_notes) + " | " + note

df.to_csv(GOLDEN_PATH, index=False)

print(f"Backup created: {backup_path}")
print(f"Corrections applied: {len(corrections)}")
print(f"Golden set size: {len(df)}")
print(
    "Human-reviewed:",
    (df["label_status"] == "human_reviewed").sum()
)

print("\nUpdated escalation distribution:")
print(df["escalation"].value_counts().to_string())