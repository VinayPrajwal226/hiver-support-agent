import pandas as pd
from pathlib import Path
from datetime import datetime

GOLDEN_PATH = Path("data/golden/golden_set.csv")
CORRECTIONS_PATH = Path("data/golden/golden_corrections.csv")

df = pd.read_csv(GOLDEN_PATH)
corrections = pd.read_csv(CORRECTIONS_PATH)

backup_path = Path(
    "data/golden/golden_set_before_corrections_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)

df.to_csv(backup_path, index=False)

for _, correction in corrections.iterrows():
    tweet_id = str(correction["tweet_id"])

    mask = df["tweet_id"].astype(str) == tweet_id

    if mask.sum() != 1:
        raise ValueError(
            f"Expected exactly one row for tweet_id {tweet_id}, "
            f"found {mask.sum()}"
        )

    current_intent = df.loc[mask, "intent"].iloc[0]
    expected_old = correction["old_intent"]
    new_intent = correction["new_intent"]

    if current_intent != expected_old:
        raise ValueError(
            f"Tweet {tweet_id}: expected old intent "
            f"'{expected_old}', found '{current_intent}'"
        )

    df.loc[mask, "intent"] = new_intent

    old_notes = df.loc[mask, "notes"].iloc[0]

    correction_note = (
        f"Manual audit correction: {expected_old} -> {new_intent}. "
        f"{correction['reason']}"
    )

    if pd.isna(old_notes) or str(old_notes).strip() == "":
        df.loc[mask, "notes"] = correction_note
    else:
        df.loc[mask, "notes"] = (
            str(old_notes) + " | " + correction_note
        )

df.to_csv(GOLDEN_PATH, index=False)

print(f"Backup created: {backup_path}")
print(f"Updated: {GOLDEN_PATH}")
print(f"Corrections applied: {len(corrections)}")

print("\nUpdated labels:")

for _, correction in corrections.iterrows():
    print(
        f"{correction['tweet_id']}: "
        f"{correction['old_intent']} -> "
        f"{correction['new_intent']}"
    )

print("\nGolden set size:", len(df))
print("Human-reviewed:", (df["label_status"] == "human_reviewed").sum())