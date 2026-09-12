import pandas as pd
import os

INPUT_PATH = "data/amazon/amazon_context.csv"
OUTPUT_PATH = "data/golden/golden_candidates.csv"

df = pd.read_csv(INPUT_PATH)

df = df.drop_duplicates(subset=["text"])

df = df.sample(
    n=min(250, len(df)),
    random_state=42
)

df = df[
    [
        "tweet_id",
        "parent_text",
        "text"
    ]
]

df = df.rename(
    columns={
        "text": "customer_text"
    }
)

df["intent"] = ""
df["escalation"] = ""
df["escalation_reason"] = ""

os.makedirs("data/golden", exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("Golden candidates:", len(df))
print("Saved to:", OUTPUT_PATH)