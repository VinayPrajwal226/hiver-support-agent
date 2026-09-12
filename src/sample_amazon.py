import pandas as pd
import os

DATA_PATH = "data/amazon/amazon_customer_messages.csv"
OUTPUT_PATH = "data/amazon/amazon_sample_2000.csv"

df = pd.read_csv(DATA_PATH)

# Remove exact duplicate messages
df = df.drop_duplicates(subset=["text"])

# Use a fixed random seed so the sample is reproducible
sample = df.sample(
    n=min(2000, len(df)),
    random_state=42
)

sample = sample[
    [
        "tweet_id",
        "author_id",
        "created_at",
        "text",
        "parent_tweet_id"
    ]
]

sample.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Original messages: {len(df):,}")
print(f"After removing duplicates: {len(df):,}")
print(f"Sample created: {len(sample):,}")
print(f"Saved to: {OUTPUT_PATH}")