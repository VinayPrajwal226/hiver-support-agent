import pandas as pd
import os

INPUT_PATH = "data/twcs/twcs.csv"
AMAZON_PATH = "data/amazon/amazon_customer_messages.csv"
OUTPUT_PATH = "data/amazon/amazon_context.csv"

amazon = pd.read_csv(AMAZON_PATH)

amazon["parent_tweet_id"] = amazon["parent_tweet_id"].astype("Int64")

parent_ids = set(amazon["parent_tweet_id"].dropna().astype(str))

chunks = []

for chunk in pd.read_csv(INPUT_PATH, chunksize=100000):
    chunk["tweet_id"] = chunk["tweet_id"].astype(str)

    matches = chunk[chunk["tweet_id"].isin(parent_ids)]

    if len(matches) > 0:
        chunks.append(matches)

parents = pd.concat(chunks, ignore_index=True)

parents["tweet_id"] = parents["tweet_id"].astype(str)

amazon["parent_tweet_id_str"] = amazon["parent_tweet_id"].astype(str)

parents = parents[
    ["tweet_id", "author_id", "inbound", "text"]
].rename(
    columns={
        "tweet_id": "parent_tweet_id_str",
        "author_id": "parent_author_id",
        "inbound": "parent_inbound",
        "text": "parent_text"
    }
)

result = amazon.merge(
    parents,
    on="parent_tweet_id_str",
    how="left"
)

result = result.drop(columns=["parent_tweet_id_str"])

os.makedirs("data/amazon", exist_ok=True)

result.to_csv(OUTPUT_PATH, index=False)

print("Customer messages:", len(amazon))
print("Messages with context:", result["parent_text"].notna().sum())
print("Messages without context:", result["parent_text"].isna().sum())
print("Saved to:", OUTPUT_PATH)