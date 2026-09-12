import pandas as pd
import os

DATA_PATH = "data/twcs/twcs.csv"
OUTPUT_PATH = "data/amazon/amazon_customer_messages.csv"

BRAND = "AmazonHelp"

os.makedirs("data/amazon", exist_ok=True)

print("Pass 1: Collecting Amazon support tweet IDs...")

brand_tweet_ids = set()

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["tweet_id", "author_id"],
    chunksize=100000
):
    amazon_rows = chunk[chunk["author_id"] == BRAND]

    brand_tweet_ids.update(
        amazon_rows["tweet_id"].astype(int)
    )

print(f"Amazon support tweets: {len(brand_tweet_ids):,}")

print("\nPass 2: Extracting customer messages...")

output_rows = []

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id"
    ],
    chunksize=100000
):
    customer_rows = chunk[
        (chunk["inbound"] == True) &
        (chunk["in_response_to_tweet_id"].notna())
    ].copy()

    if customer_rows.empty:
        continue

    customer_rows["parent_tweet_id"] = (
        customer_rows["in_response_to_tweet_id"].astype(int)
    )

    amazon_rows = customer_rows[
        customer_rows["parent_tweet_id"].isin(brand_tweet_ids)
    ]

    if not amazon_rows.empty:
        output_rows.append(
            amazon_rows[
                [
                    "tweet_id",
                    "author_id",
                    "created_at",
                    "text",
                    "parent_tweet_id"
                ]
            ]
        )

if output_rows:
    amazon_df = pd.concat(output_rows, ignore_index=True)

    amazon_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nExtraction complete!")
    print(f"Customer messages: {len(amazon_df):,}")
    print(f"Saved to: {OUTPUT_PATH}")
else:
    print("No Amazon customer messages found.")