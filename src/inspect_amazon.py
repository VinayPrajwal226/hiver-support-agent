import pandas as pd

DATA_PATH = "data/twcs/twcs.csv"

brand = "AmazonHelp"

print("Finding Amazon customer conversations...\n")

brand_tweet_ids = set()

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["tweet_id", "author_id"],
    chunksize=100000
):
    rows = chunk[chunk["author_id"] == brand]
    brand_tweet_ids.update(rows["tweet_id"].astype(int))

print(f"Amazon support tweets found: {len(brand_tweet_ids):,}")

examples = []

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

    customer_rows["parent_id"] = (
        customer_rows["in_response_to_tweet_id"].astype(int)
    )

    matches = customer_rows[
        customer_rows["parent_id"].isin(brand_tweet_ids)
    ]

    if not matches.empty:
        examples.extend(
            matches[
                [
                    "tweet_id",
                    "author_id",
                    "created_at",
                    "text",
                    "in_response_to_tweet_id"
                ]
            ].to_dict("records")
        )

    if len(examples) >= 100:
        break

print(f"\nCollected {len(examples)} examples.\n")

print("=" * 100)

for i, example in enumerate(examples[:100], 1):
    print(f"\nEXAMPLE {i}")
    print(f"Tweet ID: {example['tweet_id']}")
    print(f"Customer: {example['author_id']}")
    print(f"Time: {example['created_at']}")
    print(f"Message: {example['text']}")
    print(f"Responding to: {example['in_response_to_tweet_id']}")
    print("-" * 100)