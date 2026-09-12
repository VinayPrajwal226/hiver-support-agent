import pandas as pd
from collections import Counter

DATA_PATH = "data/twcs/twcs.csv"

candidate_brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "XboxSupport"
]

brand_tweet_ids = {brand: set() for brand in candidate_brands}

print("Pass 1: Finding brand tweets...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["tweet_id", "author_id"],
    chunksize=100000
):
    for brand in candidate_brands:
        rows = chunk[chunk["author_id"] == brand]

        if not rows.empty:
            brand_tweet_ids[brand].update(rows["tweet_id"].astype(int))

print("\nBrand tweet IDs collected.")

customer_counts = Counter()

print("Pass 2: Finding customer messages responding to brands...")

for chunk in pd.read_csv(
    DATA_PATH,
    usecols=["author_id", "inbound", "in_response_to_tweet_id"],
    chunksize=100000
):
    inbound = chunk[
        (chunk["inbound"] == True) &
        (chunk["in_response_to_tweet_id"].notna())
    ]

    if inbound.empty:
        continue

    parent_ids = set(inbound["in_response_to_tweet_id"].astype(int))

    for brand in candidate_brands:
        matching = parent_ids.intersection(brand_tweet_ids[brand])

        if matching:
            customer_counts[brand] += inbound[
                inbound["in_response_to_tweet_id"].astype(int).isin(matching)
            ].shape[0]

print("\nCustomer messages associated with each brand")
print("=" * 70)

for brand, count in customer_counts.most_common():
    print(f"{brand:25} {count:>12,}")