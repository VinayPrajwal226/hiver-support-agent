import pandas as pd

INPUT_PATH = "data/twcs/twcs.csv"
OUTPUT_PATH = "data/amazon/amazon_response_pairs.csv"


def clean_id(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


amazon_replies = []
customer_ids = set()

for chunk in pd.read_csv(INPUT_PATH, chunksize=100000):

    chunk["tweet_id"] = chunk["tweet_id"].apply(clean_id)
    chunk["in_response_to_tweet_id"] = (
        chunk["in_response_to_tweet_id"].apply(clean_id)
    )

    amazon = chunk[
        chunk["author_id"].astype(str) == "AmazonHelp"
    ]

    if len(amazon) > 0:

        amazon_replies.append(
            amazon[
                [
                    "tweet_id",
                    "created_at",
                    "text",
                    "in_response_to_tweet_id"
                ]
            ]
        )

        customer_ids.update(
            amazon["in_response_to_tweet_id"]
            [amazon["in_response_to_tweet_id"] != ""]
            .tolist()
        )


amazon_replies = pd.concat(
    amazon_replies,
    ignore_index=True
)


customer_rows = []

for chunk in pd.read_csv(INPUT_PATH, chunksize=100000):

    chunk["tweet_id"] = chunk["tweet_id"].apply(clean_id)

    matches = chunk[
        (chunk["inbound"] == True) &
        (chunk["tweet_id"].isin(customer_ids))
    ]

    if len(matches) > 0:

        customer_rows.append(
            matches[
                [
                    "tweet_id",
                    "created_at",
                    "text"
                ]
            ]
        )


if not customer_rows:
    print("No customer tweets matched Amazon responses.")
    print("Amazon replies:", len(amazon_replies))
    print("Customer IDs:", len(customer_ids))
    raise SystemExit


customer_rows = pd.concat(
    customer_rows,
    ignore_index=True
)


customer_lookup = customer_rows.set_index("tweet_id")


pairs = []

for _, reply in amazon_replies.iterrows():

    customer_id = reply["in_response_to_tweet_id"]

    if customer_id in customer_lookup.index:

        customer = customer_lookup.loc[customer_id]

        pairs.append(
            {
                "customer_tweet_id": customer_id,
                "customer_text": customer["text"],
                "customer_created_at": customer["created_at"],
                "brand_reply_id": reply["tweet_id"],
                "brand_reply_text": reply["text"],
                "brand_reply_created_at": reply["created_at"]
            }
        )


pairs_df = pd.DataFrame(pairs)


if len(pairs_df) == 0:

    print("No response pairs found.")
    print("Amazon replies:", len(amazon_replies))
    print("Customer IDs:", len(customer_ids))
    raise SystemExit


pairs_df = pairs_df.drop_duplicates(
    subset=[
        "customer_tweet_id",
        "brand_reply_id"
    ]
)


pairs_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("AMAZON RESPONSE PAIRS")
print("=====================")
print("Amazon replies:", len(amazon_replies))
print("Customer IDs:", len(customer_ids))
print("Response pairs:", len(pairs_df))
print(
    "Unique customers:",
    pairs_df["customer_tweet_id"].nunique()
)
print(
    "Unique Amazon replies:",
    pairs_df["brand_reply_id"].nunique()
)
print()
print(
    pairs_df.head(5).to_string(index=False)
)
print()
print("Saved to:", OUTPUT_PATH)