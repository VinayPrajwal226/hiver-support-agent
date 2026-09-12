import pandas as pd
from retrieve_responses import retrieve_responses

GOLDEN_PATH = "data/golden/golden_set.csv"
OUTPUT_PATH = "data/amazon/retrieval_evaluation_leakage_safe.csv"

golden = pd.read_csv(GOLDEN_PATH)

results = []

for _, row in golden.iterrows():

    query = row["customer_text"]
    tweet_id = str(row["tweet_id"])

    retrieved = retrieve_responses(
        query,
        top_k=3,
        exclude_customer_text=query
    )

    for rank, (_, result) in enumerate(
        retrieved.iterrows(),
        start=1
    ):

        results.append({
            "golden_tweet_id": tweet_id,
            "golden_text": query,
            "golden_intent": row["intent"],
            "rank": rank,
            "similarity": result["similarity"],
            "historical_customer_text": result["customer_text"],
            "historical_brand_reply": result["brand_reply_text"]
        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

top1 = results_df[
    results_df["rank"] == 1
]

print("LEAKAGE-SAFE RETRIEVAL EVALUATION")
print("=================================")
print("Golden examples:", len(golden))
print("Retrieved examples:", len(results_df))
print("Top results per example: 3")
print()
print(
    "Average top-1 similarity:",
    round(top1["similarity"].mean(), 4)
)
print(
    "Average top-3 similarity:",
    round(results_df["similarity"].mean(), 4)
)
print()
print("Saved to:", OUTPUT_PATH)