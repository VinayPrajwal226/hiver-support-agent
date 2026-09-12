import pandas as pd

RETRIEVAL_PATH = "data/amazon/retrieval_evaluation_leakage_safe.csv"
AGENT_PATH = "data/golden/agent_evaluation_leakage_safe.csv"
OUTPUT_PATH = "data/golden/evidence_human_audit_50.csv"

retrieval_df = pd.read_csv(RETRIEVAL_PATH)
agent_df = pd.read_csv(AGENT_PATH)

top1 = retrieval_df[retrieval_df["rank"] == 1].copy()

agent_df["tweet_id"] = agent_df["tweet_id"].astype(str)
top1["golden_tweet_id"] = top1["golden_tweet_id"].astype(str)

merged = top1.merge(
    agent_df[
        [
            "tweet_id",
            "predicted_intent",
            "predicted_escalation",
            "draft_reply",
            "reason"
        ]
    ],
    left_on="golden_tweet_id",
    right_on="tweet_id",
    how="inner"
)

sample = merged.sample(
    n=50,
    random_state=42
).copy()

sample["human_evidence_supported"] = ""
sample["human_reason"] = ""

sample = sample[
    [
        "golden_tweet_id",
        "golden_text",
        "golden_intent",
        "predicted_intent",
        "predicted_escalation",
        "similarity",
        "historical_customer_text",
        "historical_brand_reply",
        "draft_reply",
        "reason",
        "human_evidence_supported",
        "human_reason"
    ]
]

sample.to_csv(
    OUTPUT_PATH,
    index=False
)

print("EVIDENCE HUMAN AUDIT SET")
print("=========================")
print("Examples:", len(sample))
print("Saved to:", OUTPUT_PATH)