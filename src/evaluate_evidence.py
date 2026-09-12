import pandas as pd

RETRIEVAL_PATH = "data/amazon/retrieval_evaluation_leakage_safe.csv"
OUTPUT_PATH = "data/amazon/evidence_evaluation.csv"

df = pd.read_csv(RETRIEVAL_PATH)

top1 = df[df["rank"] == 1].copy()

def intent_compatible(golden_intent, historical_text, historical_reply):
    text = (
        str(historical_text) + " " +
        str(historical_reply)
    ).lower()

    if golden_intent == "delivery_issue":
        keywords = [
            "delivery",
            "delivered",
            "carrier",
            "tracking",
            "package",
            "shipment",
            "arrive",
            "arrived",
            "shipping"
        ]

    elif golden_intent == "order_status":
        keywords = [
            "order",
            "tracking",
            "delivery",
            "status",
            "arrive",
            "shipped"
        ]

    elif golden_intent == "refund_issue":
        keywords = [
            "refund",
            "money",
            "credited",
            "credit",
            "reimburse"
        ]

    elif golden_intent == "return_replacement":
        keywords = [
            "return",
            "replacement",
            "replace",
            "exchange"
        ]

    elif golden_intent == "cancel_order":
        keywords = [
            "cancel",
            "cancellation"
        ]

    elif golden_intent == "account_login":
        keywords = [
            "account",
            "login",
            "sign in",
            "password",
            "verification"
        ]

    elif golden_intent == "payment_issue":
        keywords = [
            "payment",
            "card",
            "charge",
            "charged",
            "billing"
        ]

    elif golden_intent == "product_issue":
        keywords = [
            "broken",
            "damaged",
            "defective",
            "not working",
            "product"
        ]

    else:
        return True

    return any(keyword in text for keyword in keywords)


top1["intent_compatible"] = top1.apply(
    lambda row: intent_compatible(
        row["golden_intent"],
        row["historical_customer_text"],
        row["historical_brand_reply"]
    ),
    axis=1
)

top1["strong_similarity"] = (
    top1["similarity"] >= 0.40
)

top1["usable_evidence"] = (
    top1["strong_similarity"] &
    top1["intent_compatible"]
)

print("EVIDENCE QUALITY EVALUATION")
print("===========================")
print("Golden examples:", len(top1))
print()

print(
    "Similarity >= 0.40:",
    top1["strong_similarity"].sum(),
    "/",
    len(top1)
)

print(
    "Intent compatible:",
    top1["intent_compatible"].sum(),
    "/",
    len(top1)
)

print(
    "Usable evidence:",
    top1["usable_evidence"].sum(),
    "/",
    len(top1)
)

print(
    "Usable evidence rate:",
    round(
        top1["usable_evidence"].mean(),
        4
    )
)

print()
print("By intent:")
print(
    top1.groupby("golden_intent")["usable_evidence"]
    .agg(["count", "sum", "mean"])
    .sort_values("mean")
)

top1.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("Saved to:", OUTPUT_PATH)