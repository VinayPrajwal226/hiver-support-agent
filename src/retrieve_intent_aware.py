import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/amazon/amazon_response_pairs.csv"


def assign_intent(text):
    text = str(text).lower()

    if any(word in text for word in [
        "refund",
        "money back",
        "refunded"
    ]):
        return "refund_issue"

    if any(word in text for word in [
        "return",
        "replacement",
        "replace",
        "exchange"
    ]):
        return "return_replacement"

    if any(word in text for word in [
        "cancel",
        "cancellation"
    ]):
        return "cancel_order"

    if any(word in text for word in [
        "login",
        "log in",
        "password",
        "account",
        "verification code"
    ]):
        return "account_login"

    if any(word in text for word in [
        "charge",
        "charged",
        "payment",
        "billing",
        "card"
    ]):
        return "payment_issue"

    if any(word in text for word in [
        "broken",
        "damaged",
        "defective",
        "not working"
    ]):
        return "product_issue"

    if any(word in text for word in [
        "delayed",
        "late",
        "missing package",
        "not delivered",
        "delivery"
    ]):
        return "delivery_issue"

    if any(word in text for word in [
        "order status",
        "where is my order",
        "track my order"
    ]):
        return "order_status"

    return "non_action_or_context"


df = pd.read_csv(DATA_PATH)

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
)

df["brand_reply_text"] = (
    df["brand_reply_text"]
    .fillna("")
    .astype(str)
)

df["intent"] = df["customer_text"].apply(assign_intent)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

matrix = vectorizer.fit_transform(
    df["customer_text"]
)


def retrieve_responses(
    query,
    predicted_intent=None,
    top_k=3
):

    query_intent = (
        predicted_intent
        if predicted_intent
        else assign_intent(query)
    )

    candidate_df = df[
        df["intent"] == query_intent
    ].copy()

    if len(candidate_df) == 0:
        candidate_df = df.copy()

    candidate_indices = candidate_df.index

    query_vector = vectorizer.transform([query])

    candidate_matrix = matrix[candidate_indices]

    similarities = cosine_similarity(
        query_vector,
        candidate_matrix
    ).flatten()

    top_positions = similarities.argsort()[
        -top_k:
    ][::-1]

    results = candidate_df.iloc[
        top_positions
    ].copy()

    results["similarity"] = similarities[
        top_positions
    ]

    return results[
        [
            "customer_text",
            "brand_reply_text",
            "intent",
            "similarity"
        ]
    ]


if __name__ == "__main__":

    query = input("Customer message: ")

    predicted_intent = assign_intent(query)

    print()
    print("Predicted intent:", predicted_intent)

    print()
    print("TOP INTENT-AWARE HISTORICAL RESPONSES")
    print("======================================")

    results = retrieve_responses(
        query,
        predicted_intent=predicted_intent,
        top_k=3
    )

    for _, row in results.iterrows():

        print()
        print(
            "Similarity:",
            round(row["similarity"], 4)
        )

        print(
            "Historical customer:"
        )
        print(row["customer_text"])

        print()
        print(
            "Amazon response:"
        )
        print(row["brand_reply_text"])