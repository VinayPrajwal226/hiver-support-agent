import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/amazon/amazon_response_pairs.csv"

df = pd.read_csv(DATA_PATH)

df["customer_text"] = df["customer_text"].fillna("").astype(str)
df["brand_reply_text"] = df["brand_reply_text"].fillna("").astype(str)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

customer_matrix = vectorizer.fit_transform(df["customer_text"])


def retrieve_responses(query, top_k=3, exclude_customer_text=None):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        customer_matrix
    ).flatten()

    if exclude_customer_text is not None:
        exclude_text = str(exclude_customer_text).strip()

        mask = (
            df["customer_text"].str.strip() == exclude_text
        )

        similarities[mask] = -1

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = df.iloc[top_indices].copy()

    results["similarity"] = similarities[top_indices]

    return results[
        [
            "customer_text",
            "brand_reply_text",
            "similarity"
        ]
    ]


if __name__ == "__main__":

    query = input("Customer message: ")

    results = retrieve_responses(query)

    print()
    print("TOP HISTORICAL RESPONSES")
    print("========================")

    for _, row in results.iterrows():

        print()
        print("Similarity:", round(row["similarity"], 4))
        print("Historical customer:")
        print(row["customer_text"])
        print()
        print("Amazon response:")
        print(row["brand_reply_text"])