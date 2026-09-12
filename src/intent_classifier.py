import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_PATH = "data/amazon/amazon_customer_messages.csv"


def assign_weak_intent(text):
    text = text.lower()

    if any(word in text for word in ["refund", "money back", "refunded"]):
        return "refund_issue"

    if any(word in text for word in ["return", "replacement", "replace", "exchange"]):
        return "return_replacement"

    if any(word in text for word in ["cancel", "cancellation"]):
        return "cancel_order"

    if any(word in text for word in [
        "login", "log in", "password", "account", "verification code"
    ]):
        return "account_login"

    if any(word in text for word in [
        "charge", "charged", "payment", "billing", "card"
    ]):
        return "payment_issue"

    if any(word in text for word in [
        "broken", "damaged", "defective", "not working"
    ]):
        return "product_issue"

    if any(word in text for word in [
        "delayed", "late", "missing package", "not delivered", "delivery"
    ]):
        return "delivery_issue"

    if any(word in text for word in [
        "order status", "where is my order", "track my order"
    ]):
        return "order_status"

    return "non_action_or_context"


class IntentClassifier:

    def __init__(self, exclude_texts=None):

        df = pd.read_csv(TRAIN_PATH)

        df["text_clean"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if exclude_texts is not None:
            exclude_texts = {
                str(text).strip()
                for text in exclude_texts
            }

            df = df[
                ~df["text_clean"].isin(exclude_texts)
            ].copy()

        df["weak_intent"] = df["text_clean"].apply(
            assign_weak_intent
        )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )

        X = self.vectorizer.fit_transform(
            df["text_clean"]
        )

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        self.model.fit(
            X,
            df["weak_intent"]
        )

    def predict(self, text):

        X = self.vectorizer.transform([text])

        return self.model.predict(X)[0]