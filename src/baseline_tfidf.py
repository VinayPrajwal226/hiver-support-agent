import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


TRAIN_PATH = "data/amazon/amazon_customer_messages.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"


train_df = pd.read_csv(TRAIN_PATH)
golden_df = pd.read_csv(GOLDEN_PATH)
golden_texts = set(
    golden_df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

train_df["text_clean"] = (
    train_df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

train_df = train_df[
    ~train_df["text_clean"].isin(golden_texts)
].copy()

print("Training examples after leakage removal:", len(train_df))

X_train = train_df["text_clean"]
X_test = golden_df["customer_text"].fillna("").astype(str)
y_test = golden_df["intent"].astype(str)


vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Create training labels using the intent taxonomy.
# These labels will come from keyword/rule mapping initially.
def assign_intent(text):
    text = text.lower()

    if any(word in text for word in ["refund", "money back", "refunded"]):
        return "refund_issue"

    if any(word in text for word in ["return", "replacement", "replace", "exchange"]):
        return "return_replacement"

    if any(word in text for word in ["cancel", "cancellation"]):
        return "cancel_order"

    if any(word in text for word in ["login", "log in", "password", "account", "verification code"]):
        return "account_login"

    if any(word in text for word in ["charge", "charged", "payment", "billing", "card"]):
        return "payment_issue"

    if any(word in text for word in ["broken", "damaged", "defective", "not working"]):
        return "product_issue"

    if any(word in text for word in ["delayed", "late", "missing package", "not delivered", "delivery"]):
        return "delivery_issue"

    if any(word in text for word in ["order status", "where is my order", "track my order"]):
        return "order_status"

    return "non_action_or_context"


y_train = X_train.apply(assign_intent)


model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)


print("TF-IDF + LOGISTIC REGRESSION")
print("===========================")
print("Training examples:", len(X_train))
print("Evaluation examples:", len(X_test))
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print()
print(classification_report(y_test, y_pred, zero_division=0))