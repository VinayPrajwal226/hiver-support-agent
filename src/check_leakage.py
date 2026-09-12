import pandas as pd

TRAIN_PATH = "data/amazon/amazon_customer_messages.csv"
GOLDEN_PATH = "data/golden/golden_set.csv"

train_df = pd.read_csv(TRAIN_PATH)
golden_df = pd.read_csv(GOLDEN_PATH)

train_texts = set(
    train_df["text"].fillna("").astype(str).str.strip()
)

golden_texts = set(
    golden_df["customer_text"].fillna("").astype(str).str.strip()
)

overlap = train_texts.intersection(golden_texts)

print("LEAKAGE CHECK")
print("=============")
print("Training messages:", len(train_df))
print("Golden examples:", len(golden_df))
print("Unique training texts:", len(train_texts))
print("Unique golden texts:", len(golden_texts))
print("Exact text overlap:", len(overlap))
print("Overlap percentage:", round(len(overlap) / len(golden_texts) * 100, 2))