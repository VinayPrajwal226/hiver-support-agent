import pandas as pd
import re

DATA_PATH = "data/amazon/amazon_customer_messages.csv"

print("Loading Amazon customer dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nMessage length statistics:")
df["message_length"] = df["text"].fillna("").str.len()

print(df["message_length"].describe())

print("\nVery short messages:")
short_messages = df[df["message_length"] <= 20]

print(f"Count: {len(short_messages):,}")

print("\nExamples:")
for text in short_messages["text"].head(30):
    print("-", text)

print("\nMessages containing common support keywords:")

keywords = [
    "refund",
    "return",
    "delivery",
    "package",
    "order",
    "account",
    "prime",
    "payment",
    "charge",
    "cancel",
    "login",
    "password",
    "shipping",
    "broken",
    "not working"
]

for keyword in keywords:
    count = df["text"].fillna("").str.lower().str.contains(
        re.escape(keyword),
        regex=True
    ).sum()

    print(f"{keyword:15} {count:,}")