import pandas as pd

DATA_PATH = "data/amazon/amazon_context.csv"

df = pd.read_csv(DATA_PATH)

print("Total rows:", len(df))
print("\nFirst 20 conversations:\n")

for i, row in df.head(20).iterrows():
    print("=" * 80)
    print(f"Customer: {row['text']}")
    print(f"Previous tweet: {row['parent_text']}")
    print(f"Previous author: {row['parent_author_id']}")
    print(f"Previous inbound: {row['parent_inbound']}")