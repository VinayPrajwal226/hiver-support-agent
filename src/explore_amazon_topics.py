import pandas as pd

DATA_PATH = "data/amazon/amazon_sample_2000.csv"

df = pd.read_csv(DATA_PATH)

print("Total messages:", len(df))
print("\nSample messages:\n")

for i, row in df.head(100).iterrows():
    print(f"{i + 1}. {row['text']}")