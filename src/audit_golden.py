import pandas as pd

PATH = "data/golden/golden_set.csv"

df = pd.read_csv(PATH)

print("=" * 80)
print("GOLDEN SET AUDIT")
print("=" * 80)

print(f"\nTotal examples: {len(df)}")

print("\n" + "=" * 80)
print("INTENT DISTRIBUTION")
print("=" * 80)

intent_counts = df["intent"].value_counts()
intent_pct = df["intent"].value_counts(normalize=True) * 100

for intent, count in intent_counts.items():
    print(f"{intent:25s} {count:3d} ({intent_pct[intent]:5.1f}%)")

print("\n" + "=" * 80)
print("ESCALATION DISTRIBUTION")
print("=" * 80)

esc_counts = df["escalation"].value_counts(dropna=False)
esc_pct = df["escalation"].value_counts(normalize=True, dropna=False) * 100

for value, count in esc_counts.items():
    print(f"{str(value):25s} {count:3d} ({esc_pct[value]:5.1f}%)")

print("\n" + "=" * 80)
print("EXAMPLES BY INTENT")
print("=" * 80)

for intent in intent_counts.index:
    print("\n" + "-" * 80)
    print(f"INTENT: {intent}")
    print("-" * 80)

    examples = df[df["intent"] == intent].sample(
        n=min(5, len(df[df["intent"] == intent])),
        random_state=42
    )

    for _, row in examples.iterrows():
        print(f"\nTweet ID: {row['tweet_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Previous Amazon: {row['parent_text']}")
        print(f"Intent: {row['intent']}")
        print(f"Escalation: {row['escalation']}")
        print(f"Reason: {row['escalation_reason']}")
        print(f"Confidence: {row['confidence']}")

print("\n" + "=" * 80)
print("ALL NON-ESCALATED EXAMPLES")
print("=" * 80)

no_escalation = df[df["escalation"].astype(str).str.lower() == "no"]

if len(no_escalation) == 0:
    print("\nNo examples have escalation = no.")
else:
    for _, row in no_escalation.iterrows():
        print(f"\nTweet ID: {row['tweet_id']}")
        print(f"Customer: {row['customer_text']}")
        print(f"Previous Amazon: {row['parent_text']}")
        print(f"Intent: {row['intent']}")
        print(f"Escalation: {row['escalation']}")
        print(f"Reason: {row['escalation_reason']}")
        print(f"Confidence: {row['confidence']}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)