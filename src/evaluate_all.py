import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

GOLDEN_PATH = "data/golden/golden_set.csv"
AGENT_PATH = "data/golden/agent_evaluation_leakage_safe.csv"
RETRIEVAL_PATH = "data/amazon/retrieval_evaluation_leakage_safe.csv"
EVIDENCE_PATH = "data/golden/evidence_llm_judgment_50.csv"
REPLY_PATH = "data/golden/reply_llm_judgment_50.csv"
AUDIT_PATH = "data/golden/evidence_human_audit_50.csv"

print("=" * 60)
print("HIVER AMAZON SUPPORT AGENT - EVALUATION SUMMARY")
print("=" * 60)

# -------------------------
# Intent + escalation
# -------------------------

gold = pd.read_csv(GOLDEN_PATH)
agent = pd.read_csv(AGENT_PATH)

intent_accuracy = accuracy_score(
    gold["intent"],
    agent["predicted_intent"]
)

intent_f1 = f1_score(
    gold["intent"],
    agent["predicted_intent"],
    average="macro"
)

human_escalation = (
    gold["escalation"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({"yes": True, "no": False})
)

predicted_escalation = (
    agent["predicted_escalation"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({"true": True, "false": False})
)

escalation_accuracy = accuracy_score(
    human_escalation,
    predicted_escalation
)

escalation_f1 = f1_score(
    human_escalation,
    predicted_escalation,
    average="macro"
)

print()
print("CLASSIFICATION")
print("-" * 60)
print("Examples:", len(gold))
print("Intent Accuracy:", round(intent_accuracy, 4))
print("Intent Macro F1:", round(intent_f1, 4))

print()
print("ESCALATION")
print("-" * 60)
print("Escalation Accuracy:", round(escalation_accuracy, 4))
print("Escalation Macro F1:", round(escalation_f1, 4))

# -------------------------
# Retrieval
# -------------------------

retrieval = pd.read_csv(RETRIEVAL_PATH)
top1 = retrieval[retrieval["rank"] == 1]

print()
print("RETRIEVAL")
print("-" * 60)
print("Golden examples:", top1["golden_tweet_id"].nunique())
print("Retrieved rows:", len(retrieval))
print(
    "Average Top-1 Similarity:",
    round(top1["similarity"].mean(), 4)
)

# -------------------------
# Evidence judge
# -------------------------

evidence = pd.read_csv(EVIDENCE_PATH)

human = evidence["human_evidence_supported"].str.lower().str.strip()
llm = evidence["llm_evidence_supported"].str.lower().str.strip()

evidence_agreement = (human == llm).mean()

print()
print("EVIDENCE JUDGE")
print("-" * 60)
print("Examples:", len(evidence))
print("Human-LLM Agreement:", round(evidence_agreement, 4))
print("Human-LLM Agreement %:", round(evidence_agreement * 100, 2))

# -------------------------
# Reply quality
# -------------------------

reply = pd.read_csv(REPLY_PATH)
audit = pd.read_csv(AUDIT_PATH)

reply = reply.merge(
    audit[["golden_tweet_id", "predicted_escalation"]],
    on="golden_tweet_id"
)

auto = reply[reply["predicted_escalation"] == False].copy()

print()
print("AUTO-HANDLED REPLY QUALITY")
print("-" * 60)
print("Auto-handled examples:", len(auto))

for column in [
    "helpfulness",
    "grounded",
    "safe",
    "appropriate",
    "overall"
]:
    print(
        column.capitalize() + ":",
        round(auto[column].mean(), 2),
        "/ 5"
    )

print()
print("OVERALL SCORE DISTRIBUTION")
print("-" * 60)

distribution = auto["overall"].value_counts().sort_index()

for score, count in distribution.items():
    print(f"{score}: {count}")

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)