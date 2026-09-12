import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

INPUT_PATH = "data/golden/evidence_human_audit_50.csv"
OUTPUT_PATH = "data/golden/evidence_llm_judgment_50.csv"

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)

df = pd.read_csv(INPUT_PATH)

if os.path.exists(OUTPUT_PATH):
    results_df = pd.read_csv(OUTPUT_PATH)
    results = results_df.to_dict("records")
    completed_ids = set(results_df["golden_tweet_id"].astype(str))
else:
    results = []
    completed_ids = set()

for _, row in df.iterrows():

    tweet_id = str(row["golden_tweet_id"])

    if tweet_id in completed_ids:
        continue

    print()
    print("=" * 80)
    print(f"Evaluating case {len(results) + 1} / {len(df)}")
    print("=" * 80)

    prompt = f"""
You are evaluating an AI customer-support agent.

Your task is to determine whether the historical Amazon support
evidence sufficiently supports the agent's handling of the customer.

CUSTOMER MESSAGE:
{row["golden_text"]}

HUMAN INTENT:
{row["golden_intent"]}

PREDICTED INTENT:
{row["predicted_intent"]}

PREDICTED ESCALATION:
{row["predicted_escalation"]}

HISTORICAL CUSTOMER MESSAGE:
{row["historical_customer_text"]}

HISTORICAL AMAZON RESPONSE:
{row["historical_brand_reply"]}

SIMILARITY SCORE:
{row["similarity"]}

AGENT DRAFT REPLY:
{row["draft_reply"]}

AGENT REASON:
{row["reason"]}

Judge only whether the historical evidence sufficiently supports
the agent's handling.

Use YES when the historical example provides relevant evidence
for the agent's response or escalation decision.

Use NO when the historical example is unrelated, too vague,
or does not support the agent's specific handling.

Do not judge whether the agent is generally helpful.
Do not use the similarity score as the only criterion.
Judge the relationship between the customer issue, historical
evidence, and agent handling.

Return JSON only:

{{
  "evidence_supported": "yes",
  "reason": "short explanation"
}}
"""

    success = False

    for attempt in range(5):

        try:

            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            result = json.loads(response.text)

            evidence_supported = str(
                result.get("evidence_supported", "")
            ).lower().strip()

            if evidence_supported not in ["yes", "no"]:
                raise ValueError(
                    "Invalid evidence_supported value"
                )

            record = {
                "golden_tweet_id": row["golden_tweet_id"],
                "human_evidence_supported": row[
                    "human_evidence_supported"
                ],
                "llm_evidence_supported": evidence_supported,
                "llm_reason": result.get("reason", "")
            }

            results.append(record)

            pd.DataFrame(results).to_csv(
                OUTPUT_PATH,
                index=False
            )

            completed_ids.add(tweet_id)

            print("Success.")
            success = True
            break

        except Exception as e:

            print(
                f"Attempt {attempt + 1}/5 failed:"
            )
            print(e)

            if attempt < 4:
                wait_time = 10 * (attempt + 1)
                print(
                    f"Waiting {wait_time} seconds..."
                )
                time.sleep(wait_time)

    if not success:
        print(
            "Skipping this example after 5 failed attempts."
        )

    time.sleep(3)

print()
print("=" * 80)
print("LLM EVIDENCE JUDGMENT COMPLETE")
print("=" * 80)

print(
    "Successful judgments:",
    len(results)
)

print(
    "Total examples:",
    len(df)
)

print(
    "Remaining:",
    len(df) - len(results)
)

print(
    "Output:",
    OUTPUT_PATH
)