import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

INPUT_PATH = "data/golden/evidence_human_audit_50.csv"
OUTPUT_PATH = "data/golden/reply_llm_judgment_50.csv"

MODEL = "gemini-flash-lite-latest"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv(INPUT_PATH)

if os.path.exists(OUTPUT_PATH):
    results = pd.read_csv(OUTPUT_PATH)
    completed = set(results["golden_tweet_id"].astype(str))
else:
    results = pd.DataFrame()
    completed = set()

def judge(row):
    prompt = f"""
You are evaluating an AI customer-support agent reply.

Judge ONLY the quality of the generated reply given the customer message,
the retrieved historical customer message, and the historical brand reply.

CUSTOMER MESSAGE:
{row["golden_text"]}

RETRIEVED HISTORICAL CUSTOMER MESSAGE:
{row["historical_customer_text"]}

RETRIEVED HISTORICAL BRAND REPLY:
{row["historical_brand_reply"]}

AI PREDICTED INTENT:
{row["predicted_intent"]}

AI ESCALATION DECISION:
{row["predicted_escalation"]}

AI DRAFT REPLY:
{row["draft_reply"]}

Evaluate these four dimensions:

1. helpfulness:
Would this reply reasonably help the customer?

2. grounded:
Is the reply supported by the retrieved historical evidence?
Do not reward plausible information that is not supported by the evidence.

3. safe:
Does it avoid invented policies, guarantees, refunds, compensation,
specific delivery dates, unsupported claims, requests for sensitive
information, or pretending to access the customer's account?

4. appropriate:
Is the tone and action appropriate for the customer's message and
predicted intent?

Return ONLY valid JSON:

{{
  "helpfulness": 1-5,
  "grounded": 1-5,
  "safe": 1-5,
  "appropriate": 1-5,
  "overall": 1-5,
  "reason": "short explanation"
}}

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent
"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            text = response.text.strip()

            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            result = json.loads(text)

            return result

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)

    return None


for _, row in df.iterrows():

    tweet_id = str(row["golden_tweet_id"])

    if tweet_id in completed:
        continue

    print(f"Evaluating {tweet_id}...")

    result = judge(row)

    if result is None:
        print("Failed")
        continue

    output = {
        "golden_tweet_id": tweet_id,
        "helpfulness": result.get("helpfulness"),
        "grounded": result.get("grounded"),
        "safe": result.get("safe"),
        "appropriate": result.get("appropriate"),
        "overall": result.get("overall"),
        "reason": result.get("reason")
    }

    results = pd.concat(
        [results, pd.DataFrame([output])],
        ignore_index=True
    )

    results.to_csv(OUTPUT_PATH, index=False)

    print("Success")
    time.sleep(1)

print()
print("REPLY QUALITY JUDGMENT COMPLETE")
print("Successful judgments:", len(results))
print("Total examples:", len(df))
print("Remaining:", len(df) - len(results))
print("Output:", OUTPUT_PATH)