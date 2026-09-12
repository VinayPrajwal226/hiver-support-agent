import pandas as pd
import time

from agent import run_agent

GOLDEN_PATH = "data/golden/golden_set.csv"
OUTPUT_PATH = "data/golden/agent_evaluation_leakage_safe.csv"

golden_df = pd.read_csv(GOLDEN_PATH)

try:
    results_df = pd.read_csv(OUTPUT_PATH)

    successful_df = results_df[
        results_df["predicted_intent"].notna()
        & results_df["predicted_escalation"].notna()
    ]

    results = successful_df.to_dict("records")
    completed_ids = set(successful_df["tweet_id"].astype(str))

    print(f"Existing successful evaluations: {len(results)}")

except FileNotFoundError:
    results = []
    completed_ids = set()

for _, row in golden_df.iterrows():

    tweet_id = str(row["tweet_id"])

    if tweet_id in completed_ids:
        continue

    print(f"Evaluating {len(results) + 1}/{len(golden_df)}...")

    success = False

    for attempt in range(5):

        try:
            result, evidence = run_agent(
                row["customer_text"],
                exclude_customer_text=row["customer_text"]
            )

            record = {
                "tweet_id": row["tweet_id"],
                "customer_text": row["customer_text"],
                "human_intent": row["intent"],
                "predicted_intent": result.get("intent"),
                "human_escalation": row["escalation"],
                "predicted_escalation": result.get("escalate"),
                "draft_reply": result.get("draft_reply"),
                "reason": result.get("reason"),
                "top_similarity": (
                    evidence.iloc[0]["similarity"]
                    if len(evidence) > 0
                    else 0
                )
            }

            if (
                record["predicted_intent"] is None
                or record["predicted_escalation"] is None
            ):
                raise RuntimeError("Incomplete agent response")

            results.append(record)

            pd.DataFrame(results).to_csv(
                OUTPUT_PATH,
                index=False
            )

            completed_ids.add(tweet_id)

            success = True

            print("Success.")

            break

        except Exception as e:

            print(f"Attempt {attempt + 1}/5 failed:")
            print(e)

            if attempt < 4:
                wait_time = 10 * (attempt + 1)
                print(f"Waiting {wait_time} seconds...")
                time.sleep(wait_time)

    if not success:
        print("Skipping this example after 5 failed attempts.")

    time.sleep(5)

print()
print("FULL LEAKAGE-SAFE EVALUATION COMPLETE")
print("======================================")
print("Successful evaluations:", len(results))
print("Total golden examples:", len(golden_df))
print("Remaining:", len(golden_df) - len(results))
print("Output:", OUTPUT_PATH)