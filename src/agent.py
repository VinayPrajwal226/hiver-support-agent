import os
import json
from dotenv import load_dotenv
from google import genai

from retrieve_responses import retrieve_responses
from intent_classifier import IntentClassifier


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

client = genai.Client(api_key=api_key)


def build_prompt(customer_message, retrieved_examples):
    evidence = ""

    for i, (_, row) in enumerate(
        retrieved_examples.iterrows(),
        start=1
    ):
        evidence += f"""
Evidence {i}
Historical customer message:
{row["customer_text"]}

Historical Amazon response:
{row["brand_reply_text"]}

Similarity score:
{row["similarity"]:.4f}
---
"""

    prompt = f"""
You are an AI customer-support assistant for Amazon.

Your job is to respond to a customer using historical Amazon support
responses as evidence.

CUSTOMER MESSAGE:
{customer_message}

HISTORICAL SUPPORT EVIDENCE:
{evidence}

RULES:

1. Use the historical evidence to guide the response.
2. Do not copy historical responses verbatim.
3. Write a fresh response based on the resolution strategy shown in the evidence.
4. Do not copy Amazon agent signatures such as ^DY, ^RS, ^KP, etc.
5. Do not copy historical customer usernames, order numbers, tracking numbers, or personal information.
6. Do not invent Amazon policies or procedures.
7. Do not invent refunds, replacements, compensation, delivery dates, discounts, or other promises.
8. Do not claim that you accessed the customer's account or order.
9. Never request passwords, full payment-card numbers, or other sensitive information.
10. If the historical evidence is weak, ambiguous, or insufficient, escalate to a human.
11. Keep the draft reply concise and professional.
12. Return JSON only.

Return exactly this structure:

{{
  "draft_reply": "short customer-facing response",
  "escalate": true,
  "reason": "short explanation for the escalation decision"
}}
"""

    return prompt


def run_agent(
    customer_message,
    exclude_customer_text=None,
    intent_classifier=None
):
    if intent_classifier is None:
        intent_classifier = IntentClassifier()

    predicted_intent = intent_classifier.predict(
        customer_message
    )

    retrieved_examples = retrieve_responses(
        customer_message,
        top_k=3,
        exclude_customer_text=exclude_customer_text
    )

    prompt = build_prompt(
        customer_message,
        retrieved_examples
    )

    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result = json.loads(response.text)

    result["intent"] = predicted_intent

    return result, retrieved_examples


if __name__ == "__main__":
    customer_message = input("Customer message: ")

    result, evidence = run_agent(
        customer_message
    )

    print()
    print("AGENT RESULT")
    print("============")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print()
    print("EVIDENCE USED")
    print("=============")

    for i, (_, row) in enumerate(
        evidence.iterrows(),
        start=1
    ):
        print()
        print(
            f"Evidence {i} "
            f"(similarity={row['similarity']:.4f})"
        )

        print(
            "Customer:",
            row["customer_text"]
        )

        print(
            "Amazon:",
            row["brand_reply_text"]
        )