# Golden Evaluation Set (AI-Assisted Human Review Workflow)

This directory contains the 250-example golden evaluation set for the AI Support Agent, built using an **AI-Assisted Human Review Workflow**.

## Workflow Overview
```
250 Candidate Examples
          ↓
  AI Generates Proposal (Intent, Escalation, Reason, Confidence, Notes)
          ↓
  Human Reviews & Modifies Suggestions in Streamlit Interface
          ↓
  Human clicks "Approve & Save Human Label"
          ↓
  Saved to data/golden/golden_set.csv (label_status = "human_reviewed")
```

*Crucial Rule: AI proposals remain `label_status = "unreviewed"` until an explicit human action ("Approve & Save Human Label") approves and finalizes the row.*

## Dataset Details
- **Source Dataset:** Customer Support on Twitter (`twcs.csv`)
- **Brand Selected:** `AmazonHelp`
- **Candidate Sample Size:** Exactly 250 candidate examples
- **Sampling Seed:** Random seed 42
- **Duplicate Handling:** Exact-text duplicate removal was applied to ensure the sample provides 250 unique customer interactions.
- **Conversation Context:** Evaluates `parent_text` (Previous Amazon Response) and `customer_text` (Customer Message) together.

*Disclaimer: These 250 examples are a reproducible evaluation sample selected from the Amazon dataset. They are NOT claimed to be statistically representative of all Amazon support interactions.*

## 9 Intent Definitions
1. **order_status**: Customer wants information about the current status, estimated delivery date, or progress of an order.
2. **delivery_issue**: Customer reports a delivery problem such as a delayed, missing, failed, rescheduled, or incorrectly marked delivered package.
3. **refund_issue**: Customer is asking about a refund, a missing or delayed refund, an incorrect refund, or disputes a refund decision.
4. **return_replacement**: Customer wants to return, replace, exchange, or arrange pickup for a product.
5. **cancel_order**: Customer wants to cancel an order or has a problem related specifically to cancelling an order.
6. **account_login**: Customer has trouble accessing their account, logging in, resetting a password, or completing account verification.
7. **payment_issue**: Customer has a problem with payment methods, card charges, billing, gift cards, or payment authorization.
8. **product_issue**: Customer reports that a product is damaged, defective, not working, or needs product-specific troubleshooting.
9. **non_action_or_context**: Message does not contain enough information for an actionable support request, or is primarily a thank-you, acknowledgement, confirmation, emotional reaction, or context-dependent follow-up.

## Escalation Policy & Rules
The escalation decision (`yes` / `no`) answers:
*"Would I be comfortable allowing an automated support agent to handle this request using only the available conversation context and historical support knowledge, without accessing the customer's account?"*

- **YES**: If account-specific investigation, internal system usage, complex problem solving, or dispute verification is required.
- **NO**: If it is a routine or informational request that can be safely handled based on historical data without account access.

### Escalation Reason (required) Rule
- If **Escalate to human?** is `yes`, **Escalation Reason (required)** MUST be non-empty after stripping whitespace.
- If **Escalate to human?** is `no`, Escalation Reason may be empty.

## AI Proposals vs Human Labels Storage
To allow measuring AI vs. human agreement later, AI suggestions and final human labels are stored separately in `golden_set.csv`:

- **AI Proposal Columns:** `ai_intent`, `ai_escalation`, `ai_escalation_reason`, `ai_confidence`
- **Final Human-Reviewed Columns:** `intent`, `escalation`, `escalation_reason`, `confidence`, `notes`, `label_status`, `labeler`, `label_timestamp`

## API Key Configuration
No API keys are hard-coded in the repository. Configure your API key via environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
# OR
export GEMINI_API_KEY="your-gemini-api-key"
```
If no LLM API key is detected, the application automatically uses the built-in intent definition heuristic generator and displays an guidance banner.

## How to Launch the Labeling Interface
From the root of the project, run:
```bash
streamlit run src/label_golden.py
```

## How to Run Workflow Verification Tests
```bash
python src/test_golden_workflow.py
```

## Final Golden Output Path
`data/golden/golden_set.csv`
