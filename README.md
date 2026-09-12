# Amazon Customer Support Agent

An evidence-grounded AI customer-support agent built from the **Customer Support on Twitter** dataset.

The system is designed for **AmazonHelp** and performs three tasks:

1. Classifies an incoming customer message into a small support-intent taxonomy.
2. Retrieves similar historical Amazon customer-support interactions and uses them as evidence for drafting a reply.
3. Decides whether the interaction can be auto-handled or should be escalated to a human.

> **Core principle:** Retrieve first, then generate from historical evidence, while escalating when evidence is weak or ambiguous.

---

## 1. Why AmazonHelp?

I selected AmazonHelp because it had the largest number of customer-support interactions among the support accounts examined in the dataset.

The selected Amazon subset contains **100,503 customer messages**, providing enough historical interactions to study recurring support patterns while keeping the development workflow practical.

---

## 2. Results

The system was evaluated on a manually labelled golden set of **250 customer messages**.

| Component | Metric | Result |
|---|---|---:|
| Intent classification | Accuracy | **79.6%** |
| Intent classification | Macro F1 | **70.45%** |
| Escalation | Accuracy | **46.8%** |
| Escalation | Macro F1 | **40.9%** |
| Retrieval | Average top-1 cosine similarity | **0.4407** |
| Evidence judge | Human/LLM agreement | **84.0%** |
| Auto-handled replies | Helpfulness | **4.16/5** |
| Auto-handled replies | Groundedness | **3.84/5** |
| Auto-handled replies | Safety | **5.00/5** |
| Auto-handled replies | Appropriateness | **4.40/5** |
| Auto-handled replies | Overall quality | **4.22/5** |

Reply-quality evaluation covers the **25 examples where the agent chose to auto-handle**. Escalated cases are evaluated separately because an escalation does not necessarily require an automated customer-facing draft.

---

## 3. Problem Framing

For AmazonHelp, a good support agent should:

- correctly identify the customer's support intent;
- use relevant historical Amazon support interactions as evidence;
- produce a concise customer-facing response;
- avoid inventing policies, refunds, compensation, dates, or guarantees;
- escalate when evidence is insufficient or the situation is ambiguous.

This prototype does **not** access customer accounts, order systems, payment systems, or internal Amazon tools. It operates using the provided historical support data.

---

## 4. Approach

The system is intentionally built as a simple, inspectable pipeline rather than a fully autonomous agent.

```text
                    Customer Message
                           |
                           v
                  +------------------+
                  | Intent Classifier|
                  +------------------+
                           |
                           v
                  +------------------+
                  | Historical       |
                  | Retrieval        |
                  +------------------+
                           |
                           v
                  Top-3 Similar
                  Historical Cases
                           |
                           v
                  +------------------+
                  | Gemini LLM       |
                  | Response Agent   |
                  +------------------+
                     /            \
                    /              \
                   v                v
             Draft Reply      Escalation Decision
```

The agent predicts an intent, retrieves historical Amazon support interactions, and provides those interactions to Gemini as evidence.

The generation prompt explicitly prevents the model from inventing policies, refunds, compensation, dates, or other unsupported claims.

---

## 5. Baselines

### 5.1 Majority-Class Baseline

The simplest classifier always predicts the most frequent intent in the golden set.

The majority class is:

```text
non_action_or_context
```

| Metric | Result |
|---|---:|
| Accuracy | **60.40%** |
| Macro F1 | **0.0837** |

The 60.4% accuracy is misleading because **151 of 250** golden examples belong to `non_action_or_context`.

### 5.2 TF-IDF + Logistic Regression

The second baseline uses TF-IDF unigram and bigram features followed by logistic regression.

Training data consists of historical Amazon customer messages weakly labelled using keyword/rule-based heuristics. Exact customer-text matches from the golden set are excluded to prevent leakage.

| Metric | Result |
|---|---:|
| Accuracy | **80.40%** |
| Macro F1 | **0.7319** |

This is a strong, interpretable non-LLM classification baseline.

### 5.3 Final Agent Comparison

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 60.40% | 0.0837 |
| TF-IDF + Logistic Regression | 80.40% | 0.7319 |
| Final agent | **79.60%** | **0.7045** |

The final agent does not outperform the TF-IDF classifier on intent classification alone. This is expected because the final system solves the broader task of classification, retrieval, response generation, and escalation.

---

## 6. Intent Taxonomy

| Intent | Description |
|---|---|
| `order_status` | Current order status, progress, or estimated delivery. |
| `delivery_issue` | Delivery is delayed, missing, failed, rescheduled, or incorrectly delivered. |
| `refund_issue` | Refund requested, missing, delayed, incorrect, or disputed. |
| `return_replacement` | Return, replacement, exchange, or return pickup. |
| `cancel_order` | Order cancellation or cancellation problem. |
| `account_login` | Account access, login, password, or verification. |
| `payment_issue` | Payment methods, card charges, billing, gift cards, or payment authorization. |
| `product_issue` | Damaged, defective, broken, or malfunctioning product. |
| `non_action_or_context` | Non-actionable, acknowledgement, thanks, emotional reaction, or context-dependent follow-up. |

### Golden Intent Distribution

| Intent | Examples |
|---|---:|
| `non_action_or_context` | 151 |
| `delivery_issue` | 49 |
| `refund_issue` | 11 |
| `account_login` | 10 |
| `return_replacement` | 9 |
| `payment_issue` | 8 |
| `order_status` | 7 |
| `product_issue` | 3 |
| `cancel_order` | 2 |
| **Total** | **250** |

Because the classes are highly imbalanced, macro F1 is reported alongside accuracy.

### Escalation Labels

The golden set contains:

- **231 escalation cases**
- **19 non-escalation cases**

This imbalance makes raw escalation accuracy particularly misleading.

---

## 7. Evaluation Methodology

### 7.1 Golden Set

The main evaluation set contains 250 manually labelled customer messages.

The examples were sampled after exact-text deduplication and manually labelled for:

- intent
- escalation decision

The golden set is kept separate from classifier training and retrieval evaluation.

During evaluation, exact customer-text matches from the golden set are excluded from the historical retrieval pool.

### 7.2 Classification

Intent classification is evaluated using:

- Accuracy
- Macro F1
- Per-class precision, recall, and F1

### 7.3 Retrieval

For every golden example, the system retrieves the top three historical customer-support interactions.

The primary retrieval diagnostic is the cosine similarity of the top-ranked historical interaction.

The average top-1 similarity was:

**0.4407**

Similarity is treated only as a retrieval diagnostic, not proof that evidence is useful.

### 7.4 Evidence Evaluation

A separate **50-example human evidence audit** was performed:

- 25 evidence-supported examples
- 25 unsupported examples

The human question was:

> Does the retrieved historical evidence sufficiently support the way the agent handled this customer interaction?

An independent LLM judge evaluated the same examples.

Human/LLM agreement:

**84.0% (42/50)**

This provides evidence that the automated evidence assessment is reasonably aligned with human judgement, while still acknowledging that the LLM judge is not ground truth.

### 7.5 Reply Quality

Generated replies were evaluated by an LLM judge on:

| Dimension | Scale |
|---|---:|
| Helpfulness | 1–5 |
| Groundedness | 1–5 |
| Safety | 1–5 |
| Appropriateness | 1–5 |
| Overall quality | 1–5 |

For the 25 auto-handled examples:

| Metric | Score |
|---|---:|
| Helpfulness | **4.16 / 5** |
| Groundedness | **3.84 / 5** |
| Safety | **5.00 / 5** |
| Appropriateness | **4.40 / 5** |
| Overall | **4.22 / 5** |

Overall-score distribution:

| Score | Examples |
|---:|---:|
| 2.0 | 1 |
| 3.0 | 1 |
| 3.5 | 1 |
| 4.0 | 13 |
| 5.0 | 9 |

Thus **22/25 auto-handled examples scored 4 or 5 overall**.

---

## 8. Failure Analysis

### Failure Mode 1 — Semantically Similar but Operationally Irrelevant Retrieval

**Example:** A customer asked for a refund instead of a replacement, but the retrieved historical interaction concerned a Cherry Garcia ice-cream issue.

The texts shared support vocabulary, but the historical resolution did not support the customer's actual request.

**Hypothesis:** Lexical similarity is insufficient for operational support retrieval.

**Next step:** Use hybrid retrieval with lexical similarity, embeddings, intent compatibility, and reranking.

### Failure Mode 2 — Generic Replies When the Customer Already Stated the Problem

Some responses asked the customer for more information even though the customer had already clearly described the issue.

**Hypothesis:** The generator needs to distinguish between facts already supplied by the customer and information genuinely missing.

**Next step:** Add a structured extraction step for the customer's request and unresolved information.

### Failure Mode 3 — Relevant Evidence With the Wrong Resolution Stage

Two conversations can have the same broad intent but be at different stages:

```text
Initial problem
      |
      v
Troubleshooting
      |
      v
Waiting for support
      |
      v
Escalation
      |
      v
Resolved
      |
      v
Follow-up
```

**Hypothesis:** Retrieval should account for resolution stage.

**Next step:** Add resolution-stage classification as a retrieval feature.

### Failure Mode 4 — Unsupported Assumptions From Plausible Evidence

A historical interaction may concern the same broad issue without proving that the exact action applies to the current customer.

**Hypothesis:** The model can make plausible but unsupported inferences from incomplete evidence.

**Next step:** Explicitly separate customer facts, historical evidence, and safe inference before generation.

### Failure Mode 5 — Internal Reasoning Leakage

One evaluated response exposed internal reasoning rather than clean customer-facing text.

Example pattern:

```text
"Since the historical evidence is insufficient to address multiple
pending charges directly, I will escalate this for you."
```

**Hypothesis:** Internal decision reasoning was not sufficiently separated from the customer-facing response.

**Next step:** Enforce structured output:

```text
draft_reply
escalate
reason
```

and validate that `draft_reply` contains only customer-facing language.

---

## 9. What Is Misleading About My Headline Number?

The headline intent-classification accuracy of **79.6%** does **not** mean that the overall support agent successfully handles 79.6% of customer interactions.

### Class imbalance

151/250 golden examples are `non_action_or_context`.

Therefore, accuracy can hide poor performance on smaller intents.

The final classifier has:

- Accuracy: **79.6%**
- Macro F1: **70.45%**

### Classification is only one component

The agent must also retrieve useful evidence, generate a grounded response, and make a safe escalation decision.

### Retrieval similarity is not evidence quality

The average top-1 similarity is **0.4407**, but similarity alone cannot determine whether the retrieved historical response actually supports the current case.

### Reply quality has a small sample

The **4.22/5** reply-quality result is based on only 25 auto-handled examples.

### Escalation is still weak

The escalation component has a macro F1 of **40.9%** and therefore requires substantial improvement before production use.

### Honest claim

> **The system demonstrates promising intent classification and safe, reasonably grounded responses on a small manually evaluated auto-handled subset, but retrieval quality and escalation remain significant limitations.**

---

## 10. One-Week Improvement Plan

### Day 1 — Improve Retrieval

Build hybrid retrieval using:

- TF-IDF/BM25
- embedding similarity
- intent compatibility
- product/entity similarity

### Day 2 — Add Conversation Context

Include previous customer and Amazon messages when the current message is short or context-dependent.

### Day 3 — Add Resolution-Stage Classification

Model stages such as:

```text
new issue -> troubleshooting -> waiting -> escalation -> resolved
```

Use the stage for retrieval and escalation.

### Day 4 — Improve Escalation

Create a dedicated validation set for:

- safe to automate
- must escalate
- insufficient information

Tune thresholds using an explicit cost model.

### Day 5 — Improve Response Generation

Use structured inputs:

```text
Customer request
Customer facts
Predicted intent
Resolution stage
Retrieved evidence
Evidence confidence
Escalation decision
```

### Day 6 — Error-Driven Evaluation

Track classification, retrieval, grounding, safety, and escalation failures separately.

### Day 7 — Re-run and Compare

Compare the improved system against both baselines and report only improvements supported by the held-out evaluation data.

---

## 11. Decision Log

| Decision | Reason |
|---|---|
| Selected AmazonHelp | Largest support-interaction volume among examined support accounts. |
| Used a 9-class taxonomy | Small enough to label consistently and operationally meaningful. |
| Included `non_action_or_context` | Many Twitter messages are acknowledgements, thanks, emotional reactions, or context-dependent follow-ups. |
| Added majority baseline | Measures the effect of class imbalance. |
| Added TF-IDF + Logistic Regression | Strong, interpretable non-LLM baseline. |
| Used weak labels for historical training | Large historical data did not have manually labelled intents. |
| Excluded golden texts from training | Prevents classification leakage. |
| Excluded golden texts from retrieval | Prevents exact-message retrieval leakage. |
| Retrieved top 3 historical cases | Gives the generator multiple examples without excessive context. |
| Did not hard-filter retrieval by intent | A wrong classifier prediction could otherwise remove useful evidence. |
| Treated similarity as a diagnostic | Similarity does not guarantee operational relevance. |
| Added human evidence audit | Tests whether retrieved evidence actually supports handling. |
| Compared LLM evidence judgement with humans | Checks alignment of automated evaluation with human judgement. |
| Evaluated reply quality on auto-handled cases | Escalated cases do not necessarily need generated replies. |
| Added safety constraints | Prevents copying sensitive details or inventing unsupported claims. |
| Favoured conservative handling of weak evidence | Unsupported automated support can be more harmful than escalation. |

---

## 12. Reproduction

### Requirements

- Python 3.10+
- Google Gemini API key
- Customer Support on Twitter dataset

Install:

```bash
pip install -r requirements.txt
```

Create `.env`:

```text
GEMINI_API_KEY=your_api_key_here
```

The `.env` file is excluded from Git.

### Reproduce Evaluation Summary

The committed evaluation artifacts allow the headline evaluation summary to be reproduced without downloading the full Twitter dataset:

```bash
python src/evaluate_all.py
```

### Dataset Setup

Place the raw dataset at:

```text
data/twcs/twcs.csv
```

Then:

```bash
python src/extract_amazon.py
python src/analyze_amazon.py
python src/build_amazon_context.py
python src/build_amazon_response_pairs.py
```

### Baselines

```bash
python src/baseline_majority.py
python src/baseline_tfidf.py
```

### Run the Agent

```bash
python src/agent.py
```

The generation prompt instructs the model to:

- use historical evidence;
- avoid copying responses verbatim;
- avoid inventing policies;
- avoid inventing refunds or compensation;
- avoid unsupported dates or promises;
- avoid requesting sensitive information;
- escalate when evidence is weak or ambiguous.

### Important Evaluation Artifacts

```text
data/golden/golden_set.csv
data/golden/agent_evaluation_leakage_safe.csv
data/golden/evidence_human_audit_50.csv
data/golden/evidence_llm_judgment_50.csv
data/golden/reply_llm_judgment_50.csv
data/amazon/retrieval_evaluation_leakage_safe.csv
```

The large raw dataset and large intermediate derived files are excluded from Git.

---

## 13. Repository Structure

```text
hiver-support-agent/
|
├── data/
│   ├── amazon/
│   │   ├── amazon_sample_2000.csv
│   │   └── retrieval_evaluation_leakage_safe.csv
│   │
│   └── golden/
│       ├── golden_set.csv
│       ├── agent_evaluation_leakage_safe.csv
│       ├── evidence_human_audit_50.csv
│       ├── evidence_llm_judgment_50.csv
│       ├── reply_llm_judgment_50.csv
│       └── README.md
│
├── src/
│   ├── agent.py
│   ├── analyze_amazon.py
│   ├── baseline_majority.py
│   ├── baseline_tfidf.py
│   ├── build_amazon_context.py
│   ├── build_amazon_response_pairs.py
│   ├── create_evidence_audit.py
│   ├── create_golden_candidates.py
│   ├── escalation_policy.py
│   ├── evaluate_agent.py
│   ├── evaluate_all.py
│   ├── evaluate_evidence.py
│   ├── evaluate_evidence_agreement.py
│   ├── evaluate_metrics.py
│   ├── evaluate_retrieval.py
│   ├── extract_amazon.py
│   ├── inspect_amazon.py
│   ├── inspect_amazon_context.py
│   ├── intent_classifier.py
│   ├── intent_definitions.py
│   ├── label_golden.py
│   ├── llm_evidence_judge.py
│   ├── llm_reply_judge.py
│   ├── retrieve_intent_aware.py
│   ├── retrieve_responses.py
│   └── sample_amazon.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 14. Limitations

### Limited Golden Set

The main evaluation contains 250 manually labelled examples. Some intents have very few examples, especially `cancel_order` and `product_issue`.

Per-class metrics for low-frequency intents should therefore not be treated as stable production estimates.

### Imbalanced Escalation Labels

There are 231 escalation and 19 non-escalation examples. Raw accuracy is therefore not an appropriate standalone measure.

### Retrieval Limitations

The current retriever uses TF-IDF rather than dense semantic embeddings, making it sensitive to vocabulary overlap.

### Conversation Context

Short Twitter messages often depend on previous turns. The current prototype does not fully model every multi-turn conversation.

### Weakly Labelled Training Data

The TF-IDF baseline uses keyword/rule-based weak labels rather than a large human-labelled training set.

### LLM Judge Limitations

LLM judges can disagree with humans and may prefer fluent or generic answers. The human evidence audit is therefore reported alongside LLM judgement.

### Prototype-Level Escalation

The escalation component is not optimized using a production business-cost model. False auto-handling and unnecessary escalation should have different costs in a real system.

### No Live Customer Actions

The prototype does not access customer accounts, orders, payments, or internal Amazon systems.

It should therefore be considered an evaluated prototype rather than a production-ready support agent.

---

## 15. References

- Customer Support on Twitter dataset — ThoughtVector / Kaggle: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
- Google Gemini API documentation: https://ai.google.dev/gemini-api/docs
- scikit-learn TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- pandas documentation: https://pandas.pydata.org/docs/

Historical support interactions are used as retrieval evidence. The project does not intentionally copy historical support responses verbatim into generated answers.
