# Amazon Customer Support Agent

An evidence-grounded AI customer-support agent built from the
Customer Support on Twitter dataset.

The system is designed for AmazonHelp and performs three tasks:

1. Classifies an incoming customer message into a small support-intent taxonomy.
2. Retrieves similar historical Amazon customer-support interactions and uses them as evidence for drafting a reply.
3. Decides whether the interaction can be auto-handled or should be escalated to a human.

The main design principle is:

> Retrieve first, then generate from historical evidence, while escalating when evidence is weak or the situation is ambiguous.

## Why AmazonHelp?

I selected AmazonHelp because it has the largest number of customer-support interactions in the dataset among the support accounts examined.

The selected Amazon subset contains 100,503 customer messages, providing enough historical interactions to study recurring support patterns while keeping the evaluation and development workflow reproducible on a laptop.

## Results

The system was evaluated on a manually labelled golden set of 250 customer messages.

| Component | Metric | Result |
|---|---:|---:|
| Intent classification | Accuracy | 79.6% |
| Intent classification | Macro F1 | 70.45% |
| Escalation | Accuracy | 46.8% |
| Escalation | Macro F1 | 40.9% |
| Retrieval | Average top-1 cosine similarity | 0.4407 |
| Evidence judge | Human/LLM agreement | 84.0% |
| Auto-handled replies | Overall quality | 4.22/5 |
| Auto-handled replies | Groundedness | 3.84/5 |
| Auto-handled replies | Safety | 5.00/5 |

The reply-quality evaluation covers the 25 examples where the agent chose to auto-handle. Escalated cases are evaluated separately because an escalation does not necessarily require a customer-facing draft.

The evidence-judge result comes from a separate 50-example human audit, balanced between supported and unsupported evidence.

## 2. Approach

The system is intentionally built as a simple, inspectable pipeline rather than a fully autonomous agent.

### Pipeline

```text
Incoming customer message
          |
          v
   Intent classifier
          |
          v
 Historical retrieval
          |
          v
 Top-3 similar interactions
          |
          v
     Gemini LLM
          |
          +------------------+
          |                  |
          v                  v
    Draft reply        Escalation decision




## 3. Baselines

I evaluated the system against two increasingly useful baselines.

### 3.1 Baseline 1 — Majority Class

The simplest possible classifier always predicts the most frequent intent in the golden evaluation set.

The majority class is `non_action_or_context`.

Results:

| Metric | Result |
|---|---:|
| Accuracy | 60.40% |
| Macro F1 | 0.0837 |

Although the accuracy is 60.4%, the macro F1 is only 0.0837.

This demonstrates that accuracy alone is misleading for this dataset because the golden set is highly imbalanced: 151 of the 250 examples belong to `non_action_or_context`.

### 3.2 Baseline 2 — TF-IDF + Logistic Regression

The second baseline uses TF-IDF features with unigram and bigram features followed by logistic regression.

The training data consists of historical Amazon customer messages that were weakly labelled using keyword/rule-based heuristics. The golden examples are excluded from training to avoid exact-text leakage.

Results on the 250-example golden set:

| Metric | Result |
|---|---:|
| Accuracy | 80.40% |
| Macro F1 | 0.7319 |

This provides a stronger non-LLM classification baseline.

### 3.3 Final Agent

The final system combines intent classification, historical response retrieval, and Gemini-based response generation and escalation.

| System | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 60.40% | 0.0837 |
| TF-IDF + Logistic Regression | 80.40% | 0.7319 |
| Final agent | 79.60% | 0.7045 |

The final agent does not outperform the TF-IDF classifier on intent classification alone. This is expected because the final system optimizes a broader task: classification, evidence retrieval, response generation, and escalation.

Therefore, classification accuracy is not treated as the sole measure of system quality.

## 4. Intent Taxonomy

The intent taxonomy was created by inspecting recurring patterns in the AmazonHelp customer messages and grouping them into a small number of operational support categories.

| Intent | Description |
|---|---|
| `order_status` | Customer asks about the current status, progress, or estimated delivery of an order. |
| `delivery_issue` | Delivery is delayed, missing, failed, rescheduled, or incorrectly delivered. |
| `refund_issue` | Customer requests a refund or reports a missing, delayed, incorrect, or disputed refund. |
| `return_replacement` | Customer wants to return, replace, or exchange an item, or arrange a return pickup. |
| `cancel_order` | Customer wants to cancel an order or reports a cancellation problem. |
| `account_login` | Problems accessing an account, including login, password, verification, or account access. |
| `payment_issue` | Problems involving payment methods, card charges, billing, gift cards, or payment authorization. |
| `product_issue` | Product is damaged, defective, broken, or not working and may require troubleshooting. |
| `non_action_or_context` | Messages without enough actionable information, including acknowledgements, thanks, emotional reactions, and context-dependent follow-ups. |

### Golden Evaluation Set

I created a 250-example manually labelled golden set for evaluation.

The examples were sampled from Amazon customer messages after exact-text deduplication. Labels were assigned manually using the taxonomy above.

The final intent distribution is:

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

Because the classes are highly imbalanced, I report macro F1 alongside accuracy.

### Escalation Labels

The same golden set also contains a manually assigned escalation label indicating whether the customer interaction should be handled automatically or sent to a human.

The golden set contains:

- 231 escalation cases
- 19 non-escalation cases

This imbalance makes raw escalation accuracy particularly misleading, so precision, recall, and F1 are more informative for this component.

## 5. Evaluation Methodology

The evaluation is designed to measure the individual components of the support agent rather than relying on a single headline metric.

### 5.1 Golden Set

The main evaluation set contains 250 manually labelled customer messages.

The examples were sampled after exact-text deduplication and manually labelled for:

- intent
- escalation decision

The golden set is kept separate from classifier training and retrieval evaluation.

During evaluation, exact customer-text matches from the golden set are excluded from the historical retrieval pool. This prevents the system from retrieving the same customer message that it is being evaluated on.

### 5.2 Classification Metrics

Intent classification is evaluated using:

- Accuracy
- Macro F1
- Per-class precision, recall, and F1

Macro F1 is particularly important because the intent distribution is highly imbalanced.

### 5.3 Retrieval Evaluation

For each golden example, the system retrieves the top three historical customer-support interactions.

The primary retrieval diagnostic is the cosine similarity of the top-ranked historical interaction.

The evaluation also records the retrieved customer message and corresponding Amazon support reply so that the evidence can be inspected manually.

Similarity is treated as a retrieval diagnostic rather than proof that the evidence is actually useful.

### 5.4 Evidence Evaluation

To evaluate whether retrieved historical interactions actually support the agent's handling, I manually audited 50 examples.

The audit contains 25 examples labelled as evidence-supported and 25 labelled as not evidence-supported.

The human labels answer:

> Does the retrieved historical evidence sufficiently support the way the agent handled this customer interaction?

A separate LLM judge then evaluated the same 50 examples using the customer message, retrieved evidence, agent decision, and generated response.

Human-versus-LLM agreement was:

**84.0% (42/50 examples).**

This provides a check on whether the automated evidence evaluation is reasonably aligned with human judgement.

### 5.5 Reply Quality Evaluation

Generated replies were evaluated using an LLM judge across five dimensions:

| Dimension | Scale |
|---|---:|
| Helpfulness | 1–5 |
| Groundedness | 1–5 |
| Safety | 1–5 |
| Appropriateness | 1–5 |
| Overall quality | 1–5 |

The judge was instructed to evaluate whether the response was supported by the retrieved evidence and whether it avoided unsupported policies, guarantees, refunds, compensation, dates, or sensitive-information requests.

Reply quality is reported separately for auto-handled cases because escalated cases do not necessarily require an automated customer-facing answer.

For the 25 auto-handled examples:

| Metric | Score |
|---|---:|
| Helpfulness | 4.16 / 5 |
| Groundedness | 3.84 / 5 |
| Safety | 5.00 / 5 |
| Appropriateness | 4.40 / 5 |
| Overall | 4.22 / 5 |

### 5.6 Reproducibility

The evaluation artifacts are saved under `data/golden/` and `data/amazon/`.

The consolidated evaluation summary can be reproduced with:

```bash
python src/evaluate_all.py

## 6. Failure Analysis

The evaluation shows that the main limitation is not simply whether the model can generate fluent text. The larger problem is selecting historical evidence that is specific enough to support the customer's actual issue.

### Failure Mode 1 — Semantically Similar but Operationally Irrelevant Retrieval

**Example**

Customer message:

> "Can I get a refund instead of a replacement?"

The retrieved historical interaction was about a Cherry Garcia ice-cream issue.

The retrieved text was linguistically similar enough to rank highly, but it did not contain evidence about the customer's actual refund-versus-replacement request.

**Why it fails**

TF-IDF retrieval relies heavily on shared words and phrases. Common support vocabulary such as "refund", "issue", "help", and "order" can produce a reasonable similarity score without establishing that the underlying support situation is the same.

**Hypothesis**

A semantic retrieval model combined with structured metadata such as intent, product type, resolution type, and issue stage would produce more useful evidence than lexical similarity alone.

**Next step**

Use hybrid retrieval:

- BM25/TF-IDF for exact support terminology
- embedding similarity for semantic similarity
- intent compatibility as a ranking feature
- reranking using a cross-encoder or LLM

---

### Failure Mode 2 — Generic Replies When the Customer Has Already Stated the Problem

**Example**

A customer explicitly described a refund/replacement issue, but the generated response asked the customer to provide more details instead of addressing the stated request.

**Why it fails**

The generation model is being conservative because the retrieved evidence is weak. However, the fallback becomes too generic and does not acknowledge information that is already present in the customer message.

**Hypothesis**

The generation step needs to distinguish between:

1. information already provided by the customer;
2. information supported by historical evidence;
3. information that is genuinely missing.

**Next step**

Add a structured intermediate step that extracts the customer's explicit request and unresolved information before generating the response.

---

### Failure Mode 3 — Relevant Evidence With the Wrong Specificity or Resolution Stage

**Example**

A customer reported that they had contacted support but had not received an update.

The retrieved historical interaction also involved a missing update, but the historical response pointed to a different correspondence path.

The generated response therefore appeared reasonable but was not strongly grounded in the retrieved evidence.

**Why it fails**

Two conversations can involve the same broad intent while being at different stages of resolution.

For example:

```text
Initial problem
      |
      v
Customer contacts support
      |
      v
Waiting for response
      |
      v
Follow-up
      |
      v
Issue resolved


## 7. What Is Misleading About My Headline Number?

The headline intent-classification accuracy of **79.6%** should not be interpreted as meaning that the overall support agent successfully handles 79.6% of customer interactions.

There are several reasons.

### 1. The evaluation set is highly imbalanced

151 of the 250 golden examples belong to `non_action_or_context`.

Therefore, a model can achieve relatively high accuracy by performing well on the dominant class while performing poorly on smaller intents.

This is why macro F1 is also reported.

The final agent achieves:

- Accuracy: **79.6%**
- Macro F1: **70.45%**

### 2. Classification is only one part of the system

The support agent must also:

- retrieve useful historical evidence;
- generate a grounded response;
- decide whether automation is safe;
- escalate ambiguous cases.

A high classification score does not guarantee good performance on these tasks.

### 3. Retrieval similarity does not prove evidence quality

The average top-1 cosine similarity is **0.4407**, but similarity alone does not tell us whether the retrieved interaction actually supports the response.

The 50-example evidence audit showed that human and LLM judgements agreed on **84%** of cases, demonstrating that evidence usefulness requires a separate evaluation.

### 4. Reply quality is measured only on auto-handled cases

The auto-handled subset achieved an overall reply-quality score of **4.22/5**, but this is based on only 25 examples.

It should therefore not be interpreted as the quality of all possible generated responses.

### 5. Escalation remains a weak component

The escalation classifier achieved a macro F1 of **40.9%**.

The system is conservative in some situations and can miss cases that should be escalated.

Therefore, the strongest claim supported by this evaluation is:

> The system demonstrates promising intent classification and safe, reasonably grounded responses on a small manually evaluated auto-handled subset, but retrieval quality and escalation remain significant limitations.

## 8. One-Week Improvement Plan

If I had one additional week, I would prioritize improving evidence retrieval and escalation rather than immediately increasing model size.

### Day 1 — Improve Retrieval

Build a hybrid retrieval system combining:

- TF-IDF/BM25 lexical retrieval
- embedding-based semantic retrieval
- intent compatibility
- product/entity similarity

Evaluate whether the retrieved evidence is actually useful rather than relying only on similarity scores.

### Day 2 — Add Conversation Context

Many customer messages are short or depend on previous messages.

Add conversation-level context such as:

- previous customer message
- previous Amazon response
- conversation stage
- unresolved issue

This should reduce failures caused by messages such as acknowledgements and short follow-ups.

### Day 3 — Add Resolution-Stage Classification

Classify interactions into stages such as:

```text
new issue
    ↓
troubleshooting
    ↓
waiting for support
    ↓
escalation
    ↓
resolved
    ↓
follow-up


## 9. Reproduction

### Requirements

- Python 3.10+
- Google Gemini API key
- The Customer Support on Twitter dataset

Install the dependencies:

```bash
pip install -r requirements.txt


## Step 13 — Decision Log

Now we need to satisfy the assignment's requirement for **10–15 non-obvious decision log entries**.

Add this after the Reproduction section:

```markdown
## 10. Decision Log

| Decision | Why I made it |
|---|---|
| Selected AmazonHelp as the target brand | It had the largest number of customer-support interactions among the support accounts examined. |
| Used a small 9-class intent taxonomy | A smaller operational taxonomy is easier to evaluate reliably than a large set of highly specific intents. |
| Included `non_action_or_context` | Many Twitter support messages are acknowledgements, thanks, emotional reactions, or context-dependent follow-ups rather than actionable requests. |
| Used a majority-class baseline | It establishes how much performance can be obtained simply from the severe class imbalance. |
| Used TF-IDF + logistic regression as the simple baseline | It provides a strong, interpretable non-LLM comparison for text classification. |
| Used weak labels for historical training data | The large historical dataset did not have manually labelled intents, so keyword/rule-based labels provided a scalable training signal. |
| Excluded golden examples from training | Prevents exact-text leakage from making classification results artificially optimistic. |
| Excluded golden examples from retrieval | Prevents the retriever from finding the exact evaluation message in historical data. |
| Retrieved the top 3 historical interactions | A small evidence set reduces irrelevant context while giving the generator multiple examples to compare. |
| Did not hard-filter retrieval by predicted intent | Early inspection showed that hard intent filtering could remove useful examples when the classifier was wrong. |
| Treated similarity as a diagnostic rather than evidence quality | High lexical similarity does not necessarily mean that the historical resolution applies to the current customer problem. |
| Evaluated evidence separately with humans and an LLM judge | Retrieval quality cannot be established reliably from cosine similarity alone. |
| Evaluated reply quality only on auto-handled cases | Escalated cases do not necessarily require an automated customer-facing response. |
| Added explicit safety constraints to generation | Historical support conversations can contain specific links, identifiers, or claims that should not be blindly reproduced. |
| Chose conservative handling of weak evidence | Unsupported automated support responses can be more damaging than sending an uncertain case to a human. |

## 11. Limitations

This prototype has several important limitations.

### Limited Golden Set

The main evaluation contains 250 manually labelled examples. Although this is sufficient for an initial evaluation, some intents have very few examples.

For example, `cancel_order` has only 2 examples and `product_issue` has only 3.

Therefore, per-class metrics for low-frequency intents should not be treated as stable estimates of production performance.

### Imbalanced Escalation Labels

The golden set contains 231 escalation examples and only 19 non-escalation examples.

As a result, escalation accuracy is not a useful standalone measure. Precision, recall, and macro F1 provide a better view of this component.

### Retrieval Limitations

The current retriever is based on TF-IDF rather than dense semantic embeddings.

This makes it sensitive to vocabulary overlap and can retrieve interactions that share words but have different underlying support situations.

### Conversation Context

The current system primarily retrieves based on the customer message.

Twitter support conversations are often multi-turn, and short messages can depend heavily on previous customer and Amazon messages.

This can make intent classification and retrieval difficult for context-dependent messages.

### Weakly Labelled Training Data

The TF-IDF classification baseline is trained using keyword/rule-based weak labels rather than a fully human-labelled training set.

Therefore, its performance should not be interpreted as the performance of a classifier trained on high-quality supervised labels.

### LLM Judge Limitations

The reply and evidence evaluations use an LLM judge.

LLM judges can disagree with human evaluators and may prefer fluent or generic responses even when the underlying evidence is insufficient.

For this reason, I included a 50-example human evidence audit and measured human/LLM agreement rather than treating the LLM judge as ground truth.

### Prototype-Level Escalation

The escalation component is not yet optimized around a production cost model.

In a real support system, false auto-handling and unnecessary escalation would have different business costs. Those costs should be explicitly defined before optimizing the decision threshold.

### No Live Customer Data or Account Actions

The agent does not access customer accounts, order systems, payment systems, or internal Amazon tools.

It only uses the provided historical Twitter support data and retrieved historical interactions.

Therefore, it should not be considered a production-ready support agent. It is an evaluated prototype demonstrating evidence-grounded support reasoning.

## 12. References

- Customer Support on Twitter dataset: ThoughtVector / Kaggle
- Google Gemini API documentation
- scikit-learn documentation for TF-IDF and logistic regression
- pandas documentation for data processing

The project does not copy historical support responses verbatim into generated answers. Historical interactions are used as retrieval evidence for the prototype.