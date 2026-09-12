import re


def should_escalate(customer_message, predicted_intent, top_similarity):
    text = customer_message.lower().strip()

    if len(text) < 20:
        return True, "The customer message is too short or incomplete to determine the issue safely."

    if top_similarity < 0.30:
        return True, "Historical evidence is too weak to support a confident automated response."

    manager_terms = [
        "manager",
        "supervisor",
        "senior manager",
        "speak to someone",
        "speak with someone",
        "escalate",
    ]

    if any(term in text for term in manager_terms):
        return True, "The customer explicitly requests escalation to a human support representative."

    security_terms = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "security",
        "stolen",
        "identity theft",
        "someone else",
    ]

    if any(term in text for term in security_terms):
        return True, "The message indicates a potential security, fraud, or unauthorized-account issue."

    repeated_support_terms = [
        "many times",
        "multiple times",
        "over 100 emails",
        "10 times",
        "still no",
        "still nothing",
        "no solution",
        "no response",
        "already contacted",
        "already called",
        "already emailed",
    ]

    if any(term in text for term in repeated_support_terms):
        return True, "The customer reports repeated or unsuccessful attempts to resolve the issue."

    if predicted_intent == "non_action_or_context" and top_similarity < 0.40:
        return True, "The message is context-dependent and the available historical evidence is weak."

    return False, "The request has sufficient historical evidence and no high-risk escalation signal."


if __name__ == "__main__":
    message = input("Customer message: ")
    similarity = float(input("Top similarity: "))

    escalate, reason = should_escalate(
        message,
        "non_action_or_context",
        similarity
    )

    print()
    print("Escalate:", escalate)
    print("Reason:", reason)