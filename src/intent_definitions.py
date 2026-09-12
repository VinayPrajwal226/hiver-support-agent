INTENTS = {
    "order_status": {
        "description": "Customer wants information about the current status, estimated delivery date, or progress of an order.",
        "examples": [
            "I need to know the status of my order.",
            "Where is my order?",
            "When will my order arrive?"
        ]
    },

    "delivery_issue": {
        "description": "Customer reports a delivery problem such as a delayed, missing, failed, rescheduled, or incorrectly marked delivered package.",
        "examples": [
            "My package was marked delivered but I never received it.",
            "My delivery is late.",
            "Why was my delivery rescheduled?"
        ]
    },

    "refund_issue": {
        "description": "Customer is asking about a refund, a missing or delayed refund, an incorrect refund, or disputes a refund decision.",
        "examples": [
            "Where is my refund?",
            "I want my money back.",
            "I was refunded the wrong amount."
        ]
    },

    "return_replacement": {
        "description": "Customer wants to return, replace, exchange, or arrange pickup for a product.",
        "examples": [
            "I want to return this item.",
            "When will my return be picked up?",
            "Can I get a replacement?"
        ]
    },

    "cancel_order": {
        "description": "Customer wants to cancel an order or has a problem related specifically to cancelling an order.",
        "examples": [
            "I want to cancel my order.",
            "Why was my cancelled order charged?",
            "I can't find the option to cancel."
        ]
    },

    "account_login": {
        "description": "Customer has trouble accessing their account, logging in, resetting a password, or completing account verification.",
        "examples": [
            "I can't log in.",
            "My account is locked.",
            "I can't receive the verification code."
        ]
    },

    "payment_issue": {
        "description": "Customer has a problem with payment methods, card charges, billing, gift cards, or payment authorization.",
        "examples": [
            "Why was my card charged?",
            "My gift card won't work.",
            "I can't use my new card."
        ]
    },

    "product_issue": {
        "description": "Customer reports that a product is damaged, defective, not working, or needs product-specific troubleshooting.",
        "examples": [
            "My TV arrived damaged.",
            "My Fire TV Stick isn't working.",
            "The product is defective."
        ]
    },

    "non_action_or_context": {
        "description": "Message does not contain enough information for an actionable support request, or is primarily a thank-you, acknowledgement, confirmation, emotional reaction, or context-dependent follow-up.",
        "examples": [
            "Thanks.",
            "Details sent. Please check.",
            "Okay, will do.",
            "I haven't."
        ]
    }
}


if __name__ == "__main__":
    for intent, details in INTENTS.items():
        print(f"\n{intent}")
        print(details["description"])
        print("Examples:")
        for example in details["examples"]:
            print(f"  - {example}")