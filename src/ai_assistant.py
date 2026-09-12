import os
import json
import urllib3
from src.intent_definitions import INTENTS

ALLOWED_INTENTS = list(INTENTS.keys())
ALLOWED_ESCALATIONS = ["yes", "no"]
ALLOWED_CONFIDENCE = ["High", "Medium", "Low"]

def validate_ai_proposal(proposal):
    if not isinstance(proposal, dict):
        return False, "Proposal must be a JSON object"
        
    intent = str(proposal.get("intent", "")).strip()
    escalation = str(proposal.get("escalation", "")).strip().lower()
    reason = str(proposal.get("escalation_reason", "")).strip()
    confidence = str(proposal.get("confidence", "")).strip().capitalize()
    
    if intent not in ALLOWED_INTENTS:
        return False, f"Invalid intent: {intent}"
        
    if escalation not in ALLOWED_ESCALATIONS:
        return False, f"Invalid escalation: {escalation}"
        
    if escalation == "yes" and not reason:
        return False, "AI selected escalation=yes but gave no escalation_reason"
        
    if confidence not in ALLOWED_CONFIDENCE:
        confidence = "High"
        
    return True, {
        "intent": intent,
        "escalation": escalation,
        "escalation_reason": reason,
        "confidence": confidence,
        "notes": str(proposal.get("notes", "")).strip()
    }

def generate_heuristic_proposal(parent_text, customer_text):
    text = (str(customer_text) + " " + str(parent_text)).lower()
    cust_lower = str(customer_text).lower()
    
    # Simple rule-based heuristic for AI proposal fallback
    if any(k in cust_lower for k in ["refund", "money back", "reimburse", "wrong amount"]):
        intent = "refund_issue"
        escalation = "yes"
        reason = "Refund request requiring account verification"
        conf = "High"
        notes = "Customer requesting refund details or reimbursement."
    elif any(k in cust_lower for k in ["cancel", "cancellation", "option to cancel"]):
        intent = "cancel_order"
        escalation = "yes"
        reason = "Order cancellation requires account-specific action"
        conf = "High"
        notes = "Customer asking to cancel order."
    elif any(k in cust_lower for k in ["delivery", "delivered", "package", "late", "where is my", "rescheduled", "carrier", "tracking"]):
        intent = "delivery_issue"
        escalation = "yes"
        reason = "Delivery problem requires package tracking and status check"
        conf = "High"
        notes = "Delivery delay or missing package reported."
    elif any(k in cust_lower for k in ["status of my order", "order status", "order #", "order number"]):
        intent = "order_status"
        escalation = "yes"
        reason = "Order-specific status investigation required"
        conf = "High"
        notes = "Customer checking order status."
    elif any(k in cust_lower for k in ["return", "replacement", "exchange", "pick up", "send back"]):
        intent = "return_replacement"
        escalation = "yes"
        reason = "Return or replacement processing required"
        conf = "High"
        notes = "Customer requesting item return/replacement."
    elif any(k in cust_lower for k in ["log in", "login", "password", "sign in", "account", "verification code"]):
        intent = "account_login"
        escalation = "yes"
        reason = "Account access and login verification required"
        conf = "High"
        notes = "Customer reporting login/account issue."
    elif any(k in cust_lower for k in ["charged", "card", "billing", "payment", "gift card"]):
        intent = "payment_issue"
        escalation = "yes"
        reason = "Payment or billing charge investigation required"
        conf = "High"
        notes = "Customer reporting billing or payment method issue."
    elif any(k in cust_lower for k in ["damaged", "defective", "broken", "not working", "echo", "stick", "tv", "device"]):
        intent = "product_issue"
        escalation = "yes"
        reason = "Product defect or troubleshooting required"
        conf = "High"
        notes = "Product troubleshooting or defect reported."
    else:
        intent = "non_action_or_context"
        escalation = "no" if len(cust_lower) < 30 and any(k in cust_lower for k in ["thanks", "thank you", "ok", "got it"]) else "yes"
        reason = "Context-dependent message or insufficient details provided" if escalation == "yes" else "No action required for general response"
        conf = "Medium"
        notes = "Message requires context or is non-actionable."
        
    return {
        "intent": intent,
        "escalation": escalation,
        "escalation_reason": reason,
        "confidence": conf,
        "notes": notes
    }

def get_api_key():
    # Check env vars (no hardcoding)
    for key in ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"]:
        val = os.environ.get(key)
        if val and len(val.strip()) > 5:
            return key, val.strip()
    return None, None

def generate_ai_proposal(parent_text, customer_text):
    key_name, api_key = get_api_key()
    
    if not api_key:
        proposal = generate_heuristic_proposal(parent_text, customer_text)
        proposal["notes"] += " (Generated via built-in intent heuristic. Configure OPENAI_API_KEY or GEMINI_API_KEY for LLM generation.)"
        return True, proposal, "No API key configured. Using heuristic AI generator."
        
    if key_name == "OPENAI_API_KEY":
        try:
            import requests
            urllib3.disable_warnings()
            
            system_prompt = (
                "You are an AI support assistant labeling customer service interactions. "
                "Analyze the conversation context and customer message. "
                "Select exactly one intent from: " + ", ".join(ALLOWED_INTENTS) + ".\n"
                "Escalation policy: set escalation to 'yes' if account investigation, order lookup, or internal action is needed. Set to 'no' if routine info or non-actionable.\n"
                "If escalation is 'yes', provide a short non-empty escalation_reason.\n"
                "Return JSON with keys: intent, escalation, escalation_reason, confidence ('High'/'Medium'/'Low'), notes."
            )
            
            user_content = f"Previous Amazon Response:\n{parent_text}\n\nCustomer Message:\n{customer_text}"
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.1
            }
            
            res = requests.post(url, headers=headers, json=payload, verify=False, timeout=10)
            if res.status_code == 200:
                data = res.json()
                raw_content = data['choices'][0]['message']['content']
                parsed = json.loads(raw_content)
                valid, result = validate_ai_proposal(parsed)
                if valid:
                    return True, result, "Generated via OpenAI API"
                else:
                    # Invalid LLM response, fallback to heuristic
                    fallback = generate_heuristic_proposal(parent_text, customer_text)
                    return True, fallback, f"LLM generated invalid format ({result}), fallback used."
            else:
                # Quota or API error fallback
                fallback = generate_heuristic_proposal(parent_text, customer_text)
                fallback["notes"] += f" (OpenAI API error {res.status_code}: fallback used.)"
                return True, fallback, f"OpenAI API error ({res.status_code}). Using heuristic AI generator."
        except Exception as e:
            fallback = generate_heuristic_proposal(parent_text, customer_text)
            return True, fallback, f"API Exception ({str(e)}). Using heuristic AI generator."
            
    # Default fallback
    proposal = generate_heuristic_proposal(parent_text, customer_text)
    return True, proposal, "Using heuristic AI generator."
