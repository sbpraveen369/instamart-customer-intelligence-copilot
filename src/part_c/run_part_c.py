"""src/part_c/run_part_c.py -- Part C: Support Ticket Triage with Guardrails.
Produces:
  outputs/triage_results.csv
  outputs/triage_budget_demo.csv
"""

import os
import sys
import re
import json
import random
import time

sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import mock_llm

MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
CATS = ["billing", "delivery", "technical", "fraud"]

# Regex patterns for PII
EMAIL_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_REGEX = re.compile(r"\b[6-9]\d{9}\b")
ORDER_REGEX = re.compile(r"ORD-\d{6}")

# System Prompts
ZERO_SHOT_PROMPT = """You are a CLASSIFY assistant. Classify the customer support ticket into exactly one of these four categories: billing, delivery, technical, fraud.
You must respond with valid JSON only in the following format:
{"category": "<category>", "reason": "<short explanation>"}"""

FEW_SHOT_PROMPT = """You are a CLASSIFY assistant. Classify the customer support ticket into exactly one of these four categories: billing, delivery, technical, fraud.
You must respond with valid JSON only in the following format:
{"category": "<category>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: I noticed an unauthorized transaction on my credit card.
Response: {"category": "fraud", "reason": "unauthorized card usage"}

Ticket: The delivery is delayed and the courier is unresponsive.
Response: {"category": "delivery", "reason": "delayed shipment"}"""

def redact_pii(text):
    text, email_cnt = EMAIL_REGEX.subn("[EMAIL]", text)
    text, phone_cnt = PHONE_REGEX.subn("[PHONE]", text)
    text, order_cnt = ORDER_REGEX.subn("[ORDER_ID]", text)
    return text, email_cnt, phone_cnt, order_cnt

def call_llm_with_retry(messages, ticket_id):
    for attempt in range(4): # 0, 1, 2, 3 (up to 3 retries)
        try:
            resp = mock_llm.chat(messages)
            return resp
        except mock_llm.TransientLLMError as e:
            if attempt == 3:
                raise
            wait = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            print(f"Log: Retry {attempt} for ticket {ticket_id} after {wait:.4f}s due to: {e}")
            time.sleep(wait)

def validate_response(resp_content):
    if not resp_content:
        return "flagged_invalid_output", ""
    
    # Try parsing JSON
    try:
        data = json.loads(resp_content)
        if isinstance(data, dict) and data.get("category") in CATS:
            return "classified_json", data["category"]
    except Exception:
        pass
        
    # Word extraction fallback
    words = re.findall(r"[a-z]+", resp_content.lower())
    matched_cats = [c for c in CATS if c in words]
    if len(matched_cats) == 1:
        return "classified_keyword", matched_cats[0]
        
    return "flagged_invalid_output", ""

def run_triage(prompt_type="few_shot", token_budget=100000):
    mock_llm.reset()
    tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    
    system_prompt = FEW_SHOT_PROMPT if prompt_type == "few_shot" else ZERO_SHOT_PROMPT
    
    results = []
    running_tokens = 0
    
    for _, row in tickets_df.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        text = row["text"]
        true_cat = row["true_category"]
        
        # 1. Input guardrails
        if not text.strip():
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue
            
        est_tok = mock_llm.est_tokens(text)
        if est_tok > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue
            
        # 2. Token Budget check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "true_category": true_cat, "tokens_used": 0
            })
            continue
            
        # 3. PII Redaction
        redacted_text, _, _, _ = redact_pii(text)
        # Assert no PII remaining
        assert not EMAIL_REGEX.search(redacted_text)
        assert not PHONE_REGEX.search(redacted_text)
        assert not ORDER_REGEX.search(redacted_text)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]
        
        # 4. LLM call with retry
        resp = call_llm_with_retry(messages, tid)
        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens
        
        # 5. Validate response
        content = resp.choices[0].message.content
        status, cat = validate_response(content)
        
        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": cat, "true_category": true_cat, "tokens_used": call_tokens
        })
        
    return pd.DataFrame(results), running_tokens

def run_part_c():
    print("=== PART C: SUPPORT TICKET TRIAGE WITH GUARDRAILS ===")
    os.makedirs("outputs", exist_ok=True)
    
    # Audit PII Redaction across all 164 tickets
    tickets_df = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    tot_emails, tot_phones, tot_orders = 0, 0, 0
    for text in tickets_df["text"]:
        _, e, p, o = redact_pii(text)
        tot_emails += e
        tot_phones += p
        tot_orders += o
    print(f"PII Redaction Audit: Emails={tot_emails}, Phones={tot_phones}, Orders={tot_orders}, Total={tot_emails + tot_phones + tot_orders}")
    
    # 1. Zero-shot Run
    print("\n--- Running Zero-Shot Pipeline ---")
    df_zs, tokens_zs = run_triage(prompt_type="zero_shot", token_budget=100000)
    classified_zs = df_zs[df_zs["status"].isin(["classified_json", "classified_keyword"])]
    acc_zs = (classified_zs["category"] == classified_zs["true_category"]).mean()
    print(f"Zero-shot: Classified={len(classified_zs)}, Accuracy={acc_zs*100:.2f}%, Total tokens={tokens_zs}, Tokens/ticket={tokens_zs/len(classified_zs):.1f}")
    print("Status counts:\n", df_zs["status"].value_counts())
    
    # 2. Few-shot Run
    print("\n--- Running Few-Shot Pipeline ---")
    df_fs, tokens_fs = run_triage(prompt_type="few_shot", token_budget=100000)
    classified_fs = df_fs[df_fs["status"].isin(["classified_json", "classified_keyword"])]
    acc_fs = (classified_fs["category"] == classified_fs["true_category"]).mean()
    print(f"Few-shot: Classified={len(classified_fs)}, Accuracy={acc_fs*100:.2f}%, Total tokens={tokens_fs}, Tokens/ticket={tokens_fs/len(classified_fs):.1f}")
    print("Status counts:\n", df_fs["status"].value_counts())
    
    # 3. Budget Demo Run
    print("\n--- Running Budget Demo (Zero-shot, Budget=6000) ---")
    df_budget, tokens_budget = run_triage(prompt_type="zero_shot", token_budget=6000)
    budget_exhausted_cnt = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Budget demo (6000 limit): {budget_exhausted_cnt} tickets budget_exhausted. Total tokens used: {tokens_budget}")
    df_budget.to_csv("outputs/triage_budget_demo.csv", index=False)
    print("Saved outputs/triage_budget_demo.csv")
    
    # 4. Final Ticket Routing with Part B Segments
    print("\n--- Routing Tickets with Customer Segments ---")
    segments = pd.read_csv("outputs/customer_segments.csv")
    
    # Merge segments
    df_final = df_fs.merge(segments[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")
    df_final["is_premium"] = df_final["is_premium"].fillna(False).astype(bool)
    df_final["segment_name"] = df_final["segment_name"].fillna("")
    
    # Routing logic
    routes = []
    priorities = []
    for _, row in df_final.iterrows():
        status = row["status"]
        cat = row["category"]
        prem = row["is_premium"]
        
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            routes.append("human_review")
            priorities.append("NORMAL")
        elif cat == "fraud":
            routes.append("security_team")
            priorities.append("HIGH")
        elif prem:
            routes.append("premium_desk")
            priorities.append("HIGH")
        else:
            routes.append("support_queue")
            priorities.append("NORMAL")
            
    df_final["route"] = routes
    df_final["priority"] = priorities
    
    # Blank category when not classified
    df_final.loc[~df_final["status"].isin(["classified_json", "classified_keyword"]), "category"] = ""
    
    output_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    final_output = df_final[output_cols]
    final_output.to_csv("outputs/triage_results.csv", index=False)
    print("Saved outputs/triage_results.csv")
    print("\nRouting Summary:")
    print(final_output["route"].value_counts())
    print("\nPriority Summary:")
    print(final_output["priority"].value_counts())

if __name__ == "__main__":
    run_part_c()
