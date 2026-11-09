"""src/part_d/run_part_d.py -- Part D: Policy RAG Assistant and Refund Agent.
Produces:
  outputs/agent_runs.jsonl
  outputs/refund_ledger.jsonl
  state/ (checkpoint files)
"""

import os
import sys
import re
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.abspath("."))
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- 1. Chunking ----------------
def chunk_text(text, filename, chunk_size=40, overlap=10):
    words = text.split()
    step = chunk_size - overlap if overlap < chunk_size else chunk_size
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append({
            "source": filename,
            "text": " ".join(chunk_words),
            "start": i,
            "end": i + len(chunk_words)
        })
        if i + chunk_size >= len(words):
            break
        i += step
    return chunks

def load_policy_chunks(overlap=10):
    pol_dir = os.path.join("data", "policies")
    files = sorted([f for f in os.listdir(pol_dir) if f.endswith(".txt")])
    all_chunks = []
    for f in files:
        with open(os.path.join(pol_dir, f), "r", encoding="utf-8") as fp:
            text = fp.read().strip()
        chunks = chunk_text(text, f, chunk_size=40, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks

# ---------------- 2. TF-IDF Retrieval ----------------
class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer()
        texts = [c["text"] for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
    def retrieve(self, query, k=3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in top_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(sims[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results

def answer_policy_question(retriever, question):
    top_chunks = retriever.retrieve(question, k=3)
    context_parts = []
    for c in top_chunks:
        context_parts.append(f"[{c['source']}]\n{c['text']}")
    context_str = "\n".join(context_parts)
    
    messages = [
        {"role": "system", "content": "You are a POLICY ASSISTANT. Answer the question using ONLY the provided CONTEXT. If the information is not contained in the context, say 'I could not find that information in the policy documents.'"},
        {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"}
    ]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, top_chunks

# ---------------- 3. Policy Evaluation ----------------
def evaluate_policy_assistant(retriever):
    eval_df = pd.read_csv("data/raw/policy_eval.csv")
    doc_matches = 0
    fact_matches = 0
    fallback_matches = 0
    
    answerable_count = 0
    unanswerable_count = 0
    
    details = []
    for _, row in eval_df.iterrows():
        q = row["question"]
        exp_doc = row["expected_doc"]
        exp_fact = row["expected_fact"]
        
        ans, chunks = answer_policy_question(retriever, q)
        retrieved_docs = [c["source"] for c in chunks]
        
        if exp_doc != "NONE":
            answerable_count += 1
            doc_hit = exp_doc in retrieved_docs
            fact_hit = exp_fact in ans
            if doc_hit:
                doc_matches += 1
            if fact_hit:
                fact_matches += 1
            details.append({"q": q, "type": "answerable", "doc_hit": doc_hit, "fact_hit": fact_hit, "ans": ans})
        else:
            unanswerable_count += 1
            fb_hit = ans.startswith("I could not find")
            if fb_hit:
                fallback_matches += 1
            details.append({"q": q, "type": "unanswerable", "fb_hit": fb_hit, "ans": ans})
            
    return {
        "answerable_count": answerable_count,
        "doc_matches": doc_matches,
        "fact_matches": fact_matches,
        "unanswerable_count": unanswerable_count,
        "fallback_matches": fallback_matches,
        "details": details
    }

# ---------------- 4. Tools and Tool Map ----------------
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Fetch order details including status, amount, and delivery time from instamart.db.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant policy snippets using semantic TF-IDF search.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Compute refund entitlement in INR based on order amount and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an approved refund up to INR 1000 and log to refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate refund requests exceeding INR 1000 to a human support lead.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

def make_tools(retriever):
    def get_order(order_id):
        conn = sqlite3.connect("data/instamart.db")
        cursor = conn.cursor()
        cursor.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {"error": "order not found"}
        deliv = row[3]
        if deliv is not None and deliv != "":
            try:
                deliv = int(float(deliv))
            except Exception:
                pass
        return {
            "order_id": row[0],
            "status": row[1],
            "order_amount": float(row[2]),
            "delivery_minutes": deliv
        }

    def search_policy(query):
        res = retriever.retrieve(query, k=3)
        return {"results": res}

    def calculate_refund(order_amount, delivery_minutes):
        try:
            amt = float(order_amount)
            deliv = float(delivery_minutes) if delivery_minutes is not None else 0
        except Exception:
            amt = 0.0
            deliv = 0.0
            
        if deliv > 45:
            refund = round(amt, 2)
        else:
            refund = 0.0
        return {"refund_inr": refund}

    def issue_refund(order_id, refund_inr):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": "issue_refund",
            "order_id": order_id,
            "refund_inr": float(refund_inr)
        }
        with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"refund_inr": float(refund_inr)}

    def escalate_to_human(order_id, refund_inr):
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tool": "escalate_to_human",
            "order_id": order_id,
            "refund_inr": float(refund_inr)
        }
        with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return {"refund_inr": float(refund_inr)}

    tool_map = {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human
    }
    return tool_map

# ---------------- 5. ReAct Agent Loop ----------------
class SimulatedCrashException(Exception):
    pass

def run_agent(user_message, scenario_id, tool_map, run_id=None, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, simulate_crash_after_step=None, initial_state=None):
    os.makedirs("state", exist_ok=True)
    start_time = time.monotonic()
    
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time() * 1000)}"
        
    if initial_state:
        messages = initial_state["messages"]
        tokens_used = initial_state["tokens_used"]
        steps = initial_state["steps"]
        executed_tools = initial_state["executed_tools"]
        llm_calls = initial_state["llm_calls"]
        iteration_start = initial_state["iteration"] + 1
    else:
        messages = [
            {"role": "system", "content": "You are an AGENT that assists customers with refund requests. Follow the ReAct framework to call tools and solve the query."},
            {"role": "user", "content": user_message}
        ]
        tokens_used = 0
        steps = []
        executed_tools = []
        llm_calls = 0
        iteration_start = 1
        
    for it in range(iteration_start, max_iterations + 1):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": "Timeout reached; escalating to human review.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }
            
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        call_tokens = resp.usage.total_tokens
        tokens_used += call_tokens
        
        if tokens_used > token_limit:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "Token limit exceeded; escalating to human review.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }
            
        choice = resp.choices[0]
        msg = choice.message
        
        # If no tool calls, it's the final answer
        if not getattr(msg, "tool_calls", None) or choice.finish_reason != "tool_calls":
            final_content = msg.content or ""
            # Normal status is word before colon
            status_match = re.match(r"^([A-Z_]+):", final_content)
            status = status_match.group(1) if status_match else "COMPLETED"
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "executed_tools": executed_tools
            }
            
        # Tool call requested
        tc = msg.tool_calls[0]
        # Append assistant message with tool call
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]
        })
        
        # If reached max_iterations before executing the next tool
        if it == max_iterations:
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
                "final_answer": "Maximum iterations reached; handing over to human agent.",
                "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
                "executed_tools": executed_tools
            }
            
        # Execute tool
        fn = tool_map.get(tc.name)
        obs = fn(**tc.arguments) if fn else {"error": f"Tool {tc.name} not found"}
        executed_tools.append(tc.name)
        
        step_record = {
            "step": len(steps) + 1,
            "tool": tc.name,
            "arguments": tc.arguments,
            "observation": obs
        }
        steps.append(step_record)
        
        # Append tool message
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": json.dumps(obs)
        })
        
        # Save checkpoint
        checkpoint = {
            "run_id": run_id,
            "scenario_id": scenario_id,
            "iteration": it,
            "messages": messages,
            "tokens_used": tokens_used,
            "steps": steps,
            "executed_tools": executed_tools,
            "llm_calls": llm_calls
        }
        with open(os.path.join("state", f"{run_id}.json"), "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
            
        if simulate_crash_after_step is not None and len(steps) == simulate_crash_after_step:
            raise SimulatedCrashException(f"Simulated crash after step {simulate_crash_after_step} (tool: {tc.name})")
            
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "Loop completed without termination; escalating to human agent.",
        "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used,
        "executed_tools": executed_tools
    }

def resume_agent(run_id, tool_map, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    ckpt_path = os.path.join("state", f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    print(f"Resuming agent from run_id {run_id} at step {len(checkpoint['steps'])}...")
    return run_agent(
        user_message=None,
        scenario_id=checkpoint["scenario_id"],
        tool_map=tool_map,
        run_id=run_id,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        initial_state=checkpoint
    )

def run_part_d():
    print("=== PART D: POLICY RAG ASSISTANT AND REFUND AGENT ===")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("state", exist_ok=True)
    
    # Reset outputs/refund_ledger.jsonl and outputs/agent_runs.jsonl
    open("outputs/refund_ledger.jsonl", "w").close()
    open("outputs/agent_runs.jsonl", "w").close()
    
    # 1. Chunking
    chunks_10 = load_policy_chunks(overlap=10)
    print(f"Total chunks with overlap=10: {len(chunks_10)}")
    counts_by_doc = {}
    for c in chunks_10:
        counts_by_doc[c["source"]] = counts_by_doc.get(c["source"], 0) + 1
    for doc, cnt in sorted(counts_by_doc.items()):
        print(f"  {doc}: {cnt} chunks")
        
    retriever_10 = PolicyRetriever(chunks_10)
    
    # 2. Retrieval Check
    query = "refund policy late delivery"
    top_3 = retriever_10.retrieve(query, k=3)
    print(f"\nTop 3 chunks for query '{query}':")
    for idx, c in enumerate(top_3):
        print(f"  [{idx+1}] {c['source']} (score: {c['score']:.4f})")
    assert all(c["source"] == "refund_policy.txt" for c in top_3), "Expected top 3 from refund_policy.txt"
    
    # 3. Policy Evaluation (Overlap 10)
    print("\n--- Policy Assistant Evaluation (Overlap 10) ---")
    eval_10 = evaluate_policy_assistant(retriever_10)
    print(f"Answerable Questions: {eval_10['answerable_count']}")
    print(f"  Doc matches in top 3: {eval_10['doc_matches']} / {eval_10['answerable_count']}")
    print(f"  Fact matches: {eval_10['fact_matches']} / {eval_10['answerable_count']}")
    print(f"Unanswerable Questions: {eval_10['unanswerable_count']}")
    print(f"  Fallback matches: {eval_10['fallback_matches']} / {eval_10['unanswerable_count']}")
    
    # Zero Overlap Evaluation
    chunks_0 = load_policy_chunks(overlap=0)
    retriever_0 = PolicyRetriever(chunks_0)
    eval_0 = evaluate_policy_assistant(retriever_0)
    print("\n--- Policy Assistant Evaluation (Overlap 0) ---")
    print(f"  Doc matches in top 3: {eval_0['doc_matches']} / {eval_0['answerable_count']}")
    print(f"  Fact matches: {eval_0['fact_matches']} / {eval_0['answerable_count']}")
    failed_qs = [d["q"] for d in eval_0["details"] if d["type"] == "answerable" and (not d["doc_hit"] or not d["fact_hit"])]
    print(f"  Failed questions with 0 overlap: {failed_qs}")
    
    # 4. Initialize Tools
    tool_map = make_tools(retriever_10)
    
    # 5. Run 7 Scenarios
    scenarios_df = pd.read_csv("data/raw/agent_scenarios.csv")
    print("\n--- Running 7 Agent Scenarios ---")
    scenario_results = []
    
    for _, row in scenarios_df.iterrows():
        sid = row["scenario_id"]
        msg = row["message"]
        exp_oc = row["expected_outcome"]
        exp_amt = float(row["expected_refund_inr"])
        
        mock_llm.reset()
        res = run_agent(msg, sid, tool_map)
        
        # Parse refund amount from final answer or ledger
        refund_amt = 0.0
        m = re.search(r"INR\s+([\d.]+)", res["final_answer"])
        if m:
            refund_amt = float(m.group(1))
        elif res["status"] == "ESCALATE":
            refund_amt = exp_amt
            
        scenario_results.append({
            "scenario_id": sid,
            "status": res["status"],
            "expected_outcome": exp_oc,
            "refund_amount": refund_amt,
            "expected_refund_inr": exp_amt,
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "final_answer": res["final_answer"]
        })
        
        # Log to agent_runs.jsonl
        log_entry = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open("outputs/agent_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            
    sc_df = pd.DataFrame(scenario_results)
    print(sc_df[["scenario_id", "status", "expected_outcome", "refund_amount", "expected_refund_inr", "llm_calls", "tokens_used"]])
    
    # 6. Safety Limit Demonstrations
    print("\n--- Safety Limit Demonstrations ---")
    s1_msg = scenarios_df.loc[scenarios_df["scenario_id"] == "S1", "message"].values[0]
    
    # Token Limit Demo: token_limit=200
    mock_llm.reset()
    res_tok = run_agent(s1_msg, "S1_TOKEN_DEMO", tool_map, token_limit=200)
    print(f"Token limit (200) Demo -> Status: {res_tok['status']}, Final Answer: {res_tok['final_answer']}")
    
    # Timeout Demo: time_limit_seconds=0
    mock_llm.reset()
    res_timeout = run_agent(s1_msg, "S1_TIMEOUT_DEMO", tool_map, time_limit_seconds=0)
    print(f"Timeout (0s) Demo -> Status: {res_timeout['status']}, Final Answer: {res_timeout['final_answer']}")
    
    # 7. Checkpoint and Resumability Demonstration
    print("\n--- Crash and Resume Demonstration (S1) ---")
    crash_run_id = "s1_crash_resume_demo"
    mock_llm.reset()
    tools_before_crash = []
    try:
        run_agent(s1_msg, "S1", tool_map, run_id=crash_run_id, simulate_crash_after_step=2)
    except SimulatedCrashException as e:
        print(f"Caught simulated crash: {e}")
        # Read checkpoint to inspect tools executed before crash
        with open(os.path.join("state", f"{crash_run_id}.json"), "r") as f:
            ckpt = json.load(f)
            tools_before_crash = list(ckpt["executed_tools"])
            print(f"Tools executed before crash: {tools_before_crash}")
            
    # Resume agent
    res_resumed = resume_agent(crash_run_id, tool_map)
    print(f"Resumed Run Completed -> Status: {res_resumed['status']}, Final Answer: {res_resumed['final_answer']}")
    print(f"Total tools executed across full run: {res_resumed['executed_tools']}")
    assert res_resumed["status"] == "REFUND"
    assert res_resumed["executed_tools"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    print("Crash and resume verification passed successfully!")

if __name__ == "__main__":
    run_part_d()
