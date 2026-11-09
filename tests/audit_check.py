"""audit_check.py -- Comprehensive verification of deliverables against brief.
"""
import os
import sqlite3
import json
import pandas as pd
import joblib

def audit():
    print("=== VERIFYING REQUIRED PATHS ===")
    required_paths = [
        "generate_data.py", "mock_llm.py", "requirements.txt", ".gitignore", "README.md",
        "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv",
        "data/instamart.db", "sql/queries.sql",
        "outputs/customer_segments.csv", "outputs/lapse_model.joblib", "outputs/retention_list.csv",
        "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
        "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "state"
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    print(f"Missing required paths: {missing} (Count: {len(missing)})")
    assert len(missing) == 0, f"Missing files: {missing}"

    print("\n=== VERIFYING SQLITE DB ===")
    conn = sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = sorted([r[0] for r in cursor.fetchall()])
    print(f"Tables found: {tables}")
    assert set(tables).issuperset({"customers", "products", "orders"})
    cursor.execute("SELECT COUNT(*) FROM customers")
    n_cust = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM products")
    n_prod = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM orders")
    n_ord = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(order_amount) FROM orders WHERE status = 'completed'")
    rev = round(cursor.fetchone()[0], 2)
    print(f"DB counts: customers={n_cust}, products={n_prod}, orders={n_ord}, completed_revenue={rev}")
    assert n_cust == 1200
    assert n_prod == 40
    assert n_ord == 12309
    assert rev == 2998668.80
    conn.close()

    print("\n=== VERIFYING PART B OUTPUTS ===")
    seg_df = pd.read_csv("outputs/customer_segments.csv")
    print(f"customer_segments.csv shape: {seg_df.shape}, cols: {list(seg_df.columns)}")
    assert list(seg_df.columns) == ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    assert len(seg_df) == 1175
    assert seg_df["is_premium"].sum() == 138

    model = joblib.load("outputs/lapse_model.joblib")
    print(f"Loaded model: {type(model)}")
    assert hasattr(model, "predict")

    ret_df = pd.read_csv("outputs/retention_list.csv")
    print(f"retention_list.csv shape: {ret_df.shape}, cols: {list(ret_df.columns)}")
    assert list(ret_df.columns) == ["customer_id", "lapse_probability", "actually_lapsed"]
    assert len(ret_df) == 10

    print("\n=== VERIFYING PART C OUTPUTS ===")
    triage_df = pd.read_csv("outputs/triage_results.csv", keep_default_na=False)
    print(f"triage_results.csv shape: {triage_df.shape}, cols: {list(triage_df.columns)}")
    assert list(triage_df.columns) == ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    assert len(triage_df) == 164

    demo_df = pd.read_csv("outputs/triage_budget_demo.csv", keep_default_na=False)
    print(f"triage_budget_demo.csv shape: {demo_df.shape}")
    assert (demo_df["status"] == "budget_exhausted").sum() > 0

    print("\n=== VERIFYING PART D OUTPUTS ===")
    runs = []
    with open("outputs/agent_runs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    print(f"agent_runs.jsonl total runs: {len(runs)}")
    assert len(runs) >= 7
    s_ids = [r["scenario_id"] for r in runs[:7]]
    assert s_ids == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]

    ledger = []
    with open("outputs/refund_ledger.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ledger.append(json.loads(line))
    print(f"refund_ledger.jsonl total records: {len(ledger)}")
    assert len(ledger) >= 3

    checkpoints = [f for f in os.listdir("state") if f.endswith(".json")]
    print(f"Checkpoints in state/: {len(checkpoints)}")
    assert len(checkpoints) >= 1

    print("\n=== ALL FILE DELIVERABLES AND SCHEMAS VERIFIED! ===")

if __name__ == "__main__":
    audit()
