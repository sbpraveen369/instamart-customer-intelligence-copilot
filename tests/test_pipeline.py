"""tests/test_pipeline.py -- Automated tests for Parts A, B, C, and D.
"""

import os
import json
import sqlite3
import pandas as pd
import joblib

def test_part_a_files_and_db():
    assert os.path.exists("data/clean/customers.csv")
    assert os.path.exists("data/clean/orders.csv")
    assert os.path.exists("data/clean/products.csv")
    assert os.path.exists("data/instamart.db")
    
    conn = sqlite3.connect("data/instamart.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM customers")
    assert cursor.fetchone()[0] == 1200
    cursor.execute("SELECT COUNT(*) FROM products")
    assert cursor.fetchone()[0] == 40
    cursor.execute("SELECT COUNT(*) FROM orders")
    assert cursor.fetchone()[0] == 12309
    cursor.execute("SELECT SUM(order_amount) FROM orders WHERE status = 'completed'")
    assert round(cursor.fetchone()[0], 2) == 2998668.80
    conn.close()

def test_part_b_outputs():
    assert os.path.exists("outputs/customer_segments.csv")
    assert os.path.exists("outputs/lapse_model.joblib")
    assert os.path.exists("outputs/retention_list.csv")
    
    seg_df = pd.read_csv("outputs/customer_segments.csv")
    assert set(seg_df.columns) == {"customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"}
    assert len(seg_df) == 1175
    assert seg_df["is_premium"].sum() == 138
    
    # Check model
    model = joblib.load("outputs/lapse_model.joblib")
    assert hasattr(model, "predict")
    
    ret_df = pd.read_csv("outputs/retention_list.csv")
    assert set(ret_df.columns) == {"customer_id", "lapse_probability", "actually_lapsed"}
    assert len(ret_df) == 10

def test_part_c_outputs():
    assert os.path.exists("outputs/triage_results.csv")
    assert os.path.exists("outputs/triage_budget_demo.csv")
    
    res_df = pd.read_csv("outputs/triage_results.csv")
    assert len(res_df) == 164
    assert set(res_df.columns) == {"ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"}
    assert set(res_df["route"].unique()).issubset({"human_review", "security_team", "premium_desk", "support_queue"})
    
    demo_df = pd.read_csv("outputs/triage_budget_demo.csv")
    assert (demo_df["status"] == "budget_exhausted").sum() > 0

def test_part_d_outputs():
    assert os.path.exists("outputs/agent_runs.jsonl")
    assert os.path.exists("outputs/refund_ledger.jsonl")
    assert os.path.isdir("state")
    
    # Read agent_runs.jsonl
    runs = []
    with open("outputs/agent_runs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    assert len(runs) >= 7
    s_ids = [r["scenario_id"] for r in runs[:7]]
    assert s_ids == ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    assert runs[0]["status"] == "REFUND"
    assert runs[2]["status"] == "ESCALATE"
    assert runs[3]["status"] == "DENIED"
    assert runs[5]["status"] == "NOT_FOUND"
    assert runs[6]["status"] == "FALLBACK_MAX_ITERATIONS"

if __name__ == "__main__":
    test_part_a_files_and_db()
    test_part_b_outputs()
    test_part_c_outputs()
    test_part_d_outputs()
    print("ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
