"""
src/part_a/validate_data.py
Step 8 — Validation Script

Implements automated assertions for the reference checkpoints from Tasks 3 & 4.
When every assertion passes, prints exactly:
ALL CHECKPOINTS PASSED
Do not print this message if any assertion fails.
"""

import os
import sqlite3
import pandas as pd

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
DB_PATH   = "data/instamart.db"


def validate():
    # ── Verify required paths exist ──────────────────────────────────────────
    assert os.path.exists(f"{RAW_DIR}/orders.csv"), "Missing raw orders.csv"
    assert os.path.exists(f"{RAW_DIR}/customers.csv"), "Missing raw customers.csv"
    assert os.path.exists(f"{CLEAN_DIR}/customers.csv"), "Missing clean customers.csv"
    assert os.path.exists(f"{CLEAN_DIR}/products.csv"), "Missing clean products.csv"
    assert os.path.exists(f"{CLEAN_DIR}/orders.csv"), "Missing clean orders.csv"
    assert os.path.exists(DB_PATH), f"Missing database {DB_PATH}"

    raw_orders = pd.read_csv(f"{RAW_DIR}/orders.csv")
    raw_cust   = pd.read_csv(f"{RAW_DIR}/customers.csv")
    clean_cust = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
    clean_prod = pd.read_csv(f"{CLEAN_DIR}/products.csv")
    clean_orders = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    # ── Task 3 Raw Checkpoints ────────────────────────────────────────────────
    # Checkpoint 1: Raw orders count
    assert len(raw_orders) == 13084, f"CP1 Failed: Expected 13084 raw orders, got {len(raw_orders)}"

    # Checkpoint 2: Exact duplicates in raw orders
    assert raw_orders.duplicated().sum() == 119, f"CP2 Failed: Expected 119 duplicates, got {raw_orders.duplicated().sum()}"

    # Checkpoint 3: Pending rows in raw orders
    assert (raw_orders["status"] == "pending").sum() == 656, f"CP3 Failed: Expected 656 pending, got {(raw_orders['status'] == 'pending').sum()}"

    # Checkpoint 4: Distinct order IDs in raw orders
    assert raw_orders["order_id"].nunique() == 12309, f"CP4 Failed: Expected 12309 distinct order_ids, got {raw_orders['order_id'].nunique()}"

    # Checkpoint 5: Non-numeric order amounts
    oa_num = pd.to_numeric(raw_orders["order_amount"], errors="coerce")
    assert oa_num.isna().sum() == 120, f"CP5 Failed: Expected 120 non-numeric amounts, got {oa_num.isna().sum()}"

    # Checkpoint 6: Negative and blank delivery minutes
    dm_num = pd.to_numeric(raw_orders["delivery_minutes"], errors="coerce")
    assert (dm_num < 0).sum() == 106, f"CP6a Failed: Expected 106 negative delivery times, got {(dm_num < 0).sum()}"
    assert raw_orders["delivery_minutes"].isna().sum() == 2541, f"CP6b Failed: Expected 2541 blank delivery times, got {raw_orders['delivery_minutes'].isna().sum()}"

    # Checkpoint 7: Customer blank and invalid ages
    assert raw_cust["age"].isna().sum() == 61, f"CP7a Failed: Expected 61 blank ages, got {raw_cust['age'].isna().sum()}"
    cust_age_num = pd.to_numeric(raw_cust["age"], errors="coerce")
    assert ((cust_age_num < 10) | (cust_age_num > 100)).sum() == 6, f"CP7b Failed: Expected 6 invalid ages, got {((cust_age_num < 10) | (cust_age_num > 100)).sum()}"

    # ── Task 4 Clean Checkpoints ──────────────────────────────────────────────
    # Checkpoint 8: Clean customers count and city distribution
    assert len(clean_cust) == 1200, f"CP8 Failed: Expected 1200 customers, got {len(clean_cust)}"
    exp_cities = {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    city_counts = clean_cust["city"].value_counts().to_dict()
    for c, cnt in exp_cities.items():
        assert city_counts.get(c) == cnt, f"CP8 Failed for city {c}: expected {cnt}, got {city_counts.get(c)}"

    # Checkpoint 9: Clean customer gender distribution and ages
    exp_genders = {"female": 610, "male": 560, "unknown": 30}
    gender_counts = clean_cust["gender"].value_counts().to_dict()
    for g, cnt in exp_genders.items():
        assert gender_counts.get(g) == cnt, f"CP9 Failed for gender {g}: expected {cnt}, got {gender_counts.get(g)}"
    assert clean_cust["age"].isna().sum() == 0, "CP9 Failed: Found missing ages in clean customers"
    assert clean_cust["age"].between(18, 60).all(), "CP9 Failed: Age outside 18-60 found"

    # Checkpoint 10: Clean orders count and status breakdown
    assert len(clean_orders) == 12309, f"CP10 Failed: Expected 12309 clean orders, got {len(clean_orders)}"
    assert (clean_orders["status"] == "completed").sum() == 11110, "CP10 Failed: Expected 11110 completed"
    assert (clean_orders["status"] == "cancelled").sum() == 1199, "CP10 Failed: Expected 1199 cancelled"
    assert (clean_orders["status"] == "pending").sum() == 0, "CP10 Failed: Found pending rows"

    # Checkpoint 11: Missing values in clean orders
    assert clean_orders["order_amount"].isna().sum() == 0, "CP11 Failed: Missing order_amount found"
    comp_orders = clean_orders[clean_orders["status"] == "completed"]
    canc_orders = clean_orders[clean_orders["status"] == "cancelled"]
    assert comp_orders["delivery_minutes"].isna().sum() == 0, "CP11 Failed: Missing delivery_minutes in completed"
    assert canc_orders["delivery_minutes"].isna().sum() == 1199, "CP11 Failed: Expected 1199 missing delivery in cancelled"

    # Checkpoint 12: Completed revenue exact sum
    completed_revenue = round(float(comp_orders["order_amount"].sum()), 2)
    assert completed_revenue == 2998668.80, f"CP12 Failed: Expected 2998668.80, got {completed_revenue}"

    # Checkpoint 13: SQLite database validation
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    c_cnt = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    p_cnt = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    o_cnt = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert c_cnt == 1200, f"CP13 Failed: DB customers has {c_cnt} rows"
    assert p_cnt == 40, f"CP13 Failed: DB products has {p_cnt} rows"
    assert o_cnt == 12309, f"CP13 Failed: DB orders has {o_cnt} rows"
    con.close()

    # If all assertions pass:
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate()
