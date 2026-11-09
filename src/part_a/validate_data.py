"""
src/part_a/validate_data.py
Step 8 — Validation Script

Implements automated assertions for the reference checkpoints.
When every assertion passes, prints exactly:
ALL CHECKPOINTS PASSED
"""

import os
import sqlite3
import pandas as pd

CLEAN_DIR = "data/clean"
DB_PATH   = "data/instamart.db"


def validate():
    # ── 1. Load clean data ──────────────────────────────────────────────────
    assert os.path.exists(f"{CLEAN_DIR}/customers.csv"), "Missing clean customers.csv"
    assert os.path.exists(f"{CLEAN_DIR}/products.csv"), "Missing clean products.csv"
    assert os.path.exists(f"{CLEAN_DIR}/orders.csv"), "Missing clean orders.csv"
    assert os.path.exists(DB_PATH), "Missing database data/instamart.db"

    cust = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
    prod = pd.read_csv(f"{CLEAN_DIR}/products.csv")
    orders = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    # ── Checkpoint 1: Clean customers row count ────────────────────────────
    assert len(cust) == 1200, f"Checkpoint 1 Failed: Expected 1200 customers, got {len(cust)}"

    # ── Checkpoint 2: City distribution ────────────────────────────────────
    expected_cities = {
        "Pune": 227,
        "Delhi": 210,
        "Bengaluru": 197,
        "Chennai": 191,
        "Hyderabad": 191,
        "Mumbai": 184
    }
    city_counts = cust["city"].value_counts().to_dict()
    for city, count in expected_cities.items():
        assert city_counts.get(city) == count, (
            f"Checkpoint 2 Failed for {city}: expected {count}, got {city_counts.get(city)}"
        )
    assert len(city_counts) == 6, f"Expected 6 cities, got {len(city_counts)}"

    # ── Checkpoint 3: Gender distribution ──────────────────────────────────
    expected_genders = {
        "female": 610,
        "male": 560,
        "unknown": 30
    }
    gender_counts = cust["gender"].value_counts().to_dict()
    for gender, count in expected_genders.items():
        assert gender_counts.get(gender) == count, (
            f"Checkpoint 3 Failed for {gender}: expected {count}, got {gender_counts.get(gender)}"
        )

    # ── Checkpoint 4: Zero missing ages in customers ────────────────────────
    missing_ages = cust["age"].isna().sum()
    assert missing_ages == 0, f"Checkpoint 4 Failed: Expected 0 missing ages, got {missing_ages}"

    # ── Checkpoint 5: Clean orders row count ────────────────────────────────
    assert len(orders) == 12309, f"Checkpoint 5 Failed: Expected 12309 orders, got {len(orders)}"

    # ── Checkpoint 6: Order status distribution ────────────────────────────
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed") == 11110, (
        f"Checkpoint 6 Failed: Expected 11110 completed, got {status_counts.get('completed')}"
    )
    assert status_counts.get("cancelled") == 1199, (
        f"Checkpoint 6 Failed: Expected 1199 cancelled, got {status_counts.get('cancelled')}"
    )
    assert status_counts.get("pending", 0) == 0, (
        f"Checkpoint 6 Failed: Expected 0 pending, got {status_counts.get('pending')}"
    )

    # ── Checkpoint 7: Zero missing order amounts ───────────────────────────
    missing_amounts = orders["order_amount"].isna().sum()
    assert missing_amounts == 0, (
        f"Checkpoint 7 Failed: Expected 0 missing order amounts, got {missing_amounts}"
    )

    # ── Checkpoint 8: Delivery minutes completeness and nulls ──────────────
    comp_orders = orders[orders["status"] == "completed"]
    canc_orders = orders[orders["status"] == "cancelled"]
    dm_comp_na = comp_orders["delivery_minutes"].isna().sum()
    dm_canc_na = canc_orders["delivery_minutes"].isna().sum()
    assert dm_comp_na == 0, (
        f"Checkpoint 8 Failed: Expected 0 missing delivery_minutes for completed, got {dm_comp_na}"
    )
    assert dm_canc_na == 1199, (
        f"Checkpoint 8 Failed: Expected 1199 missing delivery_minutes for cancelled, got {dm_canc_na}"
    )

    # ── Checkpoint 9: Completed order revenue ──────────────────────────────
    comp_revenue = round(float(comp_orders["order_amount"].sum()), 2)
    assert comp_revenue == 2998668.80, (
        f"Checkpoint 9 Failed: Expected revenue 2998668.80, got {comp_revenue}"
    )

    # ── Checkpoint 10: Clean products validation ────────────────────────────
    assert len(prod) == 40, f"Checkpoint 10 Failed: Expected 40 products, got {len(prod)}"
    assert prod["price_inr"].isna().sum() == 0, "Null price_inr found in clean products"
    assert (prod["price_inr"] <= 0).sum() == 0, "Non-positive price_inr found in clean products"

    # ── Checkpoint 11: SQLite Database Verification ────────────────────────
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    c_cnt = cur.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    p_cnt = cur.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    o_cnt = cur.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    assert c_cnt == 1200, f"DB Checkpoint Failed: customers has {c_cnt} rows"
    assert p_cnt == 40, f"DB Checkpoint Failed: products has {p_cnt} rows"
    assert o_cnt == 12309, f"DB Checkpoint Failed: orders has {o_cnt} rows"
    con.close()

    # All checkpoints passed
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    validate()
