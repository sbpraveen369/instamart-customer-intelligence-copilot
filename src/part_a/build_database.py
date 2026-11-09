"""
src/part_a/build_database.py
Step 7 — Build SQLite Database

Creates data/instamart.db with tables:
  customers, products, orders
Stores order_ts as 'YYYY-MM-DD HH:MM:SS' text.
Preserves primary key constraints, foreign keys, and indexes.
"""

import os
import sqlite3
import pandas as pd

CLEAN_DIR = "data/clean"
DB_PATH   = "data/instamart.db"


def build_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Enable foreign keys
    cur.execute("PRAGMA foreign_keys = ON;")

    # ── Create table schemas ───────────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE customers (
            customer_id  TEXT PRIMARY KEY,
            city         TEXT,
            gender       TEXT,
            age          INTEGER,
            signup_date  TEXT
        );

        CREATE TABLE products (
            product_id   TEXT PRIMARY KEY,
            product_name TEXT,
            category     TEXT,
            price_inr    REAL
        );

        CREATE TABLE orders (
            order_id         TEXT PRIMARY KEY,
            customer_id      TEXT,
            product_id       TEXT,
            category         TEXT,
            quantity         INTEGER,
            price_inr        REAL,
            discount_pct     REAL,
            order_amount     REAL,
            status           TEXT,
            order_ts         TEXT,
            delivery_minutes REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id)  REFERENCES products(product_id)
        );

        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_status   ON orders(status);
        CREATE INDEX idx_orders_ts       ON orders(order_ts);
        CREATE INDEX idx_orders_category ON orders(category);
    """)

    # ── Load clean CSVs ──────────────────────────────────────────────────────
    customers = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
    products  = pd.read_csv(f"{CLEAN_DIR}/products.csv")
    orders    = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    # Format order_ts and signup_date as clean strings
    orders["order_ts"] = pd.to_datetime(orders["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    customers["signup_date"] = pd.to_datetime(customers["signup_date"]).dt.strftime("%Y-%m-%d")

    # Append to existing schema to preserve constraints and indexes
    customers.to_sql("customers", con, if_exists="append", index=False)
    products.to_sql("products",   con, if_exists="append", index=False)
    orders.to_sql("orders",       con, if_exists="append", index=False)

    con.commit()

    # ── Verify ───────────────────────────────────────────────────────────────
    print("=== DATABASE VERIFICATION ===")
    for tbl in ("customers", "products", "orders"):
        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        schema = cur.execute(f"PRAGMA table_info({tbl})").fetchall()
        pk_cols = [c[1] for c in schema if c[5] > 0]
        print(f"Table {tbl:12s}: {cnt:6,d} rows | PK: {pk_cols}")

    indices = cur.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'").fetchall()
    print(f"\nIndexes created ({len(indices)}):")
    for idx_name, tbl_name in indices:
        if not idx_name.startswith("sqlite_autoindex"):
            print(f"  {idx_name} on {tbl_name}")

    sample_ts = cur.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    print(f"\nSample order_ts format: {sample_ts!r}")

    con.close()
    print(f"\nDatabase built successfully: {DB_PATH}")


if __name__ == "__main__":
    build_database()
