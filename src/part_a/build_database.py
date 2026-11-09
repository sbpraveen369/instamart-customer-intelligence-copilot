"""
src/part_a/build_database.py
Step 7 — Build SQLite Database

Creates data/instamart.db with tables:
  - customers
  - products
  - orders
Enforces primary key constraints, foreign key references, indexes,
and stores order_ts as 'YYYY-MM-DD HH:MM:SS' text.
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
    cur.execute("PRAGMA foreign_keys = ON;")

    # ── Table Schemas ──────────────────────────────────────────────────────────
    cur.executescript("""
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            city        TEXT,
            signup_date TEXT,
            age         INTEGER,
            gender      TEXT
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
            quantity         INTEGER,
            order_ts         TEXT,
            status           TEXT,
            discount_pct     REAL,
            order_amount     REAL,
            delivery_minutes REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id)  REFERENCES products(product_id)
        );

        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_status   ON orders(status);
        CREATE INDEX idx_orders_ts       ON orders(order_ts);
        CREATE INDEX idx_orders_product  ON orders(product_id);
    """)

    # ── Load and Insert Data ──────────────────────────────────────────────────
    customers = pd.read_csv(f"{CLEAN_DIR}/customers.csv")
    products  = pd.read_csv(f"{CLEAN_DIR}/products.csv")
    orders    = pd.read_csv(f"{CLEAN_DIR}/orders.csv")

    customers.to_sql("customers", con, if_exists="append", index=False)
    products.to_sql("products",   con, if_exists="append", index=False)
    orders.to_sql("orders",       con, if_exists="append", index=False)

    con.commit()

    # ── Verification ──────────────────────────────────────────────────────────
    print("=== DATABASE BUILD VERIFICATION ===")
    for tbl in ("customers", "products", "orders"):
        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        schema = cur.execute(f"PRAGMA table_info({tbl})").fetchall()
        pk = [c[1] for c in schema if c[5] > 0]
        print(f"Table {tbl:12s}: {cnt:6,d} rows | Primary Key: {pk}")

    indices = cur.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index'").fetchall()
    print(f"\nIndexes created ({len(indices)}):")
    for iname, tname in indices:
        if not iname.startswith("sqlite_autoindex"):
            print(f"  {iname} on {tname}")

    sample_ts = cur.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    print(f"\nSample order_ts format: {sample_ts!r}")

    con.close()
    print(f"\nDatabase successfully written to: {DB_PATH}")


if __name__ == "__main__":
    build_database()
