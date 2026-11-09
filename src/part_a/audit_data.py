"""
src/part_a/audit_data.py
Step 3 — Raw Data Audit

Loads raw CSVs, prints detailed audit statistics, and saves
outputs/audit_report.txt
"""

import os
import sys
import pandas as pd

RAW_DIR = "data/raw"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def load_raw():
    customers = pd.read_csv(f"{RAW_DIR}/customers.csv")
    products  = pd.read_csv(f"{RAW_DIR}/products.csv")
    orders    = pd.read_csv(f"{RAW_DIR}/orders.csv")
    tickets   = pd.read_csv(f"{RAW_DIR}/tickets.csv", keep_default_na=False)
    return customers, products, orders, tickets


def missing_summary(df, name):
    lines = [f"\n--- Missing values: {name} ---"]
    for col in df.columns:
        n = df[col].isna().sum()
        pct = n / len(df) * 100
        lines.append(f"  {col:30s}: {n:5d}  ({pct:.2f}%)")
    return "\n".join(lines)


def audit_customers(cust):
    lines = ["\n" + "=" * 60, "CUSTOMERS AUDIT", "=" * 60]
    lines.append(f"Shape   : {cust.shape}")
    lines.append(f"Dtypes  :\n{cust.dtypes.to_string()}")
    lines.append(missing_summary(cust, "customers"))

    # City counts (raw)
    lines.append("\nRaw city value counts:")
    for val, cnt in cust["city"].value_counts(dropna=False).items():
        lines.append(f"  {repr(val):20s}: {cnt}")

    # Gender counts (raw)
    lines.append("\nRaw gender value counts:")
    for val, cnt in cust["gender"].value_counts(dropna=False).items():
        lines.append(f"  {repr(val):20s}: {cnt}")

    # Age issues
    age_num = pd.to_numeric(cust["age"], errors="coerce")
    blank_age   = cust["age"].astype(str).str.strip().eq("").sum()
    invalid_age = ((age_num < 10) | (age_num > 100)).sum()
    lines.append(f"\nBlank ages         : {blank_age}")
    lines.append(f"Invalid ages (<10 or >100): {invalid_age}")
    lines.append(f"NaN ages (blank+invalid)  : {age_num.isna().sum()}")

    return "\n".join(lines)


def audit_products(prod):
    lines = ["\n" + "=" * 60, "PRODUCTS AUDIT", "=" * 60]
    lines.append(f"Shape   : {prod.shape}")
    lines.append(f"Dtypes  :\n{prod.dtypes.to_string()}")
    lines.append(missing_summary(prod, "products"))
    # Validate price_inr
    p = pd.to_numeric(prod["price_inr"], errors="coerce")
    lines.append(f"\nInvalid price_inr  : {p.isna().sum()}")
    lines.append(f"Negative price_inr : {(p < 0).sum()}")
    return "\n".join(lines)


def audit_orders(orders):
    lines = ["\n" + "=" * 60, "ORDERS AUDIT", "=" * 60]
    lines.append(f"Shape   : {orders.shape}")
    lines.append(f"Dtypes  :\n{orders.dtypes.to_string()}")
    lines.append(missing_summary(orders, "orders"))

    # Exact duplicates
    exact_dups = orders.duplicated().sum()
    lines.append(f"\nExact duplicate rows      : {exact_dups}")

    # Status breakdown
    lines.append("\nStatus counts (raw):")
    for s, c in orders["status"].value_counts(dropna=False).items():
        lines.append(f"  {s:15s}: {c}")

    # Pending rows
    pending = (orders["status"] == "pending").sum()
    lines.append(f"\nPending order rows        : {pending}")

    # Distinct order IDs
    distinct_oids = orders["order_id"].nunique()
    lines.append(f"Distinct order IDs        : {distinct_oids}")

    # Missing order_amount
    oa = pd.to_numeric(orders["order_amount"], errors="coerce")
    missing_amt = oa.isna().sum()
    lines.append(f"Missing order_amount      : {missing_amt}")

    # Negative / blank delivery_minutes
    dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    neg_deliv   = (dm < 0).sum()
    blank_deliv = orders["delivery_minutes"].astype(str).str.strip().eq("").sum()
    lines.append(f"Negative delivery_minutes : {neg_deliv}")
    lines.append(f"Blank delivery_minutes    : {blank_deliv}")

    return "\n".join(lines)


def audit_tickets(tickets):
    lines = ["\n" + "=" * 60, "TICKETS AUDIT", "=" * 60]
    lines.append(f"Shape   : {tickets.shape}")
    lines.append(f"Dtypes  :\n{tickets.dtypes.to_string()}")
    lines.append(missing_summary(tickets, "tickets"))
    return "\n".join(lines)


def main():
    cust, prod, orders, tickets = load_raw()

    report_parts = [
        "=" * 60,
        "INSTAMART — RAW DATA AUDIT REPORT",
        "=" * 60,
        audit_customers(cust),
        audit_products(prod),
        audit_orders(orders),
        audit_tickets(tickets),
        "\n" + "=" * 60,
        "END OF AUDIT REPORT",
        "=" * 60,
    ]

    report = "\n".join(report_parts)
    print(report)

    out_path = f"{OUT_DIR}/audit_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nAudit report saved to: {out_path}")


if __name__ == "__main__":
    main()
