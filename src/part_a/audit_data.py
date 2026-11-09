"""
src/part_a/audit_data.py
Step 3 — Raw Data Audit

Audits raw datasets from data/raw/:
- customers.csv, products.csv, orders.csv, tickets.csv
Reports dataset shapes, dtypes, missing counts/percentages,
and the reference audit figures required by the assignment brief.
Outputs formatted audit report to outputs/audit_report.txt.
"""

import os
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


def column_missing_summary(df, name):
    lines = [f"\n--- Column Missing Value Summary: {name} ---"]
    for col in df.columns:
        n = df[col].isna().sum()
        pct = (n / len(df)) * 100
        lines.append(f"  {col:25s}: {n:5d} ({pct:6.2f}%)")
    return "\n".join(lines)


def run_audit():
    customers, products, orders, tickets = load_raw()

    # ── Orders Audit Metrics ──────────────────────────────────────────────────
    raw_orders_rows = len(orders)
    exact_duplicates = orders.duplicated().sum()
    pending_rows = (orders["status"] == "pending").sum()
    distinct_order_ids = orders["order_id"].nunique()
    oa_num = pd.to_numeric(orders["order_amount"], errors="coerce")
    non_numeric_order_amounts = oa_num.isna().sum()
    dm_num = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative_delivery_minutes = (dm_num < 0).sum()
    blank_delivery_minutes = orders["delivery_minutes"].isna().sum()

    # ── Customers Audit Metrics ───────────────────────────────────────────────
    blank_ages = customers["age"].isna().sum()
    age_num = pd.to_numeric(customers["age"], errors="coerce")
    outside_range_ages = ((age_num < 10) | (age_num > 100)).sum()
    distinct_raw_cities = customers["city"].nunique()
    distinct_nonblank_genders = customers["gender"].dropna().nunique()
    blank_genders = customers["gender"].isna().sum()

    # ── Build Output Report ───────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("INSTAMART RAW DATA AUDIT REPORT")
    lines.append("=" * 80)

    lines.append("\n1. DATASET OVERVIEW & SHAPES")
    lines.append(f"  customers.csv : {customers.shape[0]:,} rows, {customers.shape[1]} columns")
    lines.append(f"  products.csv  : {products.shape[0]:,} rows, {products.shape[1]} columns")
    lines.append(f"  orders.csv    : {orders.shape[0]:,} rows, {orders.shape[1]} columns")
    lines.append(f"  tickets.csv   : {tickets.shape[0]:,} rows, {tickets.shape[1]} columns")

    lines.append("\n2. DATA TYPES & MISSING VALUES")
    for df, name in [(customers, "customers"), (products, "products"),
                     (orders, "orders"), (tickets, "tickets")]:
        lines.append(f"\n[{name.upper()} DATA TYPES]")
        lines.append(df.dtypes.to_string())
        lines.append(column_missing_summary(df, name))

    lines.append("\n" + "=" * 80)
    lines.append("3. MANDATORY AUDIT CHECKPOINTS TABLE")
    lines.append("=" * 80)
    lines.append(f"{'Audit Metric':<45} | {'Measured Value':<15} | {'Reference Target':<15}")
    lines.append("-" * 80)
    lines.append(f"{'Raw orders rows':<45} | {raw_orders_rows:<15} | {'13,084':<15}")
    lines.append(f"{'Exact duplicate rows in orders':<45} | {exact_duplicates:<15} | {'119':<15}")
    lines.append(f"{'Pending status rows in orders':<45} | {pending_rows:<15} | {'656':<15}")
    lines.append(f"{'Distinct order_id count in orders':<45} | {distinct_order_ids:<15} | {'12,309':<15}")
    lines.append(f"{'Non-numeric order_amount entries':<45} | {non_numeric_order_amounts:<15} | {'120':<15}")
    lines.append(f"{'Negative delivery_minutes entries':<45} | {negative_delivery_minutes:<15} | {'106':<15}")
    lines.append(f"{'Blank delivery_minutes entries':<45} | {blank_delivery_minutes:<15} | {'2,541':<15}")
    lines.append(f"{'Blank age entries in customers':<45} | {blank_ages:<15} | {'61':<15}")
    lines.append(f"{'Age values outside 10–100 in customers':<45} | {outside_range_ages:<15} | {'6':<15}")
    lines.append(f"{'Distinct raw city spellings in customers':<45} | {distinct_raw_cities:<15} | {'24':<15}")
    lines.append(f"{'Distinct non-blank raw gender spellings':<45} | {distinct_nonblank_genders:<15} | {'10':<15}")
    lines.append(f"{'Blank gender entries in customers':<45} | {blank_genders:<15} | {'30':<15}")
    lines.append("=" * 80)

    report_content = "\n".join(lines)
    print(report_content)

    out_file = f"{OUT_DIR}/audit_report.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"\nAudit report successfully saved to: {out_file}")


if __name__ == "__main__":
    run_audit()
