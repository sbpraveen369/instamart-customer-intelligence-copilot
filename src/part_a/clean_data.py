"""
src/part_a/clean_data.py
Steps 4–6 — Clean Customers, Orders, and Products

Implements exact documented cleaning transformations:
- Clean Customers: city whitespace/casing, gender normalization, city-median age imputation.
- Clean Orders: datetime conversion, exact deduplication, status resolution (latest order_ts),
  amount imputation via product price formula, negative delivery invalidation, category-median
  imputation for completed orders, preserving nulls for cancelled orders.
- Clean Products: validation of product IDs and positive prices in INR.

Saves clean CSVs to data/clean/.
"""

import os
import pandas as pd

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)


# ── STEP 4: Clean Customers ────────────────────────────────────────────────────
def clean_customers():
    cust = pd.read_csv(f"{RAW_DIR}/customers.csv")

    # (a) Standardise city by removing leading/trailing spaces and applying title case
    cust["city"] = cust["city"].str.strip().str.title()
    assert cust["city"].nunique() == 6, f"Expected 6 cities, got {cust['city'].nunique()}"

    # (b) Map every gender spelling to male, female, or unknown (blank -> unknown)
    def normalize_gender(g):
        if pd.isna(g) or str(g).strip() == "":
            return "unknown"
        s = str(g).strip().lower()
        if s in {"male", "m"}:
            return "male"
        elif s in {"female", "f"}:
            return "female"
        return "unknown"

    cust["gender"] = cust["gender"].apply(normalize_gender)

    # (c) Treat ages outside 10–100 as missing, then fill with city median age
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = pd.NA
    city_median_age = cust.groupby("city")["age"].transform("median")
    cust["age"] = cust["age"].fillna(city_median_age).round(0).astype(int)

    # (d) Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    # Ensure format and save
    out_cust = cust[["customer_id", "city", "signup_date", "age", "gender"]].copy()
    out_cust["signup_date"] = out_cust["signup_date"].dt.strftime("%Y-%m-%d")
    out_path = f"{CLEAN_DIR}/customers.csv"
    out_cust.to_csv(out_path, index=False)

    print("=== CLEAN CUSTOMERS ===")
    print(f"Shape: {out_cust.shape}")
    print("City distribution:\n", out_cust["city"].value_counts().to_string())
    print("\nGender distribution:\n", out_cust["gender"].value_counts().to_string())
    print(f"Missing ages: {out_cust['age'].isna().sum()} (min: {out_cust['age'].min()}, max: {out_cust['age'].max()})")
    print(f"Saved: {out_path}\n")

    return out_cust


# ── STEP 5: Clean Orders ───────────────────────────────────────────────────────
def clean_orders():
    orders = pd.read_csv(f"{RAW_DIR}/orders.csv")
    prods  = pd.read_csv(f"{RAW_DIR}/products.csv")

    # 1. Convert order_ts to a datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # 2. Remove exact duplicate rows
    initial_count = len(orders)
    orders = orders.drop_duplicates()
    exact_dups_removed = initial_count - len(orders)

    # 3. For duplicate order_id records with different statuses, keep latest order_ts
    orders = (
        orders.sort_values("order_ts", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        .reset_index(drop=True)
    )

    # 4. Convert order_amount to numeric, turning unparseable into missing, then fill
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

    # Merge product price and category for amount and delivery imputation
    orders = orders.merge(prods[["product_id", "category", "price_inr"]], on="product_id", how="left")

    mask_missing_amount = orders["order_amount"].isna()
    orders.loc[mask_missing_amount, "order_amount"] = (
        orders.loc[mask_missing_amount, "price_inr"]
        * orders.loc[mask_missing_amount, "quantity"]
        * (1 - orders.loc[mask_missing_amount, "discount_pct"] / 100)
    ).round(2)

    # 5. Set negative delivery_minutes to missing
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    # Impute missing delivery_minutes for completed orders using category median
    comp_mask = orders["status"] == "completed"
    cat_median_deliv = (
        orders.loc[comp_mask]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
    orders.loc[comp_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = (
        cat_median_deliv[comp_mask & orders["delivery_minutes"].isna()]
    )

    # Leave delivery_minutes missing for cancelled orders (preserves integrity)

    # Select standard orders columns
    cols = ["order_id", "customer_id", "product_id", "quantity", "order_ts",
            "status", "discount_pct", "order_amount", "delivery_minutes"]
    clean_orders_df = orders[cols].copy()
    clean_orders_df["order_ts"] = clean_orders_df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    out_path = f"{CLEAN_DIR}/orders.csv"
    clean_orders_df.to_csv(out_path, index=False)

    comp_orders = clean_orders_df[clean_orders_df["status"] == "completed"]
    canc_orders = clean_orders_df[clean_orders_df["status"] == "cancelled"]
    completed_revenue = comp_orders["order_amount"].sum()

    print("=== CLEAN ORDERS ===")
    print(f"Shape: {clean_orders_df.shape}")
    print(f"Exact duplicates removed : {exact_dups_removed}")
    print("Status counts:\n", clean_orders_df["status"].value_counts().to_string())
    print(f"Missing order_amount     : {clean_orders_df['order_amount'].isna().sum()}")
    print(f"Missing deliv (completed): {comp_orders['delivery_minutes'].isna().sum()}")
    print(f"Missing deliv (cancelled): {canc_orders['delivery_minutes'].isna().sum()}")
    print(f"Completed revenue (INR)  : {completed_revenue:,.2f}")
    print(f"Saved: {out_path}\n")

    return clean_orders_df


# ── STEP 6: Clean Products ─────────────────────────────────────────────────────
def clean_products():
    prods = pd.read_csv(f"{RAW_DIR}/products.csv")

    assert len(prods) == 40, f"Expected 40 products, got {len(prods)}"
    assert prods["product_id"].isna().sum() == 0, "Missing product_id found"
    prods["price_inr"] = pd.to_numeric(prods["price_inr"], errors="coerce")
    assert prods["price_inr"].isna().sum() == 0, "Non-numeric price_inr found"
    assert (prods["price_inr"] <= 0).sum() == 0, "Non-positive price_inr found"

    out_path = f"{CLEAN_DIR}/products.csv"
    prods.to_csv(out_path, index=False)

    print("=== CLEAN PRODUCTS ===")
    print(f"Shape: {prods.shape}")
    print(f"Categories ({prods['category'].nunique()}): {sorted(prods['category'].unique())}")
    print(f"Saved: {out_path}\n")

    return prods


def main():
    clean_customers()
    clean_orders()
    clean_products()
    print("Data cleaning successfully completed.")


if __name__ == "__main__":
    main()
