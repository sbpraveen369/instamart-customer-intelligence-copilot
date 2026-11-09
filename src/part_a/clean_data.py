"""
src/part_a/clean_data.py
Steps 4-6 — Clean Customers, Orders, Products

Saves:
  data/clean/customers.csv
  data/clean/orders.csv
  data/clean/products.csv
"""

import os
import pandas as pd

RAW_DIR   = "data/raw"
CLEAN_DIR = "data/clean"
os.makedirs(CLEAN_DIR, exist_ok=True)


# ── STEP 4: Clean Customers ────────────────────────────────────────────────────
def clean_customers():
    cust = pd.read_csv(f"{RAW_DIR}/customers.csv")

    # 1. Strip leading/trailing whitespace from city
    cust["city"] = cust["city"].str.strip()

    # 2. Apply title case
    cust["city"] = cust["city"].str.title()

    # 3. Verify exactly six distinct cities
    n_cities = cust["city"].nunique()
    assert n_cities == 6, f"Expected 6 cities, found {n_cities}: {cust['city'].unique()}"

    # 4. Normalize gender to male / female / unknown
    def normalize_gender(g):
        s = str(g).strip().lower()
        if s in {"female", "f"}:
            return "female"
        elif s in {"male", "m"}:
            return "male"
        else:
            return "unknown"

    cust["gender"] = cust["gender"].apply(normalize_gender)

    # 5. Treat ages outside 10–100 as missing
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = pd.NA

    # 6 & 7. Impute missing ages using the city median (groupby + transform)
    city_median = (
        cust.groupby("city")["age"]
        .transform(lambda x: x.median())
    )
    cust["age"] = cust["age"].fillna(city_median).round(0).astype(int)

    # 8. Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    # Save
    cust.to_csv(f"{CLEAN_DIR}/customers.csv", index=False)

    # Print summary
    print("=== CLEAN CUSTOMERS ===")
    print(f"Shape: {cust.shape}")
    print("City counts:")
    print(cust["city"].value_counts().to_string())
    print("\nGender counts:")
    print(cust["gender"].value_counts().to_string())
    print(f"\nMissing ages: {cust['age'].isna().sum()}")
    print(f"Saved: {CLEAN_DIR}/customers.csv\n")

    return cust


# ── STEP 5: Clean Orders ───────────────────────────────────────────────────────
def clean_orders():
    orders = pd.read_csv(f"{RAW_DIR}/orders.csv")

    # 1. Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])

    # 2. Remove exact duplicate rows
    orders = orders.drop_duplicates()

    # 3. For duplicate order_id records with different statuses, retain latest order_ts
    orders = (
        orders
        .sort_values("order_ts", ascending=False)
        .drop_duplicates(subset="order_id", keep="first")
        .reset_index(drop=True)
    )

    # 4. Convert order_amount to numeric
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")

    # 5. Fill missing order amounts: round(price_inr * quantity * (1 - discount_pct / 100), 2)
    mask_missing_amt = orders["order_amount"].isna()
    orders.loc[mask_missing_amt, "order_amount"] = (
        orders.loc[mask_missing_amt, "price_inr"]
        * orders.loc[mask_missing_amt, "quantity"]
        * (1 - orders.loc[mask_missing_amt, "discount_pct"] / 100)
    ).round(2)

    # 6. Set negative delivery_minutes to missing
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

    # 7. For completed orders, impute missing delivery_minutes using category median (groupby + transform)
    comp_mask = orders["status"] == "completed"
    cat_median = (
        orders.loc[comp_mask]
        .groupby("category")["delivery_minutes"]
        .transform(lambda x: x.median())
    )
    orders.loc[comp_mask & orders["delivery_minutes"].isna(), "delivery_minutes"] = (
        cat_median[comp_mask & orders["delivery_minutes"].isna()]
    )

    # 8. For cancelled orders, leave delivery_minutes missing (already done)

    # Save — store order_ts as YYYY-MM-DD HH:MM:SS text (for SQLite compatibility)
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    orders.to_csv(f"{CLEAN_DIR}/orders.csv", index=False)

    # Print summary
    print("=== CLEAN ORDERS ===")
    print(f"Shape: {orders.shape}")
    print("Status counts:")
    print(orders["status"].value_counts().to_string())
    print(f"\nMissing order_amount   : {orders['order_amount'].isna().sum()}")

    # Reload delivery_minutes as numeric for checks
    dm = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    comp = orders["status"] == "completed"
    canc = orders["status"] == "cancelled"
    print(f"Missing delivery_minutes (completed): {dm[comp].isna().sum()}")
    print(f"Missing delivery_minutes (cancelled): {dm[canc].isna().sum()}")

    rev = orders.loc[comp, "order_amount"].sum()
    print(f"\nCompleted revenue (INR): {rev:,.2f}")
    print(f"Saved: {CLEAN_DIR}/orders.csv\n")

    return orders


# ── STEP 6: Clean Products ─────────────────────────────────────────────────────
def clean_products():
    prod = pd.read_csv(f"{RAW_DIR}/products.csv")

    # Validate product IDs (should be non-null strings starting with 'P')
    invalid_ids = prod["product_id"].isna().sum()
    assert invalid_ids == 0, f"Found {invalid_ids} null product IDs"

    # Validate price_inr is numeric and positive
    prod["price_inr"] = pd.to_numeric(prod["price_inr"], errors="coerce")
    invalid_prices = prod["price_inr"].isna().sum()
    assert invalid_prices == 0, f"Found {invalid_prices} non-numeric prices"
    neg_prices = (prod["price_inr"] < 0).sum()
    assert neg_prices == 0, f"Found {neg_prices} negative prices"

    prod.to_csv(f"{CLEAN_DIR}/products.csv", index=False)

    print("=== CLEAN PRODUCTS ===")
    print(f"Shape: {prod.shape}")
    print(f"Invalid product IDs  : {invalid_ids}")
    print(f"Invalid price_inr    : {invalid_prices}")
    print(f"Saved: {CLEAN_DIR}/products.csv\n")

    return prod


def main():
    clean_customers()
    clean_orders()
    clean_products()
    print("All cleaning complete.")


if __name__ == "__main__":
    main()
