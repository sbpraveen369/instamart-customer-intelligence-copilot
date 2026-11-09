# Instamart Customer Intelligence & Support Copilot
**100-Mark Academic Capstone Project**

---

## 1. Overview

The **Instamart Customer Intelligence & Support Copilot** is an enterprise data and AI solution engineered for quick-commerce grocery delivery platforms. Quick-commerce demands hyper-local, sub-30-minute logistics, tight inventory turnover across urban dark stores, and proactive, empathetic customer issue resolution.

This repository implements the multi-tier system across four distinct development phases:
- **Part A — Data Foundation: Clean, Store & Query** *(Current Phase: Implemented & Verified)*
- **Part B — Customer Intelligence & RFM Segmentation** *(Upcoming Phase)*
- **Part C — Support Copilot & Policy RAG Agent** *(Upcoming Phase)*
- **Part D — Interactive Deployment & Analytics Dashboard** *(Upcoming Phase)*

All monetary metrics are consistently modeled in **Indian Rupees (INR)**. The foundation leverages Python 3.11+, Pandas, NumPy, SciPy, and SQLite to guarantee deterministic, auditable, and reproducible data workflows.

---

## 2. Reproduce

Execute the following numbered commands sequentially in your shell to replicate the complete Part A environment and data pipeline from scratch:

```bash
# 1. Create a Python 3.11+ virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate the raw synthetic dataset deterministically
python generate_data.py

# 5. Run the raw data audit and export audit report
python src/part_a/audit_data.py

# 6. Execute data cleaning transformations on Customers, Orders, and Products
python src/part_a/clean_data.py

# 7. Construct SQLite relational database with primary keys and indexes
python src/part_a/build_database.py

# 8. Run automated reference checkpoint assertions (Must output: ALL CHECKPOINTS PASSED)
python src/part_a/validate_data.py

# 9. Execute all seven SQL analytical queries
python src/part_a/sql_analysis.py

# 10. Run Pandas statistical cross-checks, EDA, and export summary charts
python src/part_a/exploratory_analysis.py

# 11. Run the automated test suite
python -m unittest tests/test_part_a.py
```

---

## 3. Part A — Data Foundation: Clean, Store & Query

### 3.1 Raw Data Generation & Audit
Synthetic quick-commerce transactional data is deterministically seeded (`SEED = 42`) in `generate_data.py`:
- `customers.csv`: 1,200 records
- `products.csv`: 40 items across 10 product categories
- `orders.csv`: 13,084 raw transaction records
- `tickets.csv`: 164 support tickets

The audit script (`src/part_a/audit_data.py`) identified key data hygiene flaws:
1. **Customers**: Leading whitespace in city strings (e.g. `' Pune'`, `' Bengaluru'`), inconsistent casing, 11 missing genders, and 28 out-of-range ages (< 10 or > 100).
2. **Orders**: 541 exact duplicate rows, 244 pending/in-flight orders, 234 duplicate `order_id` status conflicts, 221 missing `order_amount` values, and 182 negative delivery durations.
3. **Products**: 40 distinct items with valid positive prices in INR.

Full audit text is logged to [`outputs/audit_report.txt`](file:///C:/COLLEGE/PROJECTS/INSTAMART/outputs/audit_report.txt).

### 3.2 Cleaning Specifications & Business Rules
Implemented in `src/part_a/clean_data.py` adhering to strict sequential requirements:

1. **Customers**:
   - Stripped leading/trailing whitespace and applied title casing.
   - Verified exactly six metro cities: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
   - Normalized gender to `female` (610), `male` (560), or `unknown` (30).
   - Flagged ages outside 10–100 as missing, then imputed using the city median age via `groupby('city')['age'].transform('median')`.

2. **Orders**:
   - Converted `order_ts` to datetime.
   - Removed 541 exact duplicate rows.
   - Resolved conflicting `order_id` records across different statuses by retaining the record with the latest timestamp.
   - Converted `order_amount` to numeric, imputing missing amounts using:
     $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
   - Replaced negative delivery minutes with missing values (`pd.NA`).
   - Imputed missing delivery times for completed orders using the category median delivery time among completed orders via `groupby('category')['delivery_minutes'].transform('median')`.
   - Preserved missing delivery times for cancelled orders (1,199 rows).

3. **Products**:
   - Validated product IDs and verified positive unit prices in INR.

### 3.3 Reference Checkpoints Verification
Every reference milestone specified in the assignment is validated by `src/part_a/validate_data.py`:

| Checkpoint Identifier | Assignment Reference Standard | Actual Implemented Value | Verification Status |
| :--- | :--- | :--- | :--- |
| **Total Clean Customers** | 1,200 rows | 1,200 rows | **PASSED** |
| **City Distribution** | Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184 | 227 / 210 / 197 / 191 / 191 / 184 | **PASSED** |
| **Gender Distribution** | Female: 610, Male: 560, Unknown: 30 | 610 / 560 / 30 | **PASSED** |
| **Missing Ages (Clean)** | Zero missing | 0 | **PASSED** |
| **Total Clean Orders** | 12,309 rows | 12,309 rows | **PASSED** |
| **Order Status Breakdown** | 11,110 Completed, 1,199 Cancelled, 0 Pending | 11,110 Completed, 1,199 Cancelled, 0 Pending | **PASSED** |
| **Missing Order Amounts** | Zero missing | 0 | **PASSED** |
| **Delivery Time Nulls** | 0 missing in completed; 1,199 missing in cancelled | 0 missing in completed; 1,199 missing in cancelled | **PASSED** |
| **Completed Revenue** | **INR 2,998,668.80** | **INR 2,998,668.80** | **PASSED** |
| **Validation Script Output** | `ALL CHECKPOINTS PASSED` | `ALL CHECKPOINTS PASSED` | **PASSED** |

### 3.4 SQLite Database Architecture (`data/instamart.db`)
Constructed by `src/part_a/build_database.py`:
- `customers` table (`customer_id` TEXT PRIMARY KEY, `city`, `gender`, `age`, `signup_date`)
- `products` table (`product_id` TEXT PRIMARY KEY, `product_name`, `category`, `price_inr`)
- `orders` table (`order_id` TEXT PRIMARY KEY, foreign keys to `customers` and `products`, indexed on `customer_id`, `status`, `order_ts`, and `category`). `order_ts` is stored in standard `YYYY-MM-DD HH:MM:SS` format.

### 3.5 SQL Analytical Findings (`sql/queries.sql`)
Executed via `pandas.read_sql_query` in `src/part_a/sql_analysis.py`:
1. **Q1 (WHERE & HAVING)**: Pet Supplies (INR 385,295.52) and Personal Care (INR 383,892.89) lead in completed revenue among high-volume categories (> 1,000 orders), whereas Dairy (1,154 orders) and Household (1,134 orders) lead in transaction frequency.
2. **Q2 (Three-Table JOIN)**: Evaluated city-by-category revenue matrices. Pune and Delhi represent the largest regional markets across all product segments.
3. **Q3 (Scalar Subquery)**: Identified 13 premium products priced above the catalog average price (INR 112.23), led by P019 (Pet Supplies, INR 186.99) and P029 (Pet Supplies, INR 170.41).
4. **Q4 (LEFT JOIN)**: Returned 0 records, proving that all 1,200 registered users made at least one completed order (100% activation rate).
5. **Q5 (CTE)**: Identified 559 high-value customers with lifetime spend exceeding the customer population mean (INR 2,498.89), led by C0016 in Chennai (INR 7,085.82 across 20 orders).
6. **Q6 (Window Rankings)**: Calculated `ROW_NUMBER`, `RANK`, and `DENSE_RANK` partitions per city. In Bengaluru, customer C0402 holds rank #1 (INR 5,326.74).
7. **Q7 (Running Total)**: Window cumulative sum smoothly accumulates across 731 calendar days (2023-01-01 to 2024-12-31), closing at exactly **INR 2,998,668.80**.

### 3.6 Exploratory Data Analysis & Business Synthesis
Executed by `src/part_a/exploratory_analysis.py`:
- **Completed Basket Metrics**: Mean order amount = INR 269.91; Median order amount = INR 229.16. Positive right-skewness of 0.9322.
- **IQR Outlier Boundaries**: Q1 = INR 127.62, Q3 = INR 374.18, IQR = INR 246.56. Upper fence = INR 744.02, identifying 165 premium multi-item bulk baskets (1.49%).
- **Correlation Insights**: Order amount strongly correlates with item quantity ($r = 0.701$) and unit price ($r = 0.610$), but weakly with discount percentage ($r = -0.155$).

> **Business Synthesis (Core Questions)**:
> 1. **Delivery SLA & Reliability**: Median delivery across completed orders is stable (~50 min), with zero negative anomalies and category-median imputation ensuring reliable dark-store logistics.
> 2. **Category Contribution**: Pet Supplies and Personal Care yield top revenues (INR 385.3k and 383.9k), driven by higher unit prices, whereas Groceries and Dairy drive frequent transaction volume.
> 3. **Basket Disparity & Outliers**: The mean basket (INR 269.91) exceeds median (INR 229.16) with positive skewness (0.932) and 165 upper-fence outliers (> INR 744.02), reflecting premium bulk purchases.
> 4. **Promotion Dynamics**: Discount percentage displays a slight negative correlation with order amount ($r = -0.155$) and near-zero correlation with quantity ($r = 0.005$). Crucially, **correlation does not establish causation**: promotional discounts alone do not cause organic volume growth, requiring targeted retention rather than blanket discounting.

---

## 4. Part B — Customer Intelligence & RFM Segmentation
*Status: Pending Approval — Implementation will proceed upon explicit instruction.*

- **Objectives**: Construct customer-level RFM (Recency, Frequency, Monetary) feature stores, segment dark-store cohorts, analyze churn propensity, and establish predictive churn indicators.
- **Planned Modules**: `src/part_b/rfm_segmentation.py`, `src/part_b/customer_features.py`.

---

## 5. Part C — Support Copilot & Policy RAG Agent
*Status: Pending Approval — Implementation will proceed upon explicit instruction.*

- **Objectives**: Build a retrieval-augmented generation (RAG) customer support agent using `mock_llm.py`, semantic ticket routing, sentiment classification, and domain policy ingestion from `data/policies/`.
- **Planned Modules**: `src/part_c/copilot_agent.py`, `src/part_c/policy_retriever.py`.

---

## 6. Part D — Interactive Deployment & Dashboard
*Status: Pending Approval — Implementation will proceed upon explicit instruction.*

- **Objectives**: Develop an operational intelligence dashboard (Streamlit / FastAPI) providing dark-store monitoring, live ticket triage simulation, and real-time query exploration.
- **Planned Modules**: `src/part_d/app.py`, `src/part_d/api.py`.

---

## 7. Limitations and Next Steps

### Current Limitations
1. **Synthetic Simulation**: Data is synthetically generated under a seeded distribution. While matching all mathematical constraints and real-world quick-commerce profiles, it lacks unobserved exogenous shocks (e.g., severe weather events, payment gateway outages).
2. **Fixed Catalog Scope**: The product catalog is capped at 40 SKUs across 10 categories for benchmark standardization.
3. **Imputation Heuristic**: Category-median delivery imputation assumes within-category homogeneity in packing and transit times.

### Next Steps
1. Obtain approval to initiate **Part B: Customer Intelligence & RFM Segmentation**.
2. Build the RFM scoring engine and cluster customers into loyalty tiers (Champions, Loyal, At-Risk, Churned).
3. Connect the cleaned SQLite database directly to Part B analytical pipelines.
