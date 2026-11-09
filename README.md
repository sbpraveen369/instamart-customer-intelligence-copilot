# Instamart Customer Intelligence & Support Copilot
**100-Mark Academic Capstone Project**

---

## 1. Overview

The **Instamart Customer Intelligence & Support Copilot** is an enterprise data and AI system designed for a multi-city quick-commerce grocery delivery service operating in six Indian metro cities (Chennai, Mumbai, Bengaluru, Delhi, Hyderabad, Pune). Quick-commerce platforms face three critical operational bottlenecks as order volume scales: unreliable raw order data exports (duplicate records, inconsistent formatting, unparseable entries, planted anomalies), lack of predictive customer retention tools to identify lapsing cohorts, and customer support desks overwhelmed by high-volume unstructured tickets.

This repository provides an end-to-end reproducible solution structured across four sequential, interconnected parts:
- **Part A — Data Foundation: Clean, Store, Query**: Audits dirty raw exports, executes documented deterministic cleaning rules, constructs an indexed SQLite analytical database (`data/instamart.db`), validates reference checkpoints, and conducts comprehensive SQL and exploratory data analysis.
- **Part B — Customer Segments and Lapse Prediction**: Computes customer-level RFM metrics, performs K-means clustering and PCA profiling, formulates a strictly leakage-free lapse prediction target, tunes Logistic Regression and Random Forest models with GridSearchCV, evaluates on an untouched test set, executes class-balancing experiments, and exports a prioritized retention list.
- **Part C — Support Ticket Triage with Guardrails**: Implements input guardrails (empty/long filters), regex-based PII redaction (email, phone, order ID), zero-shot and few-shot classification with mock LLM calls, exponential backoff retries with jitter for transient errors, strict JSON and keyword output validation, token budget throttling, and priority ticket routing driven by Part B segments.
- **Part D — Policy RAG Assistant and Refund Agent**: Constructs an offline RAG policy question-answering assistant over company policies using 40-word sliding chunk windows and TF-IDF retrieval. Builds an autonomous ReAct refund agent invoking SQLite and policy tools across seven customer scenarios, demonstrates token and timeout safety limits, verifies crash-and-resume checkpointing in `state/`, and provides architectural comparison between ReAct and Plan-and-Execute patterns.

All monetary values throughout the repository are strictly in **Indian Rupees (INR)**; no dollar signs or foreign denominations are used. The project strictly adheres to a **text-only submission policy**: code, tables, and numeric analyses are documented entirely in markdown and text files without committing any images, charts, or multimedia assets.

---

## 2. Reproduce

To reproduce the entire project end-to-end from a clean environment, execute the following numbered shell commands:

```bash
# 1. Create an isolated virtual environment with Python 3.11 or newer
python -m venv .venv

# 2. Activate the virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.\.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# 3. Upgrade pip and install required dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Generate the deterministic raw dataset using the supplied unmodified generator
python generate_data.py

# 5. Execute the entire pipeline end-to-end via the Master Runner (Parts A -> B -> C -> D)
python run_all.py

# Alternatively, execute each part individually in sequence:
# Part A:
python src/part_a/audit_data.py
python src/part_a/clean_data.py
python src/part_a/build_database.py
python src/part_a/validate_data.py
python src/part_a/sql_analysis.py
python src/part_a/exploratory_analysis.py

# Part B:
python src/part_b/run_part_b.py

# Part C:
python src/part_c/run_part_c.py

# Part D:
python src/part_d/run_part_d.py

# 6. Run automated test suites
python -m unittest tests/test_part_a.py
python tests/test_pipeline.py
```

---

## 3. Part A — Data Foundation: Clean, Store, Query

### 3.1 Raw Data Generation & Audit Summary
The raw data is generated deterministically (`SEED = 42`) by `generate_data.py` into `data/raw/` and `data/policies/`. The audit script (`src/part_a/audit_data.py`) verified every raw checkpoint:

| Audit Checkpoint Metric | Measured Value | Brief Reference Standard | Match Status |
| :--- | :--- | :--- | :--- |
| **Raw orders rows (`orders.csv`)** | 13,084 | 13,084 | **Exact Match** |
| **Exact duplicate order rows** | 119 | 119 | **Exact Match** |
| **Pending status order rows** | 656 | 656 | **Exact Match** |
| **Distinct `order_id` values** | 12,309 | 12,309 | **Exact Match** |
| **Non-numeric `order_amount` entries** | 120 | 120 | **Exact Match** |
| **Negative `delivery_minutes` entries** | 106 | 106 | **Exact Match** |
| **Blank `delivery_minutes` entries** | 2,541 | 2,541 | **Exact Match** |
| **Blank `age` entries (`customers.csv`)** | 61 | 61 | **Exact Match** |
| **Age values outside 10–100** | 6 | 6 | **Exact Match** |
| **Distinct raw city spellings** | 24 | 24 | **Exact Match** |
| **Distinct non-blank raw gender spellings** | 10 | 10 | **Exact Match** |
| **Blank gender entries** | 30 | 30 | **Exact Match** |

Full audit details are persisted in `outputs/audit_report.txt`.

### 3.2 Cleaning Rules & Rows Changed
Executed in strict order by `src/part_a/clean_data.py`:
1. **Customers Cleaning**:
   - **City Standardization**: Stripped leading/trailing whitespace and converted to title case. Standardized 24 raw variations down to exactly 6 distinct cities: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
   - **Gender Normalization**: Mapped 10 raw gender spellings to `male` (560), `female` (610), and blank/unrecognized to `unknown` (30).
   - **Age Imputation**: 6 impossible ages outside 10–100 (-4, 0, 150, 200, 280, 999) and 61 blank ages (67 total) were treated as missing, then imputed using the customer's city median age via `groupby('city')['age'].transform('median')`. Result: 0 missing ages, all ages between 18 and 60.
   - **Signup Date**: Parsed into standard date format (`YYYY-MM-DD`).
2. **Orders Cleaning**:
   - **Rule 1 — Timestamp Conversion**: Converted `order_ts` to datetime format.
   - **Rule 2 — Exact Deduplication**: Dropped exactly 119 duplicate rows (count reduced from 13,084 to 12,965).
   - **Rule 3 — Status Conflict Resolution**: For multiple records sharing the same `order_id` (earlier pending state vs. final state), sorted by `order_ts` descending and retained only the record with the latest timestamp. Resolved all 656 pending rows, yielding exactly 12,309 unique order records (11,110 completed and 1,199 cancelled; 0 pending).
   - **Rule 4 — Monetary Imputation**: Converted `order_amount` to numeric, setting 120 unparseable entries (`'N/A'`, `'error'`) to null. Imputed missing values using catalog pricing:
     $$\text{order\_amount} = \text{round}\left(\text{price\_inr} \times \text{quantity} \times \left(1 - \frac{\text{discount\_pct}}{100}\right), 2\right)$$
     Result: 0 missing order amounts across all 12,309 rows.
   - **Rule 5 — Delivery Time Imputation**: Set 106 negative delivery minutes (-10) to missing. For completed orders, filled all missing delivery times with the median delivery time of that product category among completed orders (via `groupby('category')['delivery_minutes'].transform('median')`). Cancelled orders were left missing (1,199 rows) because cancelled orders were never delivered.
   - **Completed Revenue**: Sum of `order_amount` over completed orders is **INR 2,998,668.80**.
3. **Products Validation**:
   - Confirmed 40 products across 5 categories (Dairy, Electronics, Grocery, Personal Care, Snacks) with positive prices.

### 3.3 Reference Checkpoints Verification Table

| Checkpoint Identifier | Reference Value | Clean Implementation Value | Status |
| :--- | :--- | :--- | :--- |
| Clean Customer Count | 1,200 | 1,200 | **MATCH** |
| Clean Customer Cities | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 | **MATCH** |
| Clean Customer Genders | Female 610, Male 560, Unknown 30 | Female 610, Male 560, Unknown 30 | **MATCH** |
| Clean Customer Ages | 0 missing, all ages 18–60 | 0 missing, min 18, max 60 | **MATCH** |
| Clean Order Count | 12,309 rows | 12,309 rows | **MATCH** |
| Completed Orders | 11,110 rows | 11,110 rows | **MATCH** |
| Cancelled Orders | 1,199 rows | 1,199 rows | **MATCH** |
| Pending Orders | 0 rows | 0 rows | **MATCH** |
| Missing Order Amounts | 0 missing | 0 missing | **MATCH** |
| Completed Delivery Nulls | 0 missing | 0 missing | **MATCH** |
| Cancelled Delivery Nulls | 1,199 missing | 1,199 missing | **MATCH** |
| Completed Revenue | INR 2,998,668.80 | INR 2,998,668.80 | **MATCH** |
| Validation Output | ALL CHECKPOINTS PASSED | ALL CHECKPOINTS PASSED | **MATCH** |

### 3.4 SQLite Database Architecture (`data/instamart.db`)
Constructed by `src/part_a/build_database.py`:
- `customers` table (`customer_id` TEXT PRIMARY KEY, `city`, `signup_date`, `age`, `gender`)
- `products` table (`product_id` TEXT PRIMARY KEY, `product_name`, `category`, `price_inr`)
- `orders` table (`order_id` TEXT PRIMARY KEY, `customer_id`, `product_id`, `quantity`, `order_ts`, `status`, `discount_pct`, `order_amount`, `delivery_minutes`)
- Indexes: `idx_orders_customer`, `idx_orders_status`, `idx_orders_ts`, `idx_orders_product`
- Foreign keys enforced: `orders.customer_id -> customers.customer_id`, `orders.product_id -> products.product_id`
- Timestamp storage format: text formatted as `YYYY-MM-DD HH:MM:SS`.

### 3.5 SQL Analytical Findings (`sql/queries.sql`)
All seven queries executed via `pd.read_sql_query` in `src/part_a/sql_analysis.py`:
- **Q1 (WHERE then HAVING)**: Completed orders with `order_amount > 500`, grouped by customer, having total $> 8000$. Result: **30 customers** (top: C0693 at INR 14,310.20, C0291 at INR 13,742.60).
- **Q2 (Three-Table Join)**: City $\times$ Category completed revenue breakdown. Result: **30 rows**; total sums to **INR 2,998,668.80**; largest combination is **Bengaluru / Electronics** at **INR 233,909.45**.
- **Q3 (Scalar Subquery)**: Products priced above average catalog price (mean catalog price = INR 341.22). Result: **11 products**.
- **Q4 (LEFT JOIN)**: Customers who have no completed order. Result: **25 customers**.
- **Q5 (Common Table Expression)**: Customers with total completed spend above population average (INR 2,552.06). Result: **550 customers**.
- **Q6 (Ranking Window Functions)**: Computed `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` partitioned by city ordered by total spend descending.
  - Top customer per city:
    - **Bengaluru**: C0082 (INR 12,856.20)
    - **Chennai**: C0930 (INR 10,151.40)
    - **Delhi**: C0291 (INR 13,742.60)
    - **Hyderabad**: C0743 (INR 12,730.05)
    - **Mumbai**: C0693 (INR 14,766.55)
    - **Pune**: C0426 (INR 10,377.10)
  - Exactly **8 rows** have `RANK()` different from `ROW_NUMBER()`.
  - *Explanation of difference*: When customers tie on total spend, `RANK()` assigns identical rank numbers and skips subsequent ranks, whereas `ROW_NUMBER()` assigns strictly distinct sequential integers, causing the ranks to diverge across tied customer spend totals within city partitions.
- **Q7 (Daily Running Total)**: Daily completed revenue across 183 calendar days (2025-04-01 to 2025-09-30) with cumulative window revenue. Result: **183 rows**; final cumulative total is **INR 2,998,668.80**.

### 3.6 Pandas Cross-Check & Exploratory Data Analysis
Computed by `src/part_a/exploratory_analysis.py`:
- **Pivot Table Cross-Check**: Grand total = **INR 2,998,668.80**, matching SQL total exactly.
- **City Groupby Named Aggregation**:
  - **Bengaluru**: sum INR 523,643.40, mean 276.47, count 1,894
  - **Chennai**: sum INR 463,629.15, mean 272.72, count 1,700
  - **Delhi**: sum INR 502,829.80, mean 243.62, count 2,064
  - **Hyderabad**: sum INR 528,589.90, mean 284.95, count 1,855
  - **Mumbai**: sum INR 479,058.50, mean 297.00, count 1,613
  - **Pune**: sum INR 500,918.05, mean 252.48, count 1,984
- **Distribution & Outlier Metrics**:
  - Mean completed order amount: **INR 269.91** | Median: **INR 135.00**
  - Skewness: **4.61** (substantially right-skewed)
  - $Q_1$: **INR 64.80** | $Q_3$: **INR 297.00** | $\text{IQR}$: **INR 232.20**
  - Lower fence: **INR -283.50** (0 lower outliers)
  - Upper fence: **INR 645.30** (**897 upper outliers**)
- **Correlation Matrix (Completed Orders)**:
  - $\text{Corr}(\text{order\_amount}, \text{discount\_pct}) = -0.0951$ (largest in absolute value)
  - $\text{Corr}(\text{order\_amount}, \text{quantity}) = +0.0873$
  - $\text{Corr}(\text{order\_amount}, \text{delivery\_minutes}) = -0.0031$
  - $\text{Corr}(\text{quantity}, \text{discount\_pct}) = -0.0004$

> **Business Questions Synthesis (153 words)**:
> 1. **What is this data about?** It captures 183 days of quick-commerce transactions across six Indian metros, covering daily essentials, groceries, personal care, and high-value electronics.
> 2. **What do distributions look like?** Order value is heavily right-skewed (skewness 4.61), where the mean (INR 269.91) is double the median (INR 135.00), driven by routine low-ticket staples alongside occasional premium electronics.
> 3. **Are there outliers or anomalies?** With Q1 at INR 64.80, Q3 at INR 297.00, and an upper fence of INR 645.30, there are exactly 897 upper outliers (multi-item orders and electronics like earbuds/power banks) and zero lower outliers. Planted negative delivery minutes were successfully sanitized.
> 4. **What relationships exist?** The strongest correlation with order_amount is discount_pct at about -0.095, followed by quantity at +0.087. Crucially, a correlation of this modest magnitude **does not establish causation**: discounts reflect promotional mechanics rather than customer willingness-to-pay, requiring holistic retention modeling.

---

## 4. Part B — Customer Segments and Lapse Prediction

### 4.1 RFM Feature Engineering
RFM metrics were calculated on all customers with at least one completed order as of snapshot moment `2025-10-01 00:00:00`:
- **Cohort Size**: 1,175 customers
- **Recency**: Whole days between snapshot and latest completed order. Minimum = 0, Median = 19.0, Maximum = 180.
- **Frequency**: Distinct completed orders. Mean = 9.46, Maximum = 33.
- **Monetary**: Total completed spend. Mean = INR 2,552.06, Maximum = INR 14,766.55.

### 4.2 K-Means Clustering & Model Selection
RFM features were standardized with `StandardScaler()`. `KMeans(n_clusters=K, n_init=10, random_state=42)` was fitted for $K \in \{2..8\}$:

| Clusters ($K$) | Inertia | Silhouette Score | Evaluation & Notes |
| :---: | :---: | :---: | :--- |
| **2** | 1820.8 | 0.465 | Under-segmented; mixes high-spend with moderate-frequency users |
| **3** | 1268.2 | 0.445 | Moderate cohesion; misses distinct high-value champions |
| **4** | **743.1** | **0.526** | **Optimal Elbow & Highest Silhouette Score; clear business profiles** |
| **5** | 617.4 | 0.480 | Diminishing separation; fragments dormant users |
| **6** | 503.2 | 0.482 | Granular; lower interpretability |
| **7** | 408.5 | 0.416 | Severe cluster fragmentation |
| **8** | 366.5 | 0.412 | Over-clustering; lowest silhouette |

*Inertia Trade-off Explanation*: Inertia monotonically decreases with higher $K$ because adding cluster centroids mechanically shortens the distance from points to their nearest centroid. However, selecting $K=8$ purely for lower inertia leads to severe overfitting and trivial groupings without operational utility. $K=4$ represents the distinct elbow inflection point and delivers the global maximum silhouette score ($0.526$).

### 4.3 Cluster Profiles, Naming & PCA Projection
For $K=4$, the segments exhibit well-defined behavioral profiles:

| Cluster | Size | Recency (days) | Frequency | Monetary (INR) | Business Segment Name | Recommended Business Action |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | 436 | 28.86 | 3.77 | 1,057.62 | Steady Regulars | Category cross-sell & basket-building discounts |
| **1** | 138 | 24.12 | 6.77 | **7,129.70** | **High-Value Premium** | Dedicated account manager, VIP delivery perks |
| **2** | 419 | 8.77 | 19.40 | 3,410.02 | Frequent Champions | Loyalty subscription, early access to new launches |
| **3** | 182 | 109.04 | 2.20 | 685.99 | At-Risk Inactive | Aggressive win-back campaigns & push re-engagement |

- **Designated Premium Segment**: Cluster 1 (138 customers) carries the highest mean monetary spend (INR 7,129.70) and is designated as `is_premium = True`.
- **PCA Projection**: `PCA(n_components=2)` applied to scaled RFM yields explained variance ratios of **0.634** (PC1) and **0.218** (PC2), capturing **85.2%** of total variance.
- Outputs saved to `outputs/customer_segments.csv`.

### 4.4 Leakage-Free Lapse Problem Formulation
- **Cutoff Date**: `2025-09-01 00:00:00`.
- **Target Definition**: Customers with $\ge 1$ completed order prior to cutoff. Label `lapsed = 1` if customer placed 0 completed orders in evaluation window `[2025-09-01, 2025-09-30]`, else `0`.
- **Cohort Size**: 1,156 customers (443 lapsed, **38.32%** base lapse rate).
- **Leakage Prevention**: Strictly zero orders on or after `2025-09-01` were used for feature extraction. The whole-window RFM segment labels were never exposed to the model.
- **Features Included**: Recency before cutoff, frequency before cutoff, monetary spend before cutoff, average discount percentage, cancellation rate, average delivery minutes, favorite product category (mode), city, age, gender, and customer tenure in days.
- **Split**: `train_test_split(test_size=0.2, stratify=y, random_state=42)` yielding **924 training rows** and **232 test rows**.
- **Baseline Model (Always Predict Non-Lapsed)**: Accuracy = **61.6%**, Lapsed F1 = **0.0000**.
- **Learned Scaler Mean Check**: Scaler fitted strictly on training data yielded frequency mean **8.0671** vs. full cohort mean **8.0285** (differing because the scaler never observes the test partition).

### 4.5 Model Tuning & Test Set Evaluation
Pipelines incorporated `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder(drop='first', handle_unknown='ignore')` on categorical variables, tuned via 5-fold cross-validation on the training set:
- **Logistic Regression**: `GridSearchCV` over $C \in \{0.01, 0.1, 1, 10, 100\}$. Best $C=100$, Best CV F1 = **0.6313**.
- **Random Forest**: `GridSearchCV` over `max_depth` $\in \{3, 5, 8\}$ and `min_samples_leaf` $\in \{5, 10, 20\}$ (45 fits). Best parameters: `max_depth=5`, `min_samples_leaf=20`, Best CV F1 = **0.6742**.

#### Test Set Evaluation on Untouched Data (232 rows, 89 lapsed):

| Model Architecture | Test Accuracy | Lapsed Precision | Lapsed Recall | Lapsed F1 Score | Confusion Matrix [TN, FP / FN, TP] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Majority Class)** | 61.6% | 0.0000 | 0.0000 | 0.0000 | [143, 0 / 89, 0] |
| **Tuned Logistic Regression** | 72.4% | 0.6374 | 0.6517 | 0.6444 | [110, 33 / 31, 58] |
| **Tuned Random Forest** | **76.7%** | **0.7011** | **0.6854** | **0.6932** | **[117, 26 / 28, 61]** |

*Model Selection Justification*: The Tuned Random Forest model was selected as the production model (`outputs/lapse_model.joblib`). In quick-commerce retention, failing to identify a lapsing customer (False Negative) leads to permanent customer churn and lost lifetime value, whereas wrongly targeting an active customer with a retention coupon (False Positive) merely incurs small marketing overhead. Random Forest delivers both superior precision (0.7011 vs 0.6374) and recall (0.6854 vs 0.6517), yielding a test F1 of 0.6932 (comfortably exceeding the 0.55 requirement).

### 4.6 Class-Balance Experiment
Oversampling the minority lapsed class in the training partition to 570 rows each (1,140 balanced training rows) and refitting Logistic Regression produced:
- **Balanced LR Test Metrics**: Precision = **0.6154**, Recall = **0.8090**, F1 Score = **0.6990** (Confusion Matrix: `[98, 45 / 17, 72]`).
- *Finding*: Recall jumped dramatically from 0.6517 to **0.8090** (+15.7 percentage points) while precision experienced only a minor drop (0.6374 to 0.6154). For the retention desk, this provides a highly effective lever when operational priority shifts towards catching more than 80% of lapsing customers.

### 4.7 Feature Interpretability & Retention Prioritization
- **Top 5 Random Forest Features**: `monetary` (0.308), `frequency` (0.304), `recency` (0.104), `cancellation_rate` (0.060), `avg_delivery_minutes` (0.043). High historical spend and purchase frequency are the strongest signals of ongoing commitment.
- **Logistic Regression Key Coefficients**:
  - `frequency`: **-1.7860** (Higher purchase frequency strongly reduces lapse probability).
  - `recency`: **-0.0483** (Recent ordering moderately reduces lapse risk).
  - `monetary`: **+0.3616** (Higher monetary volume exhibits slight positive coefficient once conditioned on frequency due to basket composition dynamics).
- **Model Persistence**: Serialized to `outputs/lapse_model.joblib`. Reloading and re-evaluating produced exact matching predictions.
- **Prioritized Retention List**: Top 10 Premium test set customers ranked by lapse probability were exported to `outputs/retention_list.csv`:

| Customer ID | Predicted Lapse Probability | Actually Lapsed |
| :---: | :---: | :---: |
| **C0267** | 0.5054 | 0 |
| **C0816** | 0.4964 | 0 |
| **C0060** | 0.4871 | 0 |
| **C0743** | 0.4668 | 1 |
| **C0017** | 0.4580 | 0 |
| **C1132** | 0.4496 | 1 |
| **C0802** | 0.4390 | 0 |
| **C0098** | 0.4355 | 0 |
| **C0888** | 0.4295 | 0 |
| **C1127** | 0.4282 | 0 |

*(Note: The Premium flag is derived from the full 183-day RFM clustering, so this retention list is illustrative and intended for operational targeting rather than leakage-free model benchmarking).*

---

## 5. Part C — Support Ticket Triage with Guardrails

### 5.1 Input Guardrails & PII Redaction
- **Input Guardrails**:
  - Empty or whitespace tickets $\rightarrow$ `skipped_empty`: **4 tickets** (T0157–T0160). Never sent to model.
  - Estimated tokens $> 400$ $\rightarrow$ `flagged_long`: **4 tickets** (T0161–T0164, each estimated at 734 tokens). Never sent to model.
- **PII Redaction**:
  - Regex patterns applied sequentially before model submission:
    - Email: `[\w.+-]+@[\w-]+\.[\w.]+` $\rightarrow$ `[EMAIL]` (**55 substitutions**)
    - Phone: `\b[6-9]\d{9}\b` $\rightarrow$ `[PHONE]` (**91 substitutions**)
    - Order ID: `ORD-\d{6}` $\rightarrow$ `[ORDER_ID]` (**132 substitutions**)
  - **Total Substitutions**: Exactly **278**.
  - Code assertions verified 0 residual PII matches remained. The prompt user message contained strictly the redacted text.

### 5.2 Resilient Execution & Transient Error Recovery
- Calls wrapped in exponential backoff retry handler with jitter:
  $$\text{wait} = 0.05 \times 2^n \times \text{Uniform}(0.9, 1.15)$$
- Calling `mock_llm.reset()` before full run: Exactly **19 TransientLLMErrors (503 server busy)** were caught and successfully recovered across all 156 eligible tickets. Zero tickets failed.

### 5.3 Output Validation & Classification Performance
Replies were validated against the four allowed categories (`billing`, `delivery`, `technical`, `fraud`):
- `classified_json`: **135 tickets** (valid JSON with valid category).
- `classified_keyword`: **12 tickets** (parsed chatty responses using `re.findall(r"[a-z]+", text.lower())`).
- `flagged_invalid_output`: **9 tickets** (routed to human review due to unallowed categories like `shipping`).
- Total Classified: **147 tickets**.

#### Comparison of Zero-Shot vs. Few-Shot Pipelines:

| Pipeline Configuration | Tickets Classified | Classification Accuracy | Total Tokens Consumed | Tokens per Classified Ticket |
| :--- | :---: | :---: | :---: | :---: |
| **Zero-Shot Prompt** | 147 | **85.71%** (126 / 147) | 12,052 tokens | 82.0 tokens |
| **Few-Shot Prompt** | 147 | **100.00%** (147 / 147) | 18,916 tokens | 128.7 tokens |

- **Zero-Shot Confusion Matrix (Classified Tickets)**:
  - Fraud tickets predicted as billing: 11
  - Technical tickets predicted as billing: 10
  - Billing & Delivery: 100% precision/recall.
- *ROI of Few-Shot Prompting*: Few-shot prompting costs 46.7 additional tokens per ticket (+57% token overhead), but completely eliminates severe classification errors (11 fraud tickets misclassified as billing). In quick-commerce, misrouting a fraud case to a standard billing queue incurs substantial financial loss and SLA breach, making the token overhead well worth the investment.

### 5.4 Token Budget & Segment-Aware Ticket Routing
- **Budget Demonstration Run** (`outputs/triage_budget_demo.csv`):
  - With `TOKEN_BUDGET = 6000`, the triage pipeline automatically halted after processing 86 tickets when cumulative tokens hit 6,046. Exactly **78 tickets** received status `budget_exhausted` and were routed to human review.
- **Production Ticket Routing** (`outputs/triage_results.csv`):
  - Driven by Few-Shot classification and customer segments from Part B:
    - `human_review` (Priority NORMAL): **17 tickets** (4 empty, 4 long, 9 invalid output).
    - `security_team` (Priority HIGH): **34 tickets** (all classified fraud cases).
    - `premium_desk` (Priority HIGH): **11 tickets** (High-Value Premium customers with non-fraud issues).
    - `support_queue` (Priority NORMAL): **102 tickets** (standard routing).
  - Total: **164 tickets** processed with zero unhandled errors.

---

## 6. Part D — Policy RAG Assistant and Refund Agent

### 6.1 Policy Document Chunking & TF-IDF Retrieval
- Five policy documents in `data/policies/` were segmented using 40-word sliding windows with 10-word overlap:
  - `refund_policy.txt`: 4 chunks
  - `delivery_policy.txt`: 3 chunks
  - `escalation_policy.txt`: 3 chunks
  - `payment_policy.txt`: 3 chunks
  - `privacy_policy.txt`: 2 chunks
  - **Total**: Exactly **15 chunks**.
- **Retrieval Engine**: Fitted `TfidfVectorizer()` with cosine similarity.
  - *TF (Term Frequency)* measures how frequently a term appears within a specific policy chunk.
  - *IDF (Inverse Document Frequency)* measures the informative specificity of a term across all chunks by downweighting ubiquitous terms.
  - *Limitation*: TF-IDF relies purely on lexical keyword overlap; semantic synonyms or paraphrased queries with zero exact word matches cannot be retrieved.
  - *Retrieval Sanity Check*: Query `"refund policy late delivery"` returned top 3 chunks, all from `refund_policy.txt` (scores: 0.2719, 0.2514, 0.2454).

### 6.2 Grounded Policy Assistant Evaluation
Evaluated over 14 benchmark questions (`data/raw/policy_eval.csv`):

| Evaluation Metric | Overlap = 10 Words (Default) | Overlap = 0 Words (Ablation) | Target Reference |
| :--- | :---: | :---: | :---: |
| **Answerable Doc Match in Top 3** | **12 / 12 (100%)** | 12 / 12 (100%) | 12 / 12 |
| **Answerable Fact Substring Match** | **12 / 12 (100%)** | **10 / 12 (83.3%)** | 12 / 12 |
| **Unanswerable Fallback Match** | **2 / 2 (100%)** | 2 / 2 (100%) | 2 / 2 |

- *Ablation Analysis*: With zero chunk overlap, two questions broke:
  1. *"What delivery time makes a completed order eligible for a full refund?"*
  2. *"How soon is money returned when a payment fails but is deducted?"*
  - *Cause*: Critical semantic phrases (e.g. `"delivered in more than 45 minutes"`) were fractured across chunk boundary splits, demonstrating the necessity of sliding overlaps in document chunking.

### 6.3 Tools Specification & ReAct Refund Agent
The ReAct agent operates with five registered tools:
1. `get_order(order_id)`: Fetches status, amount, and delivery time from `data/instamart.db`.
2. `search_policy(query)`: Semantic TF-IDF search returning top policy snippets.
3. `calculate_refund(order_amount, delivery_minutes)`: Entitlement calculation (100% refund if completed and $> 45$ min; 0 otherwise).
4. `issue_refund(order_id, refund_inr)`: Logs to `outputs/refund_ledger.jsonl` and executes refund.
5. `escalate_to_human(order_id, refund_inr)`: Logs to `outputs/refund_ledger.jsonl` for refunds $> \text{INR } 1000$.

### 6.4 Seven Agent Scenarios Benchmark Results
Evaluated over `data/raw/agent_scenarios.csv`:

| Scenario ID | Message / Context | Agent Status | Expected Outcome | Refund Amount (INR) | Model Calls | Tokens Used |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | Late delivery ($> 45$ min, amount $\le 1000$) | **REFUND** | REFUND | **128.25** | 5 | 972 |
| **S2** | Late delivery ($> 45$ min, amount $\le 1000$) | **REFUND** | REFUND | **135.00** | 5 | 972 |
| **S3** | Late delivery ($> 45$ min, amount $> 1000$) | **ESCALATE** | ESCALATE | **1,999.00** | 5 | 959 |
| **S4** | On-time delivery ($\le 45$ min) | **DENIED** | DENIED | 0.00 | 4 | 665 |
| **S5** | Cancelled order | **DENIED** | DENIED | 0.00 | 3 | 382 |
| **S6** | Non-existent order ID | **NOT_FOUND** | NOT_FOUND | 0.00 | 2 | 102 |
| **S7** | Circular loop prompt (`[LOOP]`) | **FALLBACK_MAX_ITERATIONS** | FALLBACK | 0.00 | 6 | 3,090 |

- All scenario outcomes and refund amounts matched expectations within 0.01 tolerance.
- Full trace logs saved to `outputs/agent_runs.jsonl`; financial actions logged to `outputs/refund_ledger.jsonl`.

### 6.5 Safety Limits & Crash-and-Resume State Persistence
1. **Safety Limit Demonstrations**:
   - **Token Limit Demonstration** (`token_limit=200` on S1): Model stopped immediately before tool execution with status **`FALLBACK_TOKEN_LIMIT`**.
   - **Timeout Demonstration** (`time_limit_seconds=0` on S1): Stopped before first model call with status **`FALLBACK_TIMEOUT`**.
2. **Crash-and-Resume Demonstration**:
   - Simulated crash triggered on S1 immediately after tool 2 (`search_policy`), saving checkpoint in `state/s1_crash_resume_demo.json`.
   - Executing `resume_agent("s1_crash_resume_demo")` successfully loaded state and continued.
   - **Tools Executed in Order**:
     1. `get_order` (Pre-crash)
     2. `search_policy` (Pre-crash)
     3. `calculate_refund` (Post-resume)
     4. `issue_refund` (Post-resume)
   - Exactly **four tools executed** across both phases with zero duplicated calls. Final status: **`REFUND: INR 128.25 issued.`**

### 6.6 ReAct vs. Plan-and-Execute Architecture Comparison (218 words)
In our empirical quick-commerce evaluation, the ReAct agent required 5 model calls for complex valid refund workflows (S1, S2, S3), 3 calls for cancelled orders (S5), and only 2 calls for missing orders (S6). 

A Plan-and-Execute pattern prompts the language model once upfront to output an entire multi-step DAG of operations, executing them deterministically and only replanning upon failure. For linear, static workflows, Plan-and-Execute consumes fewer model calls (1 planning call plus 1 final summarization) and lower aggregate prompt token overhead. However, in customer support environments where runtime observations dynamically dictate branching logic, Plan-and-Execute is brittle: encountering an unexpected `NOT_FOUND` order (S6) or a `cancelled` status (S5) invalidates the remaining planned steps (e.g., policy search and refund calculation), forcing an expensive replan.

Conversely, ReAct dynamically conditions every next action on the immediate prior tool observation. In S6, the agent immediately halted upon seeing `{"error": "order not found"}` after just 102 tokens, saving 3 subsequent tool executions. For quick-commerce refund automation, **ReAct is distinctly superior**: its step-by-step grounding prevents hallucinated financial transactions, cleanly respects real-time database state, and ensures reliable state checkpointing between operations.

---

## 7. Limitations and Next Steps

### Known Limitations
1. **Post-Hoc Whole-Window Segment Labeling**: As stated in Section 4.7, the `is_premium` flag in `outputs/customer_segments.csv` is derived from the full 183-day transactional RFM table (April–September 2025), which encompasses the September evaluation window. Consequently, `outputs/retention_list.csv` represents an illustrative operational prioritization list rather than a strictly leakage-free benchmark. In production, segmentation must be computed on a rolling trailing window prior to the prediction cutoff.
2. **Lexical Keyword Constraints in Retrieval**: The TF-IDF retrieval system operates on exact term frequencies and inverse document frequencies. While highly effective on domain-specific policy texts with consistent vocabulary, it lacks semantic embedding representations to capture novel phrasing or cross-lingual user vernacular.
3. **Synthetic Environment Assumptions**: The dataset is deterministically simulated to guarantee reproducibility. While modeling quick-commerce dynamics with high fidelity, real-world deployments must account for live payment gateway webhooks, rider GPS telemetry streams, and inventory stockouts.

### Next Steps & Production Roadmap
1. **Dense Semantic Embeddings**: Upgrade policy retrieval to hybrid dense-sparse search (e.g. BGE or text-embedding-3 combined with BM25) to handle multilingual and colloquial refund inquiries.
2. **Online Learning & Drift Detection**: Implement Kolmogorov-Smirnov statistical tests on order values and customer frequency to detect macroeconomic demand shifts and model feature drift.
3. **Automated Human-in-the-Loop Orchestration**: Connect the `human_review` and `escalate_to_human` sinks to an enterprise ticketing system (e.g. Zendesk or Freshdesk API) with automated webhook notifications and manager approval dashboards.
