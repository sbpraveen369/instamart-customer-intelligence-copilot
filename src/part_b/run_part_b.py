"""src/part_b/run_part_b.py -- Part B: Customer Segments and Lapse Prediction.
Produces:
  outputs/customer_segments.csv
  outputs/lapse_model.joblib
  outputs/retention_list.csv
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_part_b():
    print("=== PART B: CUSTOMER SEGMENTS AND LAPSE PREDICTION ===")
    os.makedirs("outputs", exist_ok=True)
    
    # ---------------- 1. Load Data ----------------
    customers = pd.read_csv("data/clean/customers.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    
    # Filter completed orders
    completed_orders = orders[orders["status"] == "completed"].copy()
    
    # ---------------- 2. RFM Calculation ----------------
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    rfm = completed_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    
    print(f"RFM cohort size: {len(rfm)} customers")
    print(f"Recency - min: {rfm['recency'].min()}, median: {rfm['recency'].median()}, max: {rfm['recency'].max()}")
    print(f"Frequency - mean: {rfm['frequency'].mean():.2f}, max: {rfm['frequency'].max()}")
    print(f"Monetary - mean: {rfm['monetary'].mean():.2f}, max: {rfm['monetary'].max():.2f}")
    
    # ---------------- 3. K-Means Clustering ----------------
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    
    kmeans_results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(rfm_scaled)
        sil = silhouette_score(rfm_scaled, km.labels_)
        kmeans_results.append((k, km.inertia_, sil))
        print(f"K={k}: Inertia = {km.inertia_:.1f}, Silhouette = {sil:.3f}")
        
    # Fit final K=4
    final_km = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["cluster"] = final_km.fit_predict(rfm_scaled)
    
    # Profile clusters
    profiles = rfm.groupby("cluster").agg(
        size=("customer_id", "count"),
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean")
    ).reset_index()
    print("\nCluster Profiles (K=4):")
    print(profiles)
    
    # Identify Premium cluster: highest monetary_mean
    premium_cluster = profiles.loc[profiles["monetary_mean"].idxmax(), "cluster"]
    
    # Segment naming based on profile characteristics
    # Typically: high monetary = Premium, high freq/low rec = Loyal/Champions, high rec/low freq = At-Risk/Lapsed, moderate = Occasional
    segment_names = {}
    for _, row in profiles.iterrows():
        c = int(row["cluster"])
        if c == premium_cluster:
            segment_names[c] = "High-Value Premium"
        elif row["frequency_mean"] > 15:
            segment_names[c] = "Frequent Champions"
        elif row["recency_mean"] > 40:
            segment_names[c] = "At-Risk Inactive"
        else:
            segment_names[c] = "Steady Regulars"
            
    rfm["segment_id"] = rfm["cluster"]
    rfm["segment_name"] = rfm["cluster"].map(segment_names)
    rfm["is_premium"] = rfm["cluster"] == premium_cluster
    
    # PCA(n_components=2)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(rfm_scaled)
    evr = pca.explained_variance_ratio_
    print(f"\nPCA explained variance ratios: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, Total={evr.sum():.3f}")
    
    # Write customer_segments.csv
    segments_export = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    segments_export.to_csv("outputs/customer_segments.csv", index=False)
    print("Saved outputs/customer_segments.csv")
    
    # ---------------- 4. Lapse Dataset Construction ----------------
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    eval_end = pd.Timestamp("2025-09-30 23:59:59")
    
    # Orders before cutoff
    prior_orders = orders[orders["order_ts"] < cutoff].copy()
    prior_completed = prior_orders[prior_orders["status"] == "completed"].copy()
    
    # Cohort: at least 1 completed order before cutoff
    cohort_cids = prior_completed["customer_id"].unique()
    print(f"\nLapse Cohort: {len(cohort_cids)} customers")
    
    # Completed orders during September (eval window)
    sept_completed = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= eval_end) & (orders["status"] == "completed")]
    sept_active_cids = set(sept_completed["customer_id"].unique())
    
    # Build feature table for cohort
    products = pd.read_csv("data/clean/products.csv")
    prior_completed = prior_completed.merge(products[["product_id", "category"]], on="product_id", how="left")
    
    # Customer-level aggregations on completed orders before cutoff
    cust_rfm = prior_completed.groupby("customer_id").agg(
        last_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    ).reset_index()
    cust_rfm["recency"] = (cutoff - cust_rfm["last_ts"]).dt.days
    
    # Favourite category (mode)
    fav_cat = prior_completed.groupby(["customer_id", "category"]).size().reset_index(name="cat_count")
    fav_cat = fav_cat.sort_values(["customer_id", "cat_count"], ascending=[True, False]).drop_duplicates("customer_id")
    fav_cat = fav_cat.rename(columns={"category": "fav_category"})[["customer_id", "fav_category"]]
    
    # Cancellation rate before cutoff
    all_prior_final = prior_orders[prior_orders["status"].isin(["completed", "cancelled"])]
    canc_stats = all_prior_final.groupby("customer_id")["status"].agg(
        total_orders="count",
        cancelled_orders=lambda s: (s == "cancelled").sum()
    ).reset_index()
    canc_stats["cancellation_rate"] = canc_stats["cancelled_orders"] / canc_stats["total_orders"]
    
    # Merge features
    feat_df = pd.DataFrame({"customer_id": cohort_cids})
    feat_df = feat_df.merge(cust_rfm[["customer_id", "recency", "frequency", "monetary", "avg_discount", "avg_delivery_minutes"]], on="customer_id", how="left")
    feat_df = feat_df.merge(canc_stats[["customer_id", "cancellation_rate"]], on="customer_id", how="left")
    feat_df = feat_df.merge(fav_cat, on="customer_id", how="left")
    feat_df = feat_df.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
    feat_df["tenure_days"] = (cutoff - feat_df["signup_date"]).dt.days
    
    # Label: lapsed = 1 if NOT in sept_active_cids
    feat_df["lapsed"] = feat_df["customer_id"].apply(lambda cid: 1 if cid not in sept_active_cids else 0)
    
    print(f"Lapsed count: {feat_df['lapsed'].sum()} / {len(feat_df)} ({feat_df['lapsed'].mean()*100:.2f}%)")
    
    # Features list
    num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    cat_cols = ["fav_category", "city", "gender"]
    
    # Confirm no missing values
    assert feat_df[num_cols + cat_cols].isna().sum().sum() == 0, "Missing values found in feature set!"
    
    # Train test split
    X = feat_df[num_cols + cat_cols]
    y = feat_df["lapsed"]
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, feat_df["customer_id"], test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Baseline: predict 0
    base_acc = (y_test == 0).mean()
    base_f1 = f1_score(y_test, np.zeros_like(y_test), zero_division=0)
    print(f"Baseline (all 0) - Accuracy: {base_acc*100:.1f}%, Lapsed F1: {base_f1:.4f}")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols)
        ]
    )
    
    # Compare scaler learned mean vs cohort mean for frequency
    train_num_scaler = StandardScaler()
    train_num_scaler.fit(X_train[["frequency"]])
    print(f"Scaler learned mean of frequency: {train_num_scaler.mean_[0]:.4f} vs full-cohort mean: {feat_df['frequency'].mean():.4f}")
    
    # ---------------- 5. Model Tuning ----------------
    # Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    lr_grid = GridSearchCV(lr_pipe, lr_param_grid, cv=5, scoring="f1")
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"\nBest Logistic Regression: C={lr_grid.best_params_['clf__C']}, Best CV F1={lr_grid.best_score_:.4f}")
    
    # Random Forest Pipeline
    rf_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_param_grid = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    rf_grid = GridSearchCV(rf_pipe, rf_param_grid, cv=5, scoring="f1")
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print(f"Best Random Forest: max_depth={rf_grid.best_params_['clf__max_depth']}, min_samples_leaf={rf_grid.best_params_['clf__min_samples_leaf']}, Best CV F1={rf_grid.best_score_:.4f}")
    
    # Test evaluation
    y_pred_lr = best_lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)
    
    print("\n--- Test Set Evaluation ---")
    print("Logistic Regression:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}, Recall: {recall_score(y_test, y_pred_lr):.4f}, F1: {f1_score(y_test, y_pred_lr):.4f}")
    
    print("Random Forest:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}, Recall: {recall_score(y_test, y_pred_rf):.4f}, F1: {f1_score(y_test, y_pred_rf):.4f}")
    
    # ---------------- 6. Class Balance Experiment ----------------
    # Oversample minority class (lapsed=1) in training set
    train_combined = X_train.copy()
    train_combined["lapsed"] = y_train
    
    train_majority = train_combined[train_combined["lapsed"] == 0]
    train_minority = train_combined[train_combined["lapsed"] == 1]
    
    target_count = len(train_majority) # 570
    train_minority_oversampled = train_minority.sample(n=target_count, replace=True, random_state=42)
    train_balanced = pd.concat([train_majority, train_minority_oversampled], axis=0).sample(frac=1.0, random_state=42)
    
    X_train_bal = train_balanced[num_cols + cat_cols]
    y_train_bal = train_balanced["lapsed"]
    print(f"\nBalanced training set: {len(y_train_bal)} rows ({sum(y_train_bal==0)} non-lapsed, {sum(y_train_bal==1)} lapsed)")
    
    # Refit best LR on balanced data
    lr_bal_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=lr_grid.best_params_["clf__C"], max_iter=1000, random_state=42))
    ])
    lr_bal_pipe.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = lr_bal_pipe.predict(X_test)
    print("Balanced Logistic Regression Test Metrics:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_bal))
    print(f"Precision: {precision_score(y_test, y_pred_lr_bal):.4f}, Recall: {recall_score(y_test, y_pred_lr_bal):.4f}, F1: {f1_score(y_test, y_pred_lr_bal):.4f}")
    
    # ---------------- 7. Feature Interpretability & Persist ----------------
    # Feature importances of Random Forest
    rf_clf = best_rf.named_steps["clf"]
    ohe_cols = list(best_rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + ohe_cols
    
    importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("\nTop 5 Important Features (Random Forest):")
    print(importances.head(5))
    
    # Coefficients of Logistic Regression
    lr_clf = best_lr.named_steps["clf"]
    coefs = pd.Series(lr_clf.coef_[0], index=all_feature_names)
    print("\nSample Coefficients (Logistic Regression):")
    for feat in ["recency", "frequency", "monetary"]:
        if feat in coefs:
            print(f"{feat}: {coefs[feat]:.4f}")
            
    # Select best model: Random Forest (or higher F1 model)
    selected_model = best_rf if f1_score(y_test, y_pred_rf) >= f1_score(y_test, y_pred_lr) else best_lr
    joblib.dump(selected_model, "outputs/lapse_model.joblib")
    print("\nSaved selected model to outputs/lapse_model.joblib")
    
    # Verify reload
    loaded_model = joblib.load("outputs/lapse_model.joblib")
    loaded_preds = loaded_model.predict(X_test)
    assert np.array_equal(loaded_preds, selected_model.predict(X_test)), "Reloaded model failed verification!"
    print("Reloaded model verification passed.")
    
    # ---------------- 8. Retention List ----------------
    # Predict probabilities on test set
    test_probs = selected_model.predict_proba(X_test)[:, 1]
    test_df = pd.DataFrame({
        "customer_id": idx_test.values,
        "lapse_probability": test_probs,
        "actually_lapsed": y_test.values
    })
    
    # Merge whole-window is_premium flag
    test_df = test_df.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
    premium_test = test_df[test_df["is_premium"] == True].copy()
    retention_list = premium_test.sort_values(by="lapse_probability", ascending=False).head(10)
    retention_list = retention_list[["customer_id", "lapse_probability", "actually_lapsed"]]
    retention_list.to_csv("outputs/retention_list.csv", index=False)
    print("Saved outputs/retention_list.csv (Top 10 Premium test customers):")
    print(retention_list)
    print(f"Actually lapsed count in retention list: {retention_list['actually_lapsed'].sum()} / {len(retention_list)}")

if __name__ == "__main__":
    run_part_b()
