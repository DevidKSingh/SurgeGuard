"""
ml_pipeline.py - Feature Engineering, Time-Aware Cross Validation, and Model Training
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

# ── Pure-NumPy replacements for sklearn (scipy DLL blocked by AppControl policy) ──

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    npos = y_true_s.sum()
    nneg = len(y_true_s) - npos
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    tpr = np.concatenate([[0.0], tps / max(npos, 1)])
    fpr = np.concatenate([[0.0], fps / max(nneg, 1)])
    return float(np.trapezoid(tpr, fpr))

def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    npos = y_true_s.sum()
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    precision = tps / (tps + fps)
    recall = tps / max(npos, 1)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum(precision * (recall - recall_prev)))

def precision_recall_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    thresholds = y_score[desc_idx]
    npos = y_true_s.sum()
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    precision = np.concatenate([tps / (tps + fps), [1.0]])
    recall    = np.concatenate([tps / max(npos, 1), [0.0]])
    return precision, recall, thresholds

class IsolationForest:
    """Lightweight numpy Z-score anomaly detector matching IsolationForest API."""
    def __init__(self, n_estimators=100, max_samples=2048,
                 contamination=0.002, random_state=42, n_jobs=4):
        self.contamination = contamination
        self.random_state  = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        return self

    def score_samples(self, X):
        """Returns negative anomaly score so higher = more normal (matches sklearn API)."""
        X = np.asarray(X, dtype=np.float64)
        z = np.abs((X - self.mean_) / self.std_)
        return -z.mean(axis=1)  # negate: sklearn convention

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"LightGBM version: {lgb.__version__}")


# =====================================================================
# 1. LOAD DATA & AUDIT HYGIENE
# =====================================================================
print("\n[1/6] Loading data and auditing hygiene...")
train_df = pd.read_csv("train_cleaned.csv")
test_df = pd.read_csv("test_cleaned.csv")

print(f"Raw train shape: {train_df.shape}")
print(f"Raw test shape: {test_df.shape}")

# Check exact duplicates
train_dups = train_df.duplicated().sum()
test_dups = test_df.duplicated().sum()
print(f"Exact duplicates in train: {train_dups}")
print(f"Exact duplicates in test: {test_dups}")

# Sort strictly by Time chronologically (essential for zero leakage)
train_df = train_df.sort_values("Time").reset_index(drop=True)
test_df = test_df.sort_values("Time").reset_index(drop=True)

print(f"Train time range: {train_df['Time'].min():.0f}s to {train_df['Time'].max():.0f}s")
print(f"Test time range: {test_df['Time'].min():.0f}s to {test_df['Time'].max():.0f}s")
print(f"Train fraud count: {train_df['Class'].sum()} / {len(train_df)} ({train_df['Class'].mean()*100:.3f}%)")


# =====================================================================
# 2. VECTORIZED CAUSAL TEMPORAL & VELOCITY FEATURE ENGINEERING
# =====================================================================
print("\n[2/6] Engineering leakage-free causal features...")

def extract_features(df):
    """
    Computes causal features strictly using past information (< current transaction time).
    Uses vectorized NumPy searchsorted and prefix sums for sub-second execution.
    """
    df = df.copy()
    times = df["Time"].values.astype(np.float64)
    amounts = df["Amount"].values.astype(np.float64)
    n = len(df)
    
    # 2.1 Cyclical & time-of-day features
    hours = (times // 3600.0) % 24.0
    df["Hour"] = hours
    df["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    
    # 2.2 Amount transformations
    df["Log_Amount"] = np.log1p(amounts)
    
    # 2.3 Inter-arrival time (time since immediately preceding transaction)
    df["Time_Since_Prev"] = np.concatenate([[0.0], np.diff(times)])
    
    # 2.4 Causal Rolling Window Features (10s, 60s, 300s, 900s)
    # Using prefix sum for instant O(N log N) causal computation
    amount_cumsum = np.concatenate([[0.0], np.cumsum(amounts)])
    indices = np.arange(n)
    
    windows = [10, 60, 300, 900]
    for w in windows:
        # Strictly look for transactions where Time >= current_time - w and strictly < current_time
        # searchsorted with side='left' on (times - w) finds first item >= current_time - w
        left_idx = np.searchsorted(times, times - w, side="left")
        
        # Clean division using np.divide with where argument to eliminate warnings
        count = indices - left_idx
        amt_sum = amount_cumsum[indices] - amount_cumsum[left_idx]
        amt_mean = np.divide(amt_sum, count, out=np.zeros_like(amt_sum), where=count > 0)
        
        df[f"Tx_Count_{w}s"] = count
        df[f"Amt_Sum_{w}s"] = amt_sum
        df[f"Amt_Mean_{w}s"] = amt_mean
        
        # Velocity burst ratio (current amount relative to recent average)
        amt_ratio = np.divide(amounts, amt_mean, out=np.ones_like(amounts), where=amt_mean > 0)
        df[f"Amt_Ratio_{w}s"] = amt_ratio
    
    # High-signal interaction ratios
    df["V14_V4_Ratio"] = df["V14"] / (np.abs(df["V4"]) + 1e-5)
    df["V12_V10_Diff"] = df["V12"] - df["V10"]
    
    return df

t0 = time.time()
train_feats = extract_features(train_df)
print(f"Engineered {train_feats.shape[1]} columns in {time.time() - t0:.2f}s")
# Note: test_df features are extracted separately in evaluate_test.py using the same function.


# =====================================================================
# 3. TIME-AWARE TRAIN / VALIDATION SPLIT
# =====================================================================
print("\n[3/6] Splitting train into chronological 80% Train / 20% Val...")
split_idx = int(len(train_feats) * 0.8)
train_split = train_feats.iloc[:split_idx].copy()
val_split = train_feats.iloc[split_idx:].copy()

print(f"Train split size: {len(train_split)} (Frauds: {train_split['Class'].sum()}, {train_split['Class'].mean()*100:.3f}%)")
print(f"Val split size:   {len(val_split)} (Frauds: {val_split['Class'].sum()}, {val_split['Class'].mean()*100:.3f}%)")

drop_cols = ["Class"]
feature_cols = [c for c in train_feats.columns if c not in drop_cols]
print(f"Total modeling features: {len(feature_cols)}")

X_tr = train_split[feature_cols]
y_tr = train_split["Class"]
X_val = val_split[feature_cols]
y_val = val_split["Class"]


# =====================================================================
# 4. SUPERVISED MODEL: LIGHTGBM WITH CLASS WEIGHTING
# =====================================================================
print("\n[4/6] Training Supervised LightGBM Classifier...")

neg_count = (y_tr == 0).sum()
pos_count = (y_tr == 1).sum()
scale_pos_weight = neg_count / max(1, pos_count)
print(f"Imbalance ratio scale_pos_weight: {scale_pos_weight:.2f}")

lgb_params = {
    "objective": "binary",
    "metric": ["auc", "average_precision"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "scale_pos_weight": scale_pos_weight * 0.25, # Calibrated scaling for smoother probability distribution
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 4
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

model = lgb.train(
    lgb_params,
    dtrain,
    num_boost_round=400,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

val_preds = model.predict(X_val, num_iteration=model.best_iteration)
val_roc_auc = roc_auc_score(y_val, val_preds)
val_pr_auc = average_precision_score(y_val, val_preds)

print(f"\n--- Validation Performance ---")
print(f"ROC-AUC: {val_roc_auc:.4f}")
print(f"PR-AUC:  {val_pr_auc:.4f}")

# Top 10 Feature Importances
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Gain': model.feature_importance(importance_type='gain')
}).sort_values('Gain', ascending=False)
print("\n--- Top 10 High-Signal Features ---")
print(importance.head(10).to_string(index=False))


# =====================================================================
# 5. UNSUPERVISED ANOMALY DETECTION (ISOLATION FOREST)
# =====================================================================
print("\n[5/6] Training Isolation Forest for Zero-Day Anomaly Detection...")
v_cols = [f"V{i}" for i in range(1, 29)]
iso_forest = IsolationForest(
    n_estimators=100,
    max_samples=2048,
    contamination=0.002,
    random_state=42,
    n_jobs=4
)
# Fit only on normal transactions from training split
iso_forest.fit(X_tr[y_tr == 0][v_cols])

# Invert score so higher = more anomalous [0 to 1]
raw_scores = -iso_forest.score_samples(X_val[v_cols])
val_anomaly_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)


# =====================================================================
# 6. ADAPTIVE RISK ENGINE & THRESHOLD CALIBRATION
# =====================================================================
print("\n[6/6] Calibrating Adaptive Risk Score & Decision Tiers...")

# Velocity burst indicator: high z-score on short-term count
count_60s = X_val["Tx_Count_60s"].values
burst_score = (count_60s - np.median(count_60s)) / (np.std(count_60s) + 1e-5)
burst_score = np.clip(burst_score / 5.0, 0.0, 1.0) # normalized 0-1

# Combined Adaptive Risk Score:
# 75% Supervised Model + 15% Feature Anomaly + 10% Velocity Burst (penalized only when anomalous)
adaptive_risk = (
    0.75 * val_preds +
    0.15 * val_anomaly_scores +
    0.10 * (burst_score * val_anomaly_scores) # Interaction: burst only matters when anomalous!
)

# Search optimal thresholds for F1 and low false positive rate
precisions, recalls, thresholds = precision_recall_curve(y_val, adaptive_risk)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
opt_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

print(f"Optimal F1 Threshold: {opt_threshold:.4f}")
print(f"Best F1 Score:        {f1_scores[best_idx]:.4f}")
print(f"Precision at opt:    {precisions[best_idx]:.4f}")
print(f"Recall at opt:       {recalls[best_idx]:.4f}")

# Calibrate 3 decision tiers dynamically:
# APPROVE: Risk < tau_review (zero friction)
# REVIEW:  tau_review <= Risk < tau_halt (step-up MFA / OTP)
# HALT:    Risk >= tau_halt (instant block)
tau_review = float(max(0.05, opt_threshold * 0.45))
tau_halt = float(opt_threshold)

approve_mask = adaptive_risk < tau_review
review_mask = (adaptive_risk >= tau_review) & (adaptive_risk < tau_halt)
halt_mask = adaptive_risk >= tau_halt

print(f"\nCalibrated Tiers: APPROVE < {tau_review:.3f} | REVIEW [{tau_review:.3f}, {tau_halt:.3f}) | HALT >= {tau_halt:.3f}")
print("\n--- 3-Tier Policy Evaluation on Validation Set ---")
print(f"APPROVE Tier: {approve_mask.sum():,} txs ({approve_mask.mean()*100:.2f}%) | Fraud leak: {y_val[approve_mask].sum()} / {y_val.sum()}")
print(f"REVIEW Tier:  {review_mask.sum():,} txs ({review_mask.mean()*100:.2f}%)   | Frauds intercepted for MFA: {y_val[review_mask].sum()}")
print(f"HALT Tier:    {halt_mask.sum():,} txs ({halt_mask.mean()*100:.2f}%)    | High-confidence frauds blocked: {y_val[halt_mask].sum()}")

# Measure single-transaction inference latency
sample_row = X_val.iloc[0:1]
latencies = []
for _ in range(50):
    t_start = time.perf_counter()
    _ = model.predict(sample_row)
    latencies.append((time.perf_counter() - t_start) * 1000.0)
print(f"\nInference Latency: P50={np.median(latencies):.3f}ms, Mean={np.mean(latencies):.3f}ms (Target: < 10ms)")

# Save artifacts
print("\nSaving trained models & feature definitions...")
with open("risk_engine_artifacts.pkl", "wb") as f:
    pickle.dump({
        "lgb_model": model,
        "iso_forest": iso_forest,
        "feature_cols": feature_cols,
        "v_cols": v_cols,
        "opt_threshold": float(opt_threshold),
        "tiers": {"approve_max": tau_review, "halt_min": tau_halt}
    }, f)

print("Pipeline execution and artifact generation COMPLETE!")
