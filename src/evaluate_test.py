"""
evaluate_test.py - Generates Out-of-Sample Predictions and Evaluates Performance on test.csv
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events

Run from project root:  python src/evaluate_test.py
"""

import os
import sys
import time
import types
import pickle
import numpy as np
import pandas as pd

# Resolve project root (one level above src/)
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

# ── IsolationForest shim (same class saved by ml_pipeline.py) ──────────────
class IsolationForest:
    """Lightweight numpy Z-score anomaly detector (matches sklearn API)."""
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
        X = np.asarray(X, dtype=np.float64)
        z = np.abs((X - self.mean_) / self.std_)
        return -z.mean(axis=1)

for _mod in ("__main__", "__mp_main__"):
    _shim = sys.modules.get(_mod)
    if _shim is None or not hasattr(_shim, "IsolationForest"):
        _shim = types.ModuleType(_mod)
        _shim.IsolationForest = IsolationForest
        sys.modules[_mod] = _shim

# Pure-NumPy metric replacements (avoids scipy dependency)
def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    idx = np.argsort(y_score)[::-1]
    yt = y_true[idx]
    npos = yt.sum(); nneg = len(yt) - npos
    tpr = np.concatenate([[0.0], np.cumsum(yt) / max(npos, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(1 - yt) / max(nneg, 1)])
    return float(np.trapezoid(tpr, fpr))

def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    idx = np.argsort(y_score)[::-1]
    yt = y_true[idx]
    npos = yt.sum()
    tps = np.cumsum(yt); fps = np.cumsum(1 - yt)
    prec = tps / (tps + fps)
    rec  = tps / max(npos, 1)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum(prec * (rec - rec_prev)))

print("=" * 70)
print("OUT-OF-SAMPLE TEST EVALUATION: test.csv")
print("=" * 70)

# 1. Load Artifacts
print("\n[1/5] Loading trained risk engine artifacts...")
_artifact_path = os.path.join(_ROOT_DIR, "models", "risk_engine_artifacts.pkl")
with open(_artifact_path, "rb") as f:
    artifacts = pickle.load(f)

lgb_model = artifacts["lgb_model"]
iso_forest = artifacts["iso_forest"]
feature_cols = artifacts["feature_cols"]
v_cols = artifacts["v_cols"]
opt_threshold = artifacts["opt_threshold"]
tiers = artifacts["tiers"]

print(f"Loaded LightGBM model with {len(feature_cols)} features.")
print(f"Calibrated Tiers: APPROVE < {tiers['approve_max']:.3f} | REVIEW [{tiers['approve_max']:.3f}, {tiers['halt_min']:.3f}) | HALT >= {tiers['halt_min']:.3f}")


# 2. Load test.csv and Chronologically Sort
print("\n[2/5] Loading and sorting test.csv...")
test_raw = pd.read_csv(os.path.join(_ROOT_DIR, "test.csv"))
print(f"test.csv shape: {test_raw.shape}")

# Preserve original order index to restore later
test_raw["original_order"] = np.arange(len(test_raw))
test_sorted = test_raw.sort_values("Time").reset_index(drop=True)


# 3. Vectorized Causal Feature Extraction
print("\n[3/5] Extracting causal features on test.csv...")
t0 = time.time()
times = test_sorted["Time"].values.astype(np.float64)
amounts = test_sorted["Amount"].values.astype(np.float64)
n = len(test_sorted)

hours = (times // 3600.0) % 24.0
test_sorted["Hour"] = hours
test_sorted["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
test_sorted["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)
test_sorted["Log_Amount"] = np.log1p(amounts)
test_sorted["Time_Since_Prev"] = np.concatenate([[0.0], np.diff(times)])

amount_cumsum = np.concatenate([[0.0], np.cumsum(amounts)])
indices = np.arange(n)

for w in [10, 60, 300, 900]:
    left_idx = np.searchsorted(times, times - w, side="left")
    count = indices - left_idx
    amt_sum = amount_cumsum[indices] - amount_cumsum[left_idx]
    amt_mean = np.divide(amt_sum, count, out=np.zeros_like(amt_sum), where=count > 0)
    
    test_sorted[f"Tx_Count_{w}s"] = count
    test_sorted[f"Amt_Sum_{w}s"] = amt_sum
    test_sorted[f"Amt_Mean_{w}s"] = amt_mean
    test_sorted[f"Amt_Ratio_{w}s"] = np.divide(amounts, amt_mean, out=np.ones_like(amounts), where=amt_mean > 0)

test_sorted["V14_V4_Ratio"] = test_sorted["V14"] / (np.abs(test_sorted["V4"]) + 1e-5)
test_sorted["V12_V10_Diff"] = test_sorted["V12"] - test_sorted["V10"]

print(f"Features computed in {time.time() - t0:.2f}s.")


# 4. Multi-Signal Inference & Risk Scoring
print("\n[4/5] Executing inference on test.csv...")
X_test = test_sorted[feature_cols]

# 4.1 Supervised probability
ml_probs = lgb_model.predict(X_test)

# 4.2 Unsupervised anomaly score
v_matrix = test_sorted[v_cols].values
raw_iso = -iso_forest.score_samples(v_matrix)
anomaly_scores = 1.0 / (1.0 + np.exp(-((raw_iso - 0.5) * 8.0)))

# 4.3 Velocity surge index
count_60s = test_sorted["Tx_Count_60s"].values
burst_index = np.clip((count_60s - np.median(count_60s)) / (np.std(count_60s) + 1e-5) / 5.0, 0.0, 1.0)

# 4.4 Composite Adaptive Risk
adaptive_risk = 0.75 * ml_probs + 0.15 * anomaly_scores + 0.10 * (burst_index * anomaly_scores)
adaptive_risk = np.clip(adaptive_risk, 0.0, 1.0)

# 4.5 Tier Decisions
decisions = np.where(
    adaptive_risk >= tiers["halt_min"], "HALT",
    np.where(adaptive_risk >= tiers["approve_max"], "REVIEW", "APPROVE")
)

test_sorted["ML_Fraud_Prob"] = np.round(ml_probs, 4)
test_sorted["Anomaly_Score"] = np.round(anomaly_scores, 4)
test_sorted["Velocity_Surge_Index"] = np.round(burst_index, 4)
test_sorted["Adaptive_Risk_Score"] = np.round(adaptive_risk, 4)
test_sorted["Decision"] = decisions


# 5. Evaluation & Output Export
print("\n[5/5] Performance metrics and exporting test_predictions.csv...")

if "Class" in test_sorted.columns:
    y_true = test_sorted["Class"].values
    num_frauds = y_true.sum()
    print(f"\nGround Truth Labels Available in test.csv: {num_frauds} frauds out of {len(test_sorted)} ({y_true.mean()*100:.3f}%)")
    
    test_roc_auc = roc_auc_score(y_true, adaptive_risk)
    test_pr_auc = average_precision_score(y_true, adaptive_risk)
    
    print("\n--- Test Set Metrics ---")
    print(f"ROC-AUC: {test_roc_auc:.4f}")
    print(f"PR-AUC:  {test_pr_auc:.4f}")
    
    # Tier distribution and fraud interception breakdown
    approve_mask = decisions == "APPROVE"
    review_mask = decisions == "REVIEW"
    halt_mask = decisions == "HALT"
    
    print("\n--- Policy Performance on test.csv ---")
    print(f"APPROVE: {approve_mask.sum():,} txs ({approve_mask.mean()*100:.2f}%) | Fraud leak: {y_true[approve_mask].sum()} / {num_frauds}")
    print(f"REVIEW:  {review_mask.sum():,} txs ({review_mask.mean()*100:.2f}%)   | Intercepted for MFA: {y_true[review_mask].sum()} / {num_frauds}")
    print(f"HALT:    {halt_mask.sum():,} txs ({halt_mask.mean()*100:.2f}%)    | Blocked immediately: {y_true[halt_mask].sum()} / {num_frauds}")
    
    interception_rate = (y_true[review_mask].sum() + y_true[halt_mask].sum()) / max(1, num_frauds)
    print(f"\nTotal Fraud Defense Interception (HALT + REVIEW): {interception_rate*100:.2f}%")
    print(f"Friction-Free Consumer Clearance Rate: {approve_mask.mean()*100:.2f}%")

# Restore original row order
out_df = test_sorted.sort_values("original_order").reset_index(drop=True)
export_cols = [
    "Time", "Amount", "ML_Fraud_Prob", "Anomaly_Score",
    "Velocity_Surge_Index", "Adaptive_Risk_Score", "Decision"
]
if "Class" in out_df.columns:
    export_cols.append("Class")

out_df[export_cols].to_csv(os.path.join(_ROOT_DIR, "test_predictions.csv"), index=False)
print(f"\nExported {len(out_df)} predictions to test_predictions.csv successfully!")
print(f"\nEvaluation Complete: " + time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 70)
print("END OF EVALUATION")
print("=" * 70)
print("Thank you for using Securing the Surge - Real-Time Transaction Fraud Detection System")