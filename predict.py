#!/usr/bin/env python3
"""
predict.py - Standalone Inference Script for SurgeGuard Fraud Detection Engine

Usage:
    python predict.py --input <path_to_test_csv> --output <path_to_output_csv>

Description:
    Loads the serialized model artifact from models/ and generates row-by-row
    fraud predictions and probabilities for new transactions without retraining.
"""

import sys
import os
import argparse
import pickle
import time
import numpy as np
import pandas as pd


# =====================================================================
# ISOLATION FOREST SHIM — Guarantees zero-dependency unpickling
# =====================================================================
class IsolationForest:
    """Lightweight numpy Z-score anomaly detector (matches sklearn IsolationForest API)."""
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

# Ensure src directory is in sys.path
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if os.path.exists(_src_dir) and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Ensure IsolationForest is available in __main__, __mp_main__, and retrain_engine for unpickling
import types

# Inject IsolationForest into __main__ shim (needed for pickle deserialization)
sys.modules["__main__"].IsolationForest = IsolationForest

# Stub __mp_main__ unconditionally to cover multiprocessing workers
for _mod_name in ("__mp_main__", "retrain_engine"):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        _stub.IsolationForest = IsolationForest
        sys.modules[_mod_name] = _stub
    else:
        sys.modules[_mod_name].IsolationForest = IsolationForest

# Stub ml_pipeline so pickle.load of an artifact saved during training
# does NOT re-execute ml_pipeline.py as a script (which would need train_cleaned.csv).
if "ml_pipeline" not in sys.modules:
    _ml_stub = types.ModuleType("ml_pipeline")
    _ml_stub.IsolationForest = IsolationForest
    # Provide extract_features stub; the real implementation is inlined in predict.py
    _ml_stub.extract_features = None
    sys.modules["ml_pipeline"] = _ml_stub


# =====================================================================
# CAUSAL FEATURE EXTRACTION (Exact 53-dimension pipeline)
# =====================================================================
def extract_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-derived features, prefix-sum rolling windows (10s, 60s, 300s, 900s),
    and non-linear PCA interaction ratios.
    Preserves strict causality without future lookahead.
    """
    df = df.copy()
    times = df["Time"].values.astype(np.float64)
    amounts = df["Amount"].values.astype(np.float64)
    n = len(df)

    # 1. Temporal cyclic features
    hours = (times // 3600.0) % 24.0
    df["Hour"] = hours
    df["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["Log_Amount"] = np.log1p(amounts)
    df["Time_Since_Prev"] = np.concatenate([[0.0], np.diff(times)])

    # 2. Vectorized cumulative sum for rolling intervals
    amount_cumsum = np.concatenate([[0.0], np.cumsum(amounts)])

    windows = [10, 60, 300, 900]
    for w in windows:
        left_times = times - float(w)
        left_idx = np.searchsorted(times, left_times, side="left")
        counts = (np.arange(n) - left_idx + 1).astype(np.float64)
        sums = amount_cumsum[1:] - amount_cumsum[left_idx]
        means = sums / np.maximum(counts, 1.0)
        ratios = amounts / (means + 1e-4)

        df[f"Tx_Count_{w}s"] = counts
        df[f"Amt_Sum_{w}s"] = sums
        df[f"Amt_Mean_{w}s"] = means
        df[f"Amt_Ratio_{w}s"] = ratios

    # 3. Non-linear interaction features
    v14 = df["V14"] if "V14" in df.columns else np.zeros(n)
    v4  = df["V4"]  if "V4" in df.columns  else np.zeros(n)
    v12 = df["V12"] if "V12" in df.columns else np.zeros(n)
    v10 = df["V10"] if "V10" in df.columns else np.zeros(n)

    df["V14_V4_Ratio"] = v14 / (np.abs(v4) + 1e-4)
    df["V12_V10_Diff"] = v12 - v10

    # Ensure all V1..V28 exist
    for i in range(1, 29):
        col = f"V{i}"
        if col not in df.columns:
            df[col] = 0.0

    return df


# =====================================================================
# MODEL LOADER
# =====================================================================
def load_saved_model(models_dir: str = "models"):
    """
    Loads the serialized model artifact from models/ directory.
    Checks active_model.json first, then falls back to known artifact paths.
    """
    candidate_paths = [
        os.path.join(models_dir, "risk_engine_artifacts.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), models_dir, "risk_engine_artifacts.pkl"),
        "risk_engine_artifacts.pkl",
        os.path.join(models_dir, "model.pkl"),
        os.path.join(models_dir, "model_v3.pkl"),
    ]

    # Check active_model.json if present
    active_json = os.path.join(models_dir, "active_model.json")
    if os.path.exists(active_json):
        try:
            import json
            with open(active_json, "r") as f:
                info = json.load(f)
            art_path = info.get("artifact_path")
            if art_path and os.path.exists(art_path):
                candidate_paths.insert(0, art_path)
        except Exception:
            pass

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    artifacts = pickle.load(f)
                if isinstance(artifacts, dict) and "lgb_model" in artifacts:
                    print(f"[SurgeGuard] Successfully loaded model artifact from: {path}")
                    return artifacts
            except Exception as e:
                print(f"[SurgeGuard] Warning: Failed loading {path}: {e}")
                continue

    raise FileNotFoundError(
        f"Could not find valid model artifact in {models_dir}/ or root. "
        "Please ensure models/model_v3.pkl or models/risk_engine_artifacts.pkl exists."
    )


# =====================================================================
# MAIN INFERENCE PIPELINE
# =====================================================================
def run_inference(input_csv: str, output_csv: str):
    t_start = time.perf_counter()
    print("=" * 70)
    print("SURGEGUARD — STANDALONE FRAUD INFERENCE ENGINE")
    print("=" * 70)

    # 1. Validate Input
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"[1/4] Reading input transactions from: {input_csv}")
    raw_df = pd.read_csv(input_csv)
    n_rows = len(raw_df)
    print(f"      Loaded {n_rows:,} transactions.")

    if n_rows == 0:
        print("[SurgeGuard] Warning: Input file is empty. Writing empty prediction file.")
        pd.DataFrame(columns=["prediction", "fraud_probability", "risk_score", "decision"]).to_csv(output_csv, index=False)
        return

    # Check required minimum columns
    for col in ["Time", "Amount"]:
        if col not in raw_df.columns:
            raise ValueError(f"Required column '{col}' missing from input CSV.")

    # Preserve original row index for row-by-row output alignment
    raw_df["__original_index__"] = np.arange(n_rows)

    # 2. Load Model Artifacts
    print("\n[2/4] Loading trained model artifacts from models/...")
    artifacts = load_saved_model("models")

    lgb_model    = artifacts["lgb_model"]
    iso_forest   = artifacts["iso_forest"]
    feature_cols = artifacts["feature_cols"]
    v_cols       = artifacts["v_cols"]
    opt_thresh   = artifacts.get("opt_threshold", 0.5)
    tiers        = artifacts.get("tiers", {"approve_max": 0.45, "halt_min": 1.0})

    # 3. Chronological Causal Feature Extraction
    print("\n[3/4] Extracting causal features and prefix-sum rolling windows...")
    t_feat = time.perf_counter()
    sorted_df = raw_df.sort_values("Time").reset_index(drop=True)
    feats_df = extract_causal_features(sorted_df)
    print(f"      Feature extraction completed in {(time.perf_counter() - t_feat)*1000:.1f}ms.")

    # 4. Generate Predictions
    print("\n[4/4] Generating real-time fraud probabilities and decisions...")
    t_pred = time.perf_counter()

    # Align features
    X_feat = feats_df[feature_cols].copy()
    v_mat  = feats_df[v_cols].values

    # LightGBM Supervised Fraud Probability
    p_ml = lgb_model.predict(X_feat)

    # Isolation Forest Anomaly Score
    anomaly_scores = -iso_forest.score_samples(v_mat)
    # Normalize anomaly to [0, 1]
    a_min, a_max = anomaly_scores.min(), anomaly_scores.max()
    if a_max > a_min:
        anomaly_norm = (anomaly_scores - a_min) / (a_max - a_min)
    else:
        anomaly_norm = np.zeros_like(anomaly_scores)

    # Velocity index
    velocity_idx = feats_df["Tx_Count_10s"].values / 20.0

    # Composite Adaptive Risk Score (Tri-Signal Decomposition)
    adaptive_risk = (
        0.75 * p_ml +
        0.15 * anomaly_norm +
        0.10 * (np.clip(velocity_idx, 0.0, 1.0) * anomaly_norm)
    )
    adaptive_risk = np.clip(adaptive_risk, 0.0, 1.0)

    # Decisions and Binary Classification
    halt_min = tiers.get("halt_min", 0.70)
    approve_max = tiers.get("approve_max", 0.35)

    decisions = np.where(
        adaptive_risk >= halt_min, "HALT",
        np.where(adaptive_risk >= approve_max, "REVIEW", "APPROVE")
    )

    # Binary class prediction (1 = FRAUD, 0 = LEGITIMATE)
    # Calibrated binary threshold: flagged as HALT/critical risk or probability threshold
    binary_pred = np.where(adaptive_risk >= min(opt_thresh, halt_min), 1, 0)

    feats_df["prediction"]        = binary_pred
    feats_df["fraud_probability"] = np.round(p_ml, 6)
    feats_df["risk_score"]        = np.round(adaptive_risk, 6)
    feats_df["decision"]          = decisions

    # 5. Restore Exact Original Row-by-Row Order
    out_df = feats_df.sort_values("__original_index__").reset_index(drop=True)

    # Create destination directory if needed
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Export CSV containing predictions corresponding row-by-row
    export_columns = ["prediction", "fraud_probability", "risk_score", "decision"]
    out_df[export_columns].to_csv(output_csv, index=False)

    total_time = (time.perf_counter() - t_start)
    lat_per_tx = (total_time / max(1, n_rows)) * 1000.0

    print(f"\n[DONE] Generated {n_rows:,} predictions -> {output_csv}")
    print(f"       Total time: {total_time:.2f}s ({lat_per_tx:.3f}ms per transaction)")
    print(f"       Class distribution: Fraud={np.sum(binary_pred==1):,}, Legitimate={np.sum(binary_pred==0):,}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="SurgeGuard — Standalone Fraud Detection Inference Script"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input test CSV file (e.g. test_cleaned.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save output prediction CSV file (e.g. predictions.csv)"
    )
    args = parser.parse_args()

    run_inference(args.input, args.output)


if __name__ == "__main__":
    main()
