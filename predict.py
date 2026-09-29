"""
predict.py - Standalone Inference Script for SurgeGuard Fraud Detection
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events

Usage:
    python predict.py --input path/to/test.csv --output path/to/predictions.csv

Arguments:
    --input   Path to the input CSV file containing transaction data.
    --output  Path where the output prediction CSV will be saved.

Output CSV columns:
    - fraud_probability   : Model predicted probability of fraud (0.0 to 1.0)
    - predicted_class     : Binary prediction (1 = Fraud, 0 = Legitimate)
"""

import sys
import os
import argparse
import pickle
import types
import numpy as np
import pandas as pd


# =====================================================================
# ISOLATION FOREST - redefined here so pickle can deserialize the
# artifact saved by ml_pipeline.py without __main__ namespace conflicts.
# =====================================================================
class IsolationForest:
    """Lightweight numpy Z-score anomaly detector (matches sklearn API)."""

    def __init__(self, n_estimators=100, max_samples=2048,
                 contamination=0.002, random_state=42, n_jobs=4):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        return self

    def score_samples(self, X):
        """Returns negative anomaly score - higher = more normal (sklearn convention)."""
        X = np.asarray(X, dtype=np.float64)
        z = np.abs((X - self.mean_) / self.std_)
        return -z.mean(axis=1)


# =====================================================================
# FEATURE ENGINEERING - identical causal logic used during training
# =====================================================================
def extract_features(df):
    """
    Computes causal temporal and velocity features strictly from past information.
    Uses vectorized NumPy searchsorted + prefix sums for zero-leakage computation.
    """
    df = df.copy()
    times = df["Time"].values.astype(np.float64)
    amounts = df["Amount"].values.astype(np.float64)
    n = len(df)

    hours = (times // 3600.0) % 24.0
    df["Hour"] = hours
    df["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    df["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    df["Log_Amount"] = np.log1p(amounts)
    df["Time_Since_Prev"] = np.concatenate([[0.0], np.diff(times)])

    amount_cumsum = np.concatenate([[0.0], np.cumsum(amounts)])
    indices = np.arange(n)

    for w in [10, 60, 300, 900]:
        left_idx = np.searchsorted(times, times - w, side="left")
        count = indices - left_idx
        amt_sum = amount_cumsum[indices] - amount_cumsum[left_idx]
        amt_mean = np.divide(amt_sum, count, out=np.zeros_like(amt_sum), where=count > 0)
        df[f"Tx_Count_{w}s"] = count
        df[f"Amt_Sum_{w}s"] = amt_sum
        df[f"Amt_Mean_{w}s"] = amt_mean
        df[f"Amt_Ratio_{w}s"] = np.divide(
            amounts, amt_mean, out=np.ones_like(amounts), where=amt_mean > 0
        )

    df["V14_V4_Ratio"] = df["V14"] / (np.abs(df["V4"]) + 1e-5)
    df["V12_V10_Diff"] = df["V12"] - df["V10"]
    return df


# =====================================================================
# MAIN INFERENCE PIPELINE
# =====================================================================
def load_artifacts(models_dir):
    """Load serialized model artifacts from the models/ directory."""
    artifact_path = os.path.join(models_dir, "risk_engine_artifacts.pkl")
    if not os.path.exists(artifact_path):
        sys.exit(
            f"[ERROR] Model artifact not found at '{artifact_path}'.\n"
            "Ensure 'models/risk_engine_artifacts.pkl' exists.\n"
            "Run 'python src/ml_pipeline.py' to generate it."
        )
    for mod_name in ("__main__", "__mp_main__"):
        shim = sys.modules.get(mod_name)
        if shim is None or not hasattr(shim, "IsolationForest"):
            shim = types.ModuleType(mod_name)
            shim.IsolationForest = IsolationForest
            sys.modules[mod_name] = shim
    with open(artifact_path, "rb") as f:
        return pickle.load(f)


def predict(input_path, output_path, models_dir="models"):
    """
    Load saved model, run inference on input CSV, and write predictions.

    Parameters
    ----------
    input_path  : str  Path to input CSV (columns: Time, Amount, V1-V28)
    output_path : str  Path to write the output prediction CSV
    models_dir  : str  Directory containing risk_engine_artifacts.pkl
    """
    print(f"[1/4] Loading model artifacts from '{models_dir}'...")
    artifacts = load_artifacts(models_dir)
    lgb_model    = artifacts["lgb_model"]
    iso_forest   = artifacts["iso_forest"]
    feature_cols = artifacts["feature_cols"]
    v_cols       = artifacts["v_cols"]
    tiers        = artifacts["tiers"]
    print(f"      Model loaded. Features: {len(feature_cols)}")

    print(f"[2/4] Loading input data from '{input_path}'...")
    if not os.path.exists(input_path):
        sys.exit(f"[ERROR] Input file not found: '{input_path}'")
    df = pd.read_csv(input_path)
    print(f"      Loaded {len(df):,} transactions with {df.shape[1]} columns.")

    required_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] Input CSV is missing required columns: {missing}")

    df["_original_order"] = np.arange(len(df))
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    print("[3/4] Engineering features and running inference...")
    df_feats = extract_features(df_sorted)
    X = df_feats[feature_cols]

    ml_probs = lgb_model.predict(X, num_iteration=lgb_model.best_iteration)

    raw_iso = -iso_forest.score_samples(df_feats[v_cols].values)
    anomaly_scores = 1.0 / (1.0 + np.exp(-((raw_iso - 0.5) * 8.0)))

    count_60s = df_feats["Tx_Count_60s"].values
    burst_index = np.clip(
        (count_60s - np.median(count_60s)) / (np.std(count_60s) + 1e-5) / 5.0,
        0.0, 1.0
    )

    adaptive_risk = np.clip(
        0.75 * ml_probs + 0.15 * anomaly_scores + 0.10 * (burst_index * anomaly_scores),
        0.0, 1.0
    )

    predicted_class = (adaptive_risk >= tiers["halt_min"]).astype(int)

    df_sorted["fraud_probability"] = np.round(ml_probs, 4)
    df_sorted["predicted_class"]   = predicted_class

    print(f"[4/4] Saving predictions to '{output_path}'...")
    out_df = df_sorted.sort_values("_original_order").reset_index(drop=True)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_df[["fraud_probability", "predicted_class"]].to_csv(output_path, index=False)

    fraud_count = predicted_class.sum()
    print(f"\nDone. {len(out_df):,} predictions written to '{output_path}'.")
    print(f"Predicted fraud: {fraud_count} / {len(out_df)} ({fraud_count / len(out_df) * 100:.3f}%)")


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "SurgeGuard Inference Script - Fraud Detection\n"
            "Loads the trained model and predicts fraud on new transaction data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="INPUT_CSV",
        help="Path to the input CSV file (must contain: Time, Amount, V1-V28).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="OUTPUT_CSV",
        help="Path where the prediction CSV will be saved.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        metavar="MODELS_DIR",
        help="Directory containing risk_engine_artifacts.pkl. Default: models",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(
        input_path=args.input,
        output_path=args.output,
        models_dir=args.models_dir,
    )
    