"""
benchmark_latency.py - Stress & Latency Profiling for Adaptive Risk Engine
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import time
import pickle
import numpy as np
import pandas as pd

print("=" * 70)
print("LATENCY & HIGH-THROUGHPUT STRESS BENCHMARK")
print("=" * 70)

# 1. Load Artifacts
with open("risk_engine_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

lgb_model = artifacts["lgb_model"]
iso_forest = artifacts["iso_forest"]
feature_cols = artifacts["feature_cols"]
v_cols = artifacts["v_cols"]

# 2. Extract features on test transactions
from ml_pipeline import extract_features
test_df = pd.read_csv("test.csv").sort_values("Time").reset_index(drop=True)
test_feats = extract_features(test_df)
X_sample = test_feats[feature_cols].copy()
v_sample = test_feats[v_cols].values

n_trials = 1000
print(f"\nBenchmarking {n_trials} individual sequential transactions (Single-Item Real-Time Evaluation)...")

single_latencies = []
for i in range(n_trials):
    row_feat = X_sample.iloc[i:i+1]
    row_v = v_sample[i:i+1]
    
    t0 = time.perf_counter()
    p = lgb_model.predict(row_feat)
    iso = iso_forest.score_samples(row_v)
    risk = 0.75 * p[0] + 0.25 * (-iso[0])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    single_latencies.append(elapsed_ms)

single_latencies = np.array(single_latencies)

print("\n--- Single-Transaction Latency Distribution ---")
print(f"P50 (Median): {np.percentile(single_latencies, 50):.3f} ms")
print(f"P90:          {np.percentile(single_latencies, 90):.3f} ms")
print(f"P95:          {np.percentile(single_latencies, 95):.3f} ms")
print(f"P99:          {np.percentile(single_latencies, 99):.3f} ms")
print(f"Mean Latency: {np.mean(single_latencies):.3f} ms")
print(f"Max Latency:  {np.max(single_latencies):.3f} ms")

# 3. High-Volume Flash-Sale Batch Throughput Benchmark
print("\n" + "=" * 70)
print("HIGH-VOLUME FLASH SALE SURGE THROUGHPUT BENCHMARK")
print("=" * 70)

batch_sizes = [50, 200, 1000, 5000, 10000]
print(f"{'Batch Size':<12} | {'Total Time (ms)':<16} | {'Throughput (TPS)':<18} | {'Per-Tx Latency (ms)':<20}")
print("-" * 72)

for b in batch_sizes:
    batch_X = X_sample.iloc[:b]
    batch_v = v_sample[:b]
    
    t_start = time.perf_counter()
    preds = lgb_model.predict(batch_X)
    iso_scores = iso_forest.score_samples(batch_v)
    tot_time_ms = (time.perf_counter() - t_start) * 1000.0
    tps = b / (tot_time_ms / 1000.0)
    per_tx_ms = tot_time_ms / b
    
    print(f"{b:<12} | {tot_time_ms:<16.2f} | {tps:<18.1f} | {per_tx_ms:<20.4f}")

print("\nLatency & throughput benchmark complete!")
