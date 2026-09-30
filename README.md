# SurgeGuard — Securing the Surge: Real-Time Fraud Detection

> **Problem Statement:** Protecting digital transactions during peak e-commerce events by distinguishing legitimate flash-sale volume surges from coordinated bot attacks in under 1 millisecond.

---

## Submission Checklist

- [x] `src/` — Complete source code (training pipeline, inference API, dashboard)
- [x] `models/` — Serialized trained model artifact (`risk_engine_artifacts.pkl`)
- [x] `predict.py` — Standalone inference script (`--input` / `--output` CLI)
- [x] `requirements.txt` — Exact pinned dependency versions
- [x] `README.md` — Step-by-step setup and execution instructions

---

## Quick Start: Running `predict.py`

### Step 1 — Prerequisites

Ensure you have **Python 3.10+** installed.

```bash
python --version
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** All versions are pinned for reproducibility. A virtual environment is recommended:
> ```bash
> python -m venv venv
> # Windows:
> venv\Scripts\activate
> # macOS / Linux:
> source venv/bin/activate
> pip install -r requirements.txt
> ```

### Step 3 — Run Inference

```bash
python predict.py --input test_cleaned.csv --output predictions.csv
```

**Arguments:**

| Argument | Short | Description | Example |
|----------|-------|-------------|---------|
| `--input` | `-i` | Path to input test CSV file | `test_cleaned.csv` |
| `--output` | `-o` | Path to save prediction CSV | `predictions.csv` |

**Minimum required columns in input CSV:**

| Column | Type | Description |
|--------|------|-------------|
| `Time` | float | Seconds elapsed from first transaction |
| `Amount` | float | Transaction amount |
| `V1`–`V28` | float | PCA-anonymised card features (optional, default 0.0) |

### Step 4 — Inspect Output

The output CSV contains one row per input transaction, in the **same order** as the input:

| Column | Type | Description |
|--------|------|-------------|
| `prediction` | `0` / `1` | Binary class — `1` = Fraud, `0` = Legitimate |
| `fraud_probability` | `[0.0, 1.0]` | Calibrated LightGBM fraud probability |
| `risk_score` | `[0.0, 1.0]` | Tri-signal adaptive composite risk score |
| `decision` | string | Policy decision: `APPROVE`, `REVIEW`, or `HALT` |

**Example output:**

```
prediction,fraud_probability,risk_score,decision
0,0.001823,0.003471,APPROVE
0,0.002109,0.004218,APPROVE
1,0.998740,0.892310,HALT
0,0.003512,0.005821,APPROVE
```

---

## Verify the Setup

Run the automated verification test suite to confirm everything works end-to-end:

```bash
python src/test.py
```

Expected output:
```
====================================================================== 
SURGEGUARD VERIFICATION TEST SUITE (src/test.py)
======================================================================
[1/4] Verifying model artifact in models/risk_engine_artifacts.pkl...
[PASS] Model loaded successfully: 53 features.
[2/4] Testing causal feature extraction on synthetic batch...
[PASS] Feature pipeline verified. Exact 53 features extracted without future leakage.
[3/4] Testing single-item and batch model predictions...
[PASS] Inference predictions valid: [...]
[4/4] Verifying standalone predict.py CLI interface...
[PASS] predict.py successfully executed and verified on 42,691 rows.
======================================================================
ALL TESTS PASSED SUCCESSFULLY!
======================================================================
```

---

## Optional: Launch the Real-Time Web Dashboard

To start the full adaptive fraud operations console (requires all dependencies):

```bash
python src/run_server.py
```

Then open in your browser:
- **Live Authorization Feed:** [http://localhost:8000](http://localhost:8000)
- **Fraud Operations & Admin Console:** [http://localhost:8000/admin](http://localhost:8000/admin)

---

## Project Structure

```
SurgeGuard/
├── predict.py                    # Standalone inference script (judges run this)
├── requirements.txt              # Pinned dependencies for full reproducibility
├── README.md                     # This file
├── .env.example                  # Environment variable template (copy to .env)
│
├── models/
│   └── risk_engine_artifacts.pkl # Trained LightGBM + IsolationForest + calibrated tiers
│
└── src/
    ├── ml_pipeline.py            # Feature engineering, training & tier calibration
    ├── evaluate_test.py          # Out-of-sample evaluation on test split
    ├── api.py                    # FastAPI real-time scoring & admin endpoints
    ├── feedback_store.py         # SQLite persistent human-in-the-loop feedback store
    ├── retrain_engine.py         # Champion/Challenger adaptive retraining cycle
    ├── benchmark_latency.py      # Latency & throughput stress profiler
    ├── run_server.py             # Uvicorn server launcher
    ├── test.py                   # Automated verification test suite
    ├── test_adaptive_loop.py     # Adaptive retraining end-to-end integration test
    └── dashboard/
        ├── index.html            # Real-time transaction authorization feed UI
        ├── style.css             # Dark design system & shared component styles
        ├── app.js                # Live stream controller & scenario simulator
        ├── admin.html            # Fraud Operations & Model Center console
        ├── admin-style.css       # Admin drawer & ledger styles
        └── admin.js              # Review queue, human feedback & retraining controller
```

---

## Solution Architecture

### Tri-Signal Adaptive Risk Engine

```
Transaction Input
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Signal 1 (75%): LightGBM Supervised Classifier    │
│   • 53 causal features (zero future leakage)        │
│   • Temporal cyclic encoding (hour sin/cos)         │
│   • Prefix-sum rolling windows: 10s / 60s / 300s    │
│   • V-feature interaction ratios                    │
├─────────────────────────────────────────────────────┤
│  Signal 2 (15%): IsolationForest Anomaly Score     │
│   • Trained on legitimate-only transactions         │
│   • Detects zero-day fraud patterns                 │
├─────────────────────────────────────────────────────┤
│  Signal 3 (10%): Real-Time Velocity Surge Index    │
│   • In-memory sliding window buffer (900s)          │
│   • Penalises burst velocity ONLY when anomalous    │
│   • Legitimate flash-sales pass through             │
└─────────────────────────────────────────────────────┘
       │
       ▼
  Adaptive Risk Score = 0.75·ML + 0.15·Anomaly + 0.10·(Velocity × Anomaly)
       │
       ▼
┌──────────────┬────────────────┬──────────────┐
│   APPROVE    │    REVIEW      │    HALT      │
│ Risk < 0.35  │ 0.35–0.77      │ Risk ≥ 0.77  │
│ Zero friction│ Step-up MFA   │ Instant block│
└──────────────┴────────────────┴──────────────┘
```

### Validated Performance (39,755-transaction holdout set)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9667** |
| PR-AUC | **0.6295** |
| Best F1 Score | **0.5385** |
| Inference Latency (P50) | **< 1 ms** |
| Inference Latency (P99) | **< 5 ms** |
| APPROVE tier | 98.64% (legitimate traffic) |
| REVIEW tier (MFA) | 1.22% (step-up challenge) |
| HALT tier | 0.14% (high-confidence block) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'lightgbm'`**
```bash
pip install -r requirements.txt
```

**`FileNotFoundError: models/risk_engine_artifacts.pkl`**

The model artifact must be in `models/`. If missing, retrain:
```bash
python src/ml_pipeline.py
```

**`KeyError: 'Time'` or `'Amount'`**

Your input CSV must contain at minimum `Time` and `Amount` columns. V1–V28 are optional and default to `0.0`.

**Slow first run**

The first inference call warms up the LightGBM booster. Subsequent calls run at sub-millisecond latency.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 2.5.3 | Vectorised numerical computation |
| `pandas` | 3.0.6 | CSV I/O and DataFrame operations |
| `lightgbm` | 4.7.0 | Gradient boosted fraud classifier |
| `fastapi` | 0.141.1 | Real-time REST inference API |
| `uvicorn` | 0.54.0 | ASGI server runtime |
| `pydantic` | 2.13.5 | API request/response validation |
| `python-dotenv` | 1.2.3 | Environment variable management |

> `predict.py` only requires `numpy`, `pandas`, and `lightgbm`. The remaining packages are needed for the optional real-time API server.
