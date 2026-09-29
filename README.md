# 🛡️ SurgeGuard — Securing the Surge

> **Protecting Digital Transactions During Peak E-Commerce Events**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

During flash sales and holiday surges, transaction volume spikes **10×**. Legacy banking security systems panic: static thresholds block genuine shoppers while loosening thresholds allows distributed bot attacks to drain customer accounts.

**SurgeGuard** proves that *High Transaction Volume ≠ Fraud*. Using a **Tri-Signal Adaptive Risk Engine**, it differentiates legitimate flash-sale surges from coordinated bot attacks in **under 7 milliseconds** — ensuring **96.6% friction-free approvals** for genuine buyers while halting **78.1% of fraudulent account drains**.

---

## 🏗️ Architecture

```
Transaction Stream
    │
    ▼
[Data Hygiene Layer]
    │
    ▼
[Causal Velocity Engine]  ← Stateful in-memory rolling window (10s / 60s / 300s / 900s)
    │
    ├──▶ [LightGBM Classifier]        (75% weight — supervised fraud probability)
    ├──▶ [Isolation Forest Detector]  (15% weight — zero-day anomaly score)
    └──▶ [Surge Burst Multiplier]     (10% weight — velocity · anomaly interaction)
            │
            ▼
    Composite Adaptive Risk Score
    = 0.75·P(ML) + 0.15·S(Anomaly) + 0.10·(Burst × Anomaly)
            │
            ▼
    [Modular Context Adapters]
    AccountContext | MerchantContext | GeoContext
            │
    ┌───────┼────────┐
    ▼       ▼        ▼
 APPROVE  REVIEW   HALT
(<0.359) [0.359,  (≥0.798)
          0.798)
```

### Key Design Decisions

| Principle | Implementation |
|:---|:---|
| **Zero future leakage** | Causal features computed with `searchsorted` prefix sums (no lookahead) |
| **Surge ≠ Fraud** | Velocity burst only escalates risk when combined with anomaly score |
| **Missing context ≠ Fraud** | Pluggable adapters gracefully degrade when account/geo data is absent |
| **Sub-10ms latency** | In-memory `deque` sliding buffer; LightGBM inference at 0.005 ms/tx |

---

## 📊 Performance Benchmarks

### Model Accuracy

| Metric | Score |
|:---|:---|
| Test ROC-AUC | **0.9667** |
| Test PR-AUC | **0.4560** |
| Validation PR-AUC | **0.6295** (time-aware split, no future leakage) |

### Business & Operational Policy (out-of-sample)

| Metric | Result | Value |
|:---|:---|:---|
| Friction-Free Clearance | **96.57%** (41,255 txs) | High conversion during flash sales |
| Fraud Defense Interception | **78.08%** (57/73 frauds blocked) | Bot drain attacks quarantined |
| Step-Up Review Rate | **2.63%** (1,125 txs) | Stepped-up MFA / OTP challenge |
| Causal Feature Extraction | **0.12 sec** for 200K txs | Vectorized prefix sum algorithm |

### Latency & Throughput

| Metric | Value | Target |
|:---|:---|:---|
| Median Latency (P50) | **6.91 ms** | < 10 ms ✅ |
| P90 Latency | **8.52 ms** | < 10 ms ✅ |
| Surge Batch Throughput | **183,727 TPS** | — |

---

## 📁 Repository Structure

```
SurgeGuard/
├── ml_pipeline.py              # Feature engineering + model training (run this first)
├── retrain_engine.py           # Adaptive retraining, validation & champion/challenger promotion
├── feedback_store.py           # Persistent SQLite storage for Human-in-the-Loop verified labels
├── api.py                      # FastAPI real-time scoring engine + adaptive ML endpoints
├── run_server.py               # Server entry point (uvicorn launcher)
├── test_adaptive_loop.py       # E2E test suite for feedback, retraining, and promotion
├── evaluate_test.py            # Out-of-sample evaluation → test_predictions.csv
├── benchmark_latency.py        # Latency stress profiler
├── models/                     # Versioned LightGBM artifacts & active model metadata
│   ├── model_v1.pkl            # Base champion model artifact
│   ├── model_v2.pkl            # Retrained challenger artifact
│   └── active_model.json       # Active champion pointer, metrics & audit history
├── feedback.db                 # Persistent SQLite database for human-verified feedback
├── risk_engine_artifacts.pkl   # Serialized model weights + calibrated tiers (auto-generated)
├── test_predictions.csv        # 42,720-row scored prediction output (auto-generated)
├── .env.example                # Environment variable template
├── .env                        # Your local config (gitignored)
├── dashboard/
│   ├── index.html              # Live transaction dashboard UI
│   ├── style.css               # Dashboard styles (glassmorphism dark theme)
│   ├── app.js                  # Dashboard simulation + live metrics controller
│   ├── admin.html              # Admin Override & Adaptive Learning Console
│   ├── admin.js                # Human feedback, model status & retraining controller
│   └── admin-style.css         # Admin panel & adaptive learning styles
├── train_cleaned.csv           # Cleaned training data (198,773 rows)
└── test_cleaned.csv            # Cleaned test data (42,691 rows)
```

---

## ⚙️ Prerequisites

- **Python 3.10+** (tested on Python 3.14)
- **pip** (comes with Python)
- The dataset files: `train_cleaned.csv` and `test_cleaned.csv`

---

## 🚀 Getting Started

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/surgeguard.git
cd surgeguard
```

### Step 2 — Install dependencies

```bash
pip install fastapi uvicorn lightgbm numpy pandas python-dotenv
```

> **Note:** If pip warns that scripts are not on PATH, you can still run everything via `python run_server.py`. No PATH changes are needed.

### Step 3 — Configure environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` if needed (defaults work out-of-the-box):

```env
HOST=0.0.0.0
PORT=8000
WORKERS=4
ALLOWED_ORIGINS=*
ARTIFACTS_PATH=risk_engine_artifacts.pkl
```

---

## 🧠 Step 4 — Train the ML Pipeline

> **Skip this step if `risk_engine_artifacts.pkl` already exists in the project directory.**

```bash
python ml_pipeline.py
```

**What it does:**
1. Loads `train_cleaned.csv` and `test_cleaned.csv`
2. Engineers 50+ causal temporal & velocity features (zero future leakage)
3. Trains a class-weighted **LightGBM** gradient boosting classifier
4. Fits an **Isolation Forest** anomaly detector on normal transactions only
5. Calibrates 3-tier decision thresholds (APPROVE / REVIEW / HALT)
6. Saves `risk_engine_artifacts.pkl`

Expected output (abridged):
```
[1/6] Loading data and auditing hygiene...
[2/6] Engineering leakage-free causal features...
[3/6] Splitting train into chronological 80% Train / 20% Val...
[4/6] Training Supervised LightGBM Classifier...
[5/6] Training Isolation Forest for Zero-Day Anomaly Detection...
[6/6] Calibrating Adaptive Risk Score & Decision Tiers...
Pipeline execution and artifact generation COMPLETE!
```

---

## 🖥️ Step 5 — Start the Server

```bash
python run_server.py
```

The server starts on **`http://0.0.0.0:8000`** with 4 worker processes.

```
Starting Adaptive Risk Engine Server on http://0.0.0.0:8000...
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

> Press `Ctrl+C` to stop the server.

---

## 🌐 Step 6 — Open the Dashboard

With the server running, open your browser:

| Page | URL | Description |
|:---|:---|:---|
| **Live Dashboard** | http://localhost:8000 | Real-time transaction feed with scenario simulator |
| **Admin Console** | http://localhost:8000/admin | Review and approve HALT/REVIEW transactions |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI for all endpoints |
| **Health Check** | http://localhost:8000/api/v1/health | Engine status & model load confirmation |

---

## 🎮 Dashboard — Demo Scenarios

### 🟢 Baseline Flow (~15–20 TPS)
Normal shopping traffic. All transactions show low risk scores and receive instant **APPROVE** decisions.

### 🟡 Flash Sale Surge (~120–160 TPS)
Volume spikes **7.5×** but feature vectors match legitimate buyers. SurgeGuard keeps approving — **zero false declines**.

### 🔴 Coordinated Bot Attack (~150–200 TPS)
High velocity *combined* with anomalous feature vectors (V14, V4, V12 deviations). Risk scores spike above 0.80. **HALT** badges quarantine malicious traffic while concurrent legitimate shoppers are seamlessly cleared.

---

## 🔄 Human-in-the-Loop Adaptive Fraud Learning

SurgeGuard features a production-grade **Human-in-the-Loop (HITL) Adaptive Supervised Learning System**. Rather than treating machine learning decisions as immutable or blindly retraining on noisy operational overrides, SurgeGuard implements explicit human feedback capture, zero-leakage causal feature preservation, and a **Champion vs Challenger** promotion protocol.

```
Suspicious Transaction (HALT / REVIEW)
            │
            ▼
    Admin Investigation
            │
            ▼
   Explicit Ground Truth:
  [ LEGITIMATE (0) ]  [ FRAUD (1) ]
            │
            ▼
Persistent SQLite Feedback Database (`feedback.db`)
(Stores exact 53-dimension feature vector from inference time)
            │
            ▼
Accumulated Feedback ≥ Threshold (or manual trigger)
            │
            ▼
Retraining Engine (`retrain_engine.py`)
Combined Training Data = Base Historical Training Data + Human Verified Feedback
            │
            ▼
Train Challenger LightGBM Classifier
            │
            ▼
Evaluate Challenger vs Champion on Pristine Validation Benchmark
            │
      ┌─────┴────────────────┐
      ▼                      ▼
Validation Passed?      Validation Failed?
      │                      │
   [ YES ]                [ NO ]
      │                      │
Promote Challenger      Reject Challenger
(model_v1 → model_v2)  (model_v1 remains active)
Update active_model.json Preserve audit history
Hot-reload in API
```

### Core Architectural Principles

1. **Separation of Operational Overrides vs ML Ground Truth**:
   - An operational override (`current_decision = "APPROVE"`) unblocks a customer transaction for business continuity.
   - Machine learning ground truth (`verified_label`: `0` for LEGITIMATE, `1` for FRAUD) is explicitly confirmed by an analyst with optional investigation notes. Operational overrides never pollute training data as false positives.

2. **Zero-Leakage Feature Preservation**:
   - When a transaction is scored by `/api/v1/evaluate`, its **complete 53-dimension causal feature vector** (including time-since-previous and causal rolling velocity over 10s, 60s, 300s, 900s) is captured at inference time and stored alongside the feedback record.
   - Retraining uses these preserved features directly, avoiding any recomputation with future data or lookahead leakage.

3. **Catastrophic Drift Protection**:
   - Retraining uses `Base Historical Training Data + Human Verified Feedback`. The model is **never** trained on recent feedback alone, preventing catastrophic forgetting or overfitting on small sample sizes.

4. **Champion vs Challenger Validation Guardrails**:
   - The active production model is the **Champion**.
   - The retrained model is the **Challenger**.
   - Both models are evaluated on the exact same chronological validation split using pure-NumPy ROC-AUC and PR-AUC.
   - If Challenger PR-AUC or ROC-AUC fails validation criteria (e.g. noisy feedback or performance degradation), the Challenger is **rejected** and the Champion remains active in production.
   - Only when Challenger passes validation is it promoted to Active Champion.

5. **Model Versioning & Audit Trail**:
   - Model artifacts are versioned (`models/model_v1.pkl`, `models/model_v2.pkl`).
   - `models/active_model.json` tracks active version, activation timestamp, champion metrics, and full promotion history.
   - The running FastAPI engine hot-reloads the newly promoted model in memory without server restarts.

---

## 🔌 API Reference

### `POST /api/v1/evaluate`
Real-time transaction risk scoring with 53 causal features.

**Request body:**
```json
{
  "Time": 86400.0,
  "Amount": 149.99,
  "V1": -1.35, "V2": -0.07,
  "Account_ID": "ACC-001",
  "Merchant_ID": "MERCH-FLASH-42"
}
```

**Response:**
```json
{
  "transaction_id": "TX-86400-14999",
  "decision": "APPROVE",
  "risk_score": 0.0312,
  "ml_fraud_prob": 0.0289,
  "anomaly_score": 0.0541,
  "velocity_surge_index": 0.12,
  "is_surge_detected": false,
  "latency_ms": 6.91,
  "reasons": ["Normal transactional pattern verified."],
  "context_adapters_active": ["AccountContext", "MerchantContext", "GeoContext"]
}
```

### `POST /api/v1/admin/feedback/{tx_id}`
Submit human-in-the-loop verified ground truth label for a flagged transaction.

**Request body:**
```json
{
  "verified_label": 1,
  "admin_note": "Confirmed coordinated bot account draining",
  "reviewer_id": "senior_analyst_01",
  "allow_override": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Transaction TX-2010-8008 verified as FRAUD.",
  "transaction_id": "TX-2010-8008",
  "verified_label": 1,
  "label_name": "FRAUD",
  "total_feedback_count": 12,
  "retrain_threshold": 10,
  "ready_for_retraining": true,
  "feedback_timestamp": "2026-09-30T00:35:00Z",
  "model_version": "model_v1"
}
```

### `GET /api/v1/admin/feedback`
Returns verified feedback records and summary statistics.

### `POST /api/v1/admin/retrain`
Triggers the adaptive retraining cycle, training a Challenger LightGBM model, evaluating against Champion on the validation benchmark, and promoting if criteria pass.

**Query parameters:**
- `force_promote` (bool, optional, default: `false`): Force promotion if override requested.

**Response:**
```json
{
  "status": "success",
  "candidate_model": "model_v2",
  "promoted": true,
  "previous_model": "model_v1",
  "champion_metrics": { "roc_auc": 0.7801, "pr_auc": 0.6216 },
  "challenger_metrics": { "roc_auc": 0.7937, "pr_auc": 0.6345 },
  "feedback_samples": 12,
  "duration_seconds": 3.59
}
```

### `GET /api/v1/admin/model-status`
Returns active model version, champion metrics, feedback pool size, and promotion audit trail.

### `POST /api/v1/admin/reload-model`
Safely reloads the active model artifact into the live server in memory without downtime.

### `GET /api/v1/admin/flagged`
Returns the most recent HALT/REVIEW transactions (newest first, up to 200).

### `POST /api/v1/admin/approve/{tx_id}`
Admin operational override: approve a previously HALT-ed or REVIEW-ed transaction.

### `GET /api/v1/health`
```json
{ "status": "healthy", "models_loaded": true, "buffer_size": 1024 }
```

---

## 📈 Optional Steps

### Evaluate on Test Set
```bash
python evaluate_test.py
```
Generates `test_predictions.csv` with 42,720 scored transactions.

### Benchmark Latency
```bash
python benchmark_latency.py
```
Stress-tests the engine and reports P50, P90, P99 latencies and maximum throughput.

---

## 🧩 Modular Context Adapter Architecture

SurgeGuard uses pluggable adapters for contextual enrichment. Missing context is **never** treated as a fraud signal.

| Adapter | Status | Purpose |
|:---|:---|:---|
| `AccountContextAdapter` | ✅ Implemented | Account risk factors, device fingerprint hooks |
| `MerchantContextAdapter` | ✅ Implemented | Flash-sale event whitelisting |
| `GeoContextAdapter` | ✅ Implemented | Impossible-speed geo velocity checks |
| Redis feature store | 🔧 Pluggable | Real-time account history lookup |
| Fraud intel API | 🔧 Pluggable | External threat intelligence feeds |

---

## 🛠️ Troubleshooting

| Problem | Solution |
|:---|:---|
| `ModuleNotFoundError: No module named 'uvicorn'` | `pip install uvicorn` |
| `ModuleNotFoundError: No module named 'fastapi'` | `pip install fastapi` |
| `ModuleNotFoundError: No module named 'lightgbm'` | `pip install lightgbm` |
| `503 — Risk Engine models still compiling` | Run `python ml_pipeline.py` first to generate `risk_engine_artifacts.pkl` |
| Admin page shows **ENGINE: OFFLINE** | Server is not running — start with `python run_server.py` |
| Dashboard stops after navigating to Admin | Always use server URLs (`/` and `/admin`), not file:// paths |

---

## 📄 License

This project is licensed under the MIT License.

---

*Built for the 36-Hour YUVA Megathon.*
