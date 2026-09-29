# 🛡️ SurgeGuard — Securing the Surge

> **Real-Time Banking Fraud Detection & Adaptive Transaction Security During Peak E-Commerce Events**

SurgeGuard differentiates legitimate flash-sale volume surges from distributed bot attacks in **under 1 millisecond** using a **Tri-Signal Adaptive Risk Engine** (Supervised LightGBM + Isolation Forest Anomaly Detection + Velocity Surge Index).

---

## ⚡ Quick Start: Standalone Inference (`predict.py`)

Judges and evaluators can immediately evaluate any test transaction dataset using the standalone `predict.py` script without retraining.

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Inference

```bash
python predict.py --input test_cleaned.csv --output predictions.csv
```

**Parameters:**
- `--input`, `-i`: Path to the input test CSV file (e.g. `test.csv` or `test_cleaned.csv`).
- `--output`, `-o`: Path to save the resulting prediction CSV file (e.g. `predictions.csv`).

### 3. Output Format

The output CSV contains predictions aligned **row-by-row** with the input transactions:
- `prediction`: Binary classification (`0` = Legitimate, `1` = Fraud).
- `fraud_probability`: Continuous calibrated fraud probability in range `[0.0, 1.0]`.
- `risk_score`: Tri-signal adaptive composite risk score in range `[0.0, 1.0]`.
- `decision`: Operational policy decision (`APPROVE`, `REVIEW`, `HALT`).

---

## 🖥️ Running the Real-Time Web Console

To launch the real-time banking fraud operations console and live transaction authorization feed:

```bash
python src/run_server.py
```

Open in your browser:
- **Fraud Operations & Audit Console**: [http://localhost:8000/admin](http://localhost:8000/admin)
- **Live Authorization Stream**: [http://localhost:8000/](http://localhost:8000/)

---

## 🧪 Verification & Testing

To run the automated verification test suite:

```bash
python src/test.py
```

This verifies:
1. Model artifact integrity in `models/risk_engine_artifacts.pkl`.
2. Causal feature pipeline extraction (exact 53 features with zero future leakage).
3. Out-of-sample prediction and calibrated tier boundaries.
4. End-to-end `predict.py` command-line execution on 42,691 test rows.

---

## 📁 Submission Package Structure

```
SurgeGuard/
├── predict.py                         # Standalone inference script (--input and --output)
├── requirements.txt                   # Exact pinned dependencies
├── README.md                          # Installation and execution instructions
├── .gitignore                         # Git exclusion rules (whitelists trained model)
├── models/
│   └── risk_engine_artifacts.pkl      # Serialized trained model weights + calibrated tiers
└── src/
    ├── ml_pipeline.py                 # Data hygiene, feature engineering & model training
    ├── api.py                         # FastAPI real-time scoring engine & endpoints
    ├── evaluate_test.py               # Out-of-sample evaluation on test data
    ├── benchmark_latency.py           # Latency & throughput stress profiler
    ├── run_server.py                  # Server launcher
    ├── test.py                        # Automated verification test suite
    └── dashboard/                     # Banking SOC & Fraud Operations Console
        ├── index.html                 # Real-time authorization feed UI
        ├── style.css                  # Centralized dark design system
        ├── app.js                     # Live stream & scenario controller
        ├── admin.html                 # Fraud Operations & Model Center UI
        ├── admin.js                   # Review queue, feedback & model controller
        └── admin-style.css            # Console & investigation drawer styles
```
