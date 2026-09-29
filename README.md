# SurgeGuard — Securing the Surge (Fraud Detection)

> **Problem Statement 1 · Megathon Hackathon Submission**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen)](https://lightgbm.readthedocs.io/)

SurgeGuard is a real-time fraud detection engine that distinguishes coordinated bot attacks from legitimate flash-sale traffic. It uses a **Tri-Signal Adaptive Risk Engine** — LightGBM (75%) + Isolation Forest (15%) + Velocity Burst (10%) — to score transactions in under 7 ms.

---

## Repository Structure

```
SurgeGuard/
├── predict.py                    # Standalone inference script (start here)
├── requirements.txt              # Pinned dependencies
├── README.md                     # This file
├── models/
│   └── risk_engine_artifacts.pkl # Serialized trained model + calibrated tiers
└── src/
    ├── ml_pipeline.py            # Feature engineering + model training
    ├── evaluate_test.py          # Out-of-sample evaluation script
    ├── api.py                    # FastAPI real-time scoring engine
    ├── run_server.py             # Uvicorn server launcher
    ├── benchmark_latency.py      # Latency stress profiler
    ├── test.py                   # Unit tests
    └── dashboard/                # Live transaction dashboard (HTML/JS/CSS)
        ├── index.html
        ├── style.css
        ├── app.js
        ├── admin.html
        ├── admin.js
        └── admin-style.css
```

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.10+**. All packages and exact versions are listed in `requirements.txt`.

---

## Step 2 — Run Inference with predict.py

The `predict.py` script loads the pre-trained model from `models/` and generates predictions on any input CSV **without retraining**.

```bash
python predict.py --input path/to/test.csv --output path/to/predictions.csv
```

**Arguments:**

| Argument | Required | Description |
|:---|:---|:---|
| `--input` | ✅ Yes | Path to the input CSV file (must contain `Time`, `Amount`, `V1`–`V28`) |
| `--output` | ✅ Yes | Path where the output prediction CSV will be saved |
| `--models-dir` | ❌ No | Directory containing the `.pkl` artifact (default: `models`) |

**Example:**

```bash
python predict.py --input test.csv --output predictions.csv
```

**Output CSV format** (row-by-row, same order as input):

| Column | Type | Description |
|:---|:---|:---|
| `fraud_probability` | float [0–1] | Model's predicted probability of fraud |
| `predicted_class` | int (0 or 1) | Binary label — 1 = Fraud, 0 = Legitimate |

---

## Step 3 — (Optional) Retrain the Model

> Skip this step — `models/risk_engine_artifacts.pkl` is already included.

If you need to retrain from scratch (requires `train_cleaned.csv` and `test_cleaned.csv`):

```bash
python src/ml_pipeline.py
```

Move the generated artifact:

```bash
# Windows
move risk_engine_artifacts.pkl models\risk_engine_artifacts.pkl

# macOS / Linux
mv risk_engine_artifacts.pkl models/risk_engine_artifacts.pkl
```

---

## Step 4 — (Optional) Launch the Live Dashboard

```bash
python src/run_server.py
```

| Page | URL |
|:---|:---|
| Live Dashboard | http://localhost:8000 |
| Admin Console | http://localhost:8000/admin |
| API Docs | http://localhost:8000/docs |

---

## Model Performance

| Metric | Score |
|:---|:---|
| Test ROC-AUC | **0.9667** |
| Test PR-AUC | **0.4560** |
| Fraud Defense Interception | **78.08%** (57/73 frauds blocked) |
| Friction-Free Clearance | **96.57%** (legitimate txs approved) |
| Inference Latency (P50) | **6.91 ms** |

---

## Troubleshooting

| Error | Fix |
|:---|:---|
| `ModuleNotFoundError: lightgbm` | `pip install -r requirements.txt` |
| `Model artifact not found` | Ensure `models/risk_engine_artifacts.pkl` exists |
| `Missing required columns` | Input CSV must have `Time`, `Amount`, `V1`–`V28` |
