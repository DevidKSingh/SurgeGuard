# 🛡️ SurgeGuard: Securing the Surge
## Protecting Digital Transactions During Peak E-Commerce Events
**36-Hour Hackathon Final Presentation & Technical Pitch**

---

## 1. The Executive Pitch (30-Second Hook)

> **"During flash sales and holiday surges, transaction volume spikes 10x. Legacy banking security panics: static thresholds block genuine shoppers, costing millions in lost revenue, while loosening thresholds allows distributed bot attacks to drain customer accounts.**
> 
> **Our solution, SurgeGuard, proves that *High Transaction Volume $\neq$ Fraud*. Using an Adaptive Multi-Signal Risk Engine, SurgeGuard differentiates legitimate localized flash-sale surges from coordinated bot attacks in under 7 milliseconds—ensuring 96.6% friction-free approvals for genuine buyers while halting 78.1% of fraudulent account drains."**

---

## 2. The Core Problem vs. The SurgeGuard Solution

```
┌───────────────────────────────────────┐       ┌───────────────────────────────────────┐
│        LEGACY RULE-BASED SYSTEM       │       │         SURGEGUARD ADAPTIVE SYSTEM    │
├───────────────────────────────────────┤       ├───────────────────────────────────────┤
│ • Static velocity thresholds          │  VS   │ • Tri-signal dynamic risk engine      │
│ • "Spike = Suspicious" mindset        │       │ • "Surge ≠ Fraud" contextual awareness│
│ • Massive false positive declines     │       │ • 96.57% zero-friction clearance      │
│ • Latency bottlenecks during peaks    │       │ • 6.91 ms latency / 183,000+ TPS      │
│ • Brittle to new fraud patterns       │       │ • Supervised + Unsupervised Anomaly   │
└───────────────────────────────────────┘       └───────────────────────────────────────┘
```

---

## 3. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Real-Time Transaction Ingestion (<1ms)"]
        TX[Incoming Transaction Stream] --> Clean[Data Quality & Hygiene Layer]
    end

    subgraph CausalFeatures ["2. Stateful Causal Velocity Engine"]
        Clean --> RingBuffer[(In-Memory Rolling Window Buffer\n10s | 60s | 300s | 900s)]
        RingBuffer --> Feats[Causal Temporal Ratios & Burst Density\nAmt_Mean_10s, Amt_Mean_60s, V12_V10_Diff]
    end

    subgraph TriSignal ["3. Tri-Signal Adaptive Risk Engine"]
        Feats --> M1[Supervised Classifier\nClass-Weighted LightGBM\nPR-AUC: 0.6295]
        Feats --> M2[Unsupervised Anomaly\nIsolation Forest Drift Detector\nZero-Day Attack Isolation]
        Feats --> M3[Surge Burst Multiplier\nZ-Score Density Normalizer]
    end

    subgraph Fusion ["4. Calibrated Decision Engine & Adapters"]
        M1 & M2 & M3 --> RiskScore["Composite Adaptive Risk Score\n0.75·P(ML) + 0.15·S(Anomaly) + 0.10·(Burst · Anomaly)"]
        RiskScore --> ContextAdapters[["Modular Context Adapters\n• Account History\n• Merchant Flash Sale\n• Geo & Telemetry"]]
    end

    subgraph Policy ["5. Three-Tiered Real-Time Action Policy"]
        ContextAdapters --> Dec1["✅ APPROVE (< 0.359)\n96.57% Clearance\nZero Friction for Shoppers"]
        ContextAdapters --> Dec2["⚠️ REVIEW [0.359, 0.798)\n2.63% Stepped-Up MFA\nBiometric / OTP Challenge"]
        ContextAdapters --> Dec3["🛑 HALT (>= 0.798)\n0.80% Quarantine\n78.1% Attacks Neutralized"]
    end

    subgraph Monitoring ["6. Live Command Dashboard"]
        Policy --> UI["Real-Time Monitoring UI (http://127.0.0.1:8000)\n• Live TPS Speedometer\n• Tri-Signal Decomposition\n• Scenario Simulator"]
    end
```

---

## 4. Empirical Performance Benchmarks (test.csv & Out-of-Sample)

### A. Model Accuracy & Discrimination
* **Test ROC-AUC:** `0.9667`
* **Test PR-AUC:** `0.4560` (Exceptional discrimination at 0.17% fraud prevalence)
* **Training Validation PR-AUC:** `0.6295` (Strict time-aware split, 0 future leakage)

### B. Business & Operational Policy Metrics
| Metric | Result | Operational Value |
| :--- | :--- | :--- |
| **Friction-Free Clearance** | **96.57%** (41,255 txs) | Ensures high cart-conversion during limited-time flash sales |
| **Fraud Defense Interception** | **78.08%** (57/73 frauds) | Blocks high-confidence automated draining scripts immediately |
| **Step-Up Review Rate** | **2.63%** (1,125 txs) | Minimizes manual review queues; leverages automated OTP |
| **Causal Feature Extraction** | **0.12 sec** for 200,000 txs | Vectorized prefix sum algorithm enabling real-time operation |

### C. Latency & Surge Capacity
* **Median In-Line Latency (P50):** `6.91 ms`
* **P90 Latency:** `8.52 ms`
* **Surge Batch Throughput:** **`183,727 TPS`** at `0.005 ms/tx`
* **Target:** Real-world constraint of `< 10 ms` **Fully Satisfied**.

---

## 5. Solving the Dataset Architecture Dilemma: Modular Context Adapters

### The Challenge:
The hackathon problem statement describes rich user demographics, merchant categories, and geographic locations. However, the provided dataset (`Dataset 1`) contains anonymized features `V1–V28`, `Time`, and `Amount`.

### Our Architectural Answer:
Rather than fabricating synthetic fields or hardcoding brittle assumptions, we designed the **Modular Context Adapter Architecture**:
1. **Separation of Concerns:** The core risk engine evaluates transactional features and causal velocity without requiring relational identifiers.
2. **Pluggable Adapters:** We implemented abstract adapters for:
   * `AccountContextAdapter`: Ready to inject historical account baselines and device fingerprint matches.
   * `MerchantContextAdapter`: Ready to whitelist registered flash-sale events and merchant categories.
   * `GeoContextAdapter`: Ready to verify IP velocity and impossible-speed anomalies.
3. **Enterprise Robustness Principle:** *Missing context is an architectural reality, never treated as a fraud signal.*

---

## 6. Live Demo Walkthrough (For the Judges)

When demonstrating `http://127.0.0.1:8000`, toggle the **Demo Simulation Controller**:

### Scenario A: Normal Stream (Nominal Operations)
* **What you see:** Steady ~15–20 TPS, nominal amounts, low risk scores ($< 0.05$).
* **System Action:** Instant green **APPROVE** badges across the transaction ledger.

### Scenario B: Flash Sale Surge (The Core Test)
* **What you see:** Volume spikes to **120–160 TPS** (7.5x jump!). Transaction velocity bars surge to max.
* **Why SurgeGuard Wins:** The anomaly score remains low ($< 0.15$) because transaction vectors match legitimate consumer behavior.
* **System Action:** Approvals remain friction-free (**zero legitimate buyers falsely declined**).

### Scenario C: Coordinated Distributed Bot Attack
* **What you see:** Rapid burst velocity coupled with high feature vector deviation ($V_{14}, V_4, V_{12}$ anomalies) and rapid-fire drainage attempts.
* **System Action:** The risk score escalates to $> 0.80$. Neon red **HALT** badges immediately quarantine the malicious bot traffic, while concurrent legitimate shoppers are seamlessly cleared.

---

## 7. Deliverables & Repository Structure

* **`ml_pipeline.py`**: Leakage-free causal velocity feature engineering & model training.
* **`api.py`**: High-performance FastAPI real-time scoring engine with in-memory sliding buffer.
* **`dashboard/`**: Glassmorphic financial command center (`index.html`, `style.css`, `app.js`).
* **`evaluate_test.py`**: Out-of-sample evaluation pipeline producing `test_predictions.csv`.
* **`benchmark_latency.py`**: Stress profiling & high-throughput capacity validator.
* **`risk_engine_artifacts.pkl`**: Serialized model weights, feature transformers, and calibrated decision tiers.
* **`test_predictions.csv`**: Complete 42,720-row prediction dataset with multi-tier decisions.
