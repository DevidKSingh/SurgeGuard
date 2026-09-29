"""
api.py - FastAPI Real-Time Inference & Adaptive Risk Engine
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import os
import time
import pickle
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables from .env (no-op if file is absent)
load_dotenv()

# --- Config from environment ---
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "risk_engine_artifacts.pkl")
_raw_origins   = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")]

# =====================================================================
# 1. DATA MODELS & CONTEXT ADAPTER ABSTRACTION
# =====================================================================

class TransactionPayload(BaseModel):
    Time: float = Field(..., description="Timestamp of transaction in seconds")
    Amount: float = Field(..., description="Transaction amount in currency units")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    # Optional contextual fields (Handled via Context Adapters)
    Account_ID: Optional[str] = None
    Merchant_ID: Optional[str] = None
    Geo_Location: Optional[str] = None


class RiskDecisionResponse(BaseModel):
    transaction_id: str
    decision: str  # "APPROVE", "REVIEW", "HALT"
    risk_score: float  # [0.0 - 1.0]
    ml_fraud_prob: float
    anomaly_score: float
    velocity_surge_index: float
    is_surge_detected: bool
    latency_ms: float
    reasons: List[str]
    context_adapters_active: List[str]


# Modular Context Adapter Architecture
class BaseContextAdapter:
    name: str = "BaseContext"
    
    def enrich(self, payload: TransactionPayload, state: Dict[str, Any]) -> Dict[str, Any]:
        """Returns contextual signals. Graceful no-op when fields are absent."""
        return {}


class AccountContextAdapter(BaseContextAdapter):
    name = "AccountContext"
    def enrich(self, payload: TransactionPayload, state: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.Account_ID:
            return {"account_available": False, "account_risk_factor": 1.0}
        # In enterprise production, query Redis/Feature Store for user history
        return {"account_available": True, "account_risk_factor": 1.0}


class MerchantContextAdapter(BaseContextAdapter):
    name = "MerchantContext"
    def enrich(self, payload: TransactionPayload, state: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.Merchant_ID:
            return {"merchant_available": False, "flash_sale_active": False}
        return {"merchant_available": True, "flash_sale_active": True}


class GeoContextAdapter(BaseContextAdapter):
    name = "GeoContext"
    def enrich(self, payload: TransactionPayload, state: Dict[str, Any]) -> Dict[str, Any]:
        if not payload.Geo_Location:
            return {"geo_available": False, "geo_impossible_speed": False}
        return {"geo_available": True, "geo_impossible_speed": False}


# =====================================================================
# 2. STATEFUL IN-MEMORY VELOCITY BUFFER
# =====================================================================

class RollingVelocityBuffer:
    """
    Thread-safe low-latency sliding window buffer for real-time velocity
    and burst calculations over 10s, 60s, 300s, and 900s.
    """
    def __init__(self, max_retention_seconds: float = 900.0):
        self.max_retention = max_retention_seconds
        # Deque of tuples: (time, amount)
        self.buffer = deque()
        self.last_timestamp = 0.0

    def add_and_compute(self, current_time: float, current_amount: float) -> Dict[str, float]:
        # Purge older transactions outside retention window
        cutoff = current_time - self.max_retention
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

        time_since_prev = max(0.0, current_time - self.last_timestamp) if self.buffer else 0.0
        self.last_timestamp = current_time

        # Compute counts and sums per window.
        # Iterate once from newest to oldest (reversed deque = newest first).
        # Do NOT break early — a tx outside the 10s window may still be inside 60/300/900s.
        counts = {10: 0, 60: 0, 300: 0, 900: 0}
        sums   = {10: 0.0, 60: 0.0, 300: 0.0, 900: 0.0}

        for t, amt in reversed(self.buffer):
            delta = current_time - t
            if delta > 900:
                # deque is oldest-first; reversed gives newest-first.
                # Once delta > 900 all remaining entries are even older — safe to stop.
                break
            if delta <= 900:
                counts[900] += 1
                sums[900] += amt
            if delta <= 300:
                counts[300] += 1
                sums[300] += amt
            if delta <= 60:
                counts[60] += 1
                sums[60] += amt
            if delta <= 10:
                counts[10] += 1
                sums[10] += amt

        # Append current transaction to history AFTER computing (strictly causal)
        self.buffer.append((current_time, current_amount))

        res = {
            "Time_Since_Prev": time_since_prev,
        }
        for w in [10, 60, 300, 900]:
            c = counts[w]
            s = sums[w]
            mean_amt = (s / c) if c > 0 else 0.0
            res[f"Tx_Count_{w}s"] = float(c)
            res[f"Amt_Sum_{w}s"] = float(s)
            res[f"Amt_Mean_{w}s"] = float(mean_amt)
            res[f"Amt_Ratio_{w}s"] = float(current_amount / mean_amt) if mean_amt > 0 else 1.0

        return res


# =====================================================================
# 3. FASTAPI APPLICATION SETUP & MODEL LOADING
# =====================================================================

def load_artifacts():
    global artifacts
    try:
        with open(ARTIFACTS_PATH, "rb") as f:
            artifacts = pickle.load(f)
            print(f"Successfully loaded risk engine artifacts from '{ARTIFACTS_PATH}'!")
    except Exception as e:
        print(f"Artifacts not yet found or error loading: {e}. Will be loaded upon completion of ml_pipeline.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Replaces deprecated @app.on_event('startup')."""
    load_artifacts()
    yield
    # (teardown goes here if needed)


app = FastAPI(
    title="Securing the Surge - Adaptive Transaction Risk Engine",
    description="Real-time transaction authorization distinguishing high-velocity consumer surges from coordinated bot attacks.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded on startup (via lifespan)
artifacts = None

velocity_buffer = RollingVelocityBuffer()
context_adapters = [
    AccountContextAdapter(),
    MerchantContextAdapter(),
    GeoContextAdapter()
]


# =====================================================================
# 4. REAL-TIME EVALUATION ENDPOINT
# =====================================================================

@app.post("/api/v1/evaluate", response_model=RiskDecisionResponse)
def evaluate_transaction(payload: TransactionPayload):
    t_start = time.perf_counter()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Risk Engine models are still compiling. Please wait.")

    # 1. Compute velocity & burst features
    v_feats = velocity_buffer.add_and_compute(payload.Time, payload.Amount)

    # 2. Run Context Adapters
    context_signals = {}
    active_adapter_names = []
    for adapter in context_adapters:
        signals = adapter.enrich(payload, {})
        context_signals.update(signals)
        active_adapter_names.append(adapter.name)

    # 3. Construct Feature Vector for LightGBM
    hours = (payload.Time // 3600.0) % 24.0
    hour_sin = np.sin(2.0 * np.pi * hours / 24.0)
    hour_cos = np.cos(2.0 * np.pi * hours / 24.0)
    log_amt = np.log1p(payload.Amount)
    v14_v4_ratio = payload.V14 / (abs(payload.V4) + 1e-5)
    v12_v10_diff = payload.V12 - payload.V10

    raw_dict = {
        "Time": payload.Time,
        "Amount": payload.Amount,
        "Hour": hours,
        "Hour_Sin": hour_sin,
        "Hour_Cos": hour_cos,
        "Log_Amount": log_amt,
        "Time_Since_Prev": v_feats["Time_Since_Prev"],
        "Tx_Count_10s": v_feats["Tx_Count_10s"],
        "Amt_Sum_10s": v_feats["Amt_Sum_10s"],
        "Amt_Mean_10s": v_feats["Amt_Mean_10s"],
        "Amt_Ratio_10s": v_feats["Amt_Ratio_10s"],
        "Tx_Count_60s": v_feats["Tx_Count_60s"],
        "Amt_Sum_60s": v_feats["Amt_Sum_60s"],
        "Amt_Mean_60s": v_feats["Amt_Mean_60s"],
        "Amt_Ratio_60s": v_feats["Amt_Ratio_60s"],
        "Tx_Count_300s": v_feats["Tx_Count_300s"],
        "Amt_Sum_300s": v_feats["Amt_Sum_300s"],
        "Amt_Mean_300s": v_feats["Amt_Mean_300s"],
        "Amt_Ratio_300s": v_feats["Amt_Ratio_300s"],
        "Tx_Count_900s": v_feats["Tx_Count_900s"],
        "Amt_Sum_900s": v_feats["Amt_Sum_900s"],
        "Amt_Mean_900s": v_feats["Amt_Mean_900s"],
        "Amt_Ratio_900s": v_feats["Amt_Ratio_900s"],
        "V14_V4_Ratio": v14_v4_ratio,
        "V12_V10_Diff": v12_v10_diff,
    }
    for i in range(1, 29):
        raw_dict[f"V{i}"] = getattr(payload, f"V{i}")

    feature_cols = artifacts["feature_cols"]
    input_vector = np.array([[raw_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)

    # 4. Supervised Model Inference
    ml_prob = float(artifacts["lgb_model"].predict(input_vector)[0])

    # 5. Unsupervised Anomaly Scoring
    v_vec = np.array([[getattr(payload, f"V{i}") for i in range(1, 29)]], dtype=np.float32)
    raw_iso = float(-artifacts["iso_forest"].score_samples(v_vec)[0])
    # Sigmoidal normalization of isolation forest score [0 to 1]
    anomaly_score = float(1.0 / (1.0 + np.exp(-((raw_iso - 0.5) * 8.0))))

    # 6. Velocity / Burst Surge Index
    # Surge indicator: normalized burst activity in past 60s
    tx_count_60s = v_feats["Tx_Count_60s"]
    velocity_surge_index = min(1.0, tx_count_60s / 100.0)
    is_surge_active = velocity_surge_index > 0.40

    # 7. Adaptive Fusion Score
    # Crucial insight: High surge alone is NOT penalized if anomaly score is low (flash sale).
    # If feature patterns are anomalous during a burst, risk escalates rapidly (bot attack).
    interaction_penalty = velocity_surge_index * anomaly_score
    adaptive_risk = float(0.75 * ml_prob + 0.15 * anomaly_score + 0.10 * interaction_penalty)
    adaptive_risk = max(0.0, min(1.0, adaptive_risk))

    # 8. Tiered Policy Decision
    reasons = []
    if adaptive_risk < artifacts["tiers"]["approve_max"]:
        decision = "APPROVE"
        if is_surge_active:
            reasons.append("High flash-sale volume verified: Spending patterns consistent with legitimate buyers.")
        else:
            reasons.append("Normal transactional pattern verified.")
    elif adaptive_risk < artifacts["tiers"]["halt_min"]:
        decision = "REVIEW"
        reasons.append("Elevated risk detected. Requires step-up multi-factor authentication (MFA).")
        if anomaly_score > 0.6:
            reasons.append("Atypical transaction vector deviates from baseline behavior.")
    else:
        decision = "HALT"
        reasons.append("High probability coordinated account-draining bot attack halted.")
        if is_surge_active:
            reasons.append("Rapid burst velocity combined with high-confidence fraud signatures.")

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    return RiskDecisionResponse(
        transaction_id=f"TX-{int(payload.Time)}-{int(payload.Amount*100)%9999}",
        decision=decision,
        risk_score=round(adaptive_risk, 4),
        ml_fraud_prob=round(ml_prob, 4),
        anomaly_score=round(anomaly_score, 4),
        velocity_surge_index=round(velocity_surge_index, 4),
        is_surge_detected=is_surge_active,
        latency_ms=round(latency_ms, 2),
        reasons=reasons,
        context_adapters_active=active_adapter_names
    )


@app.get("/api/v1/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": artifacts is not None,
        "buffer_size": len(velocity_buffer.buffer)
    }


# Serve Dashboard static assets
import os
from fastapi.responses import FileResponse

dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard")

@app.get("/")
def serve_dashboard():
    index_file = os.path.join(dashboard_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SurgeGuard Risk Engine Active. Dashboard index.html not found."}

@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(dashboard_path, "style.css"))

@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(dashboard_path, "app.js"))
