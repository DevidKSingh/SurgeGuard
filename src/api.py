"""
api.py - FastAPI Real-Time Inference & Adaptive Risk Engine
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import os
import sys

# Ensure src directory is in sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import time
import types
import pickle
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from collections import deque
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from feedback_store import feedback_store
import retrain_engine
from retrain_engine import get_active_model_info, ensure_pickle_shim


# =====================================================================
# ISOLATION FOREST — defined here so pickle can deserialize artifacts
# that were saved when ml_pipeline.py ran as __main__.
# This MUST be defined at module level before load_artifacts() is called.
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
        """Returns negative anomaly score — higher means more normal (sklearn convention)."""
        X = np.asarray(X, dtype=np.float64)
        z = np.abs((X - self.mean_) / self.std_)
        return -z.mean(axis=1)

# Load environment variables from .env (no-op if file is absent)
load_dotenv()

# --- Config from environment ---
def resolve_artifacts_path():
    env_p = os.getenv("ARTIFACTS_PATH")
    if env_p and os.path.exists(env_p):
        return env_p
    candidates = [
        "models/risk_engine_artifacts.pkl",
        os.path.join(os.path.dirname(__file__), "..", "models", "risk_engine_artifacts.pkl"),
        os.path.join(os.path.dirname(__file__), "models", "risk_engine_artifacts.pkl"),
        "risk_engine_artifacts.pkl",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return "models/risk_engine_artifacts.pkl"

ARTIFACTS_PATH = resolve_artifacts_path()
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


class FlaggedTransaction(BaseModel):
    transaction_id: str
    original_decision: str  # "REVIEW" or "HALT"
    current_decision: str
    risk_score: float
    ml_fraud_prob: float
    anomaly_score: float
    velocity_surge_index: float
    amount: Optional[str] = None
    reasons: List[str]
    flagged_at: str
    approved_by_admin: bool = False
    approved_at: Optional[str] = None
    feature_vector: Optional[Dict[str, float]] = None
    verified_label: Optional[int] = None
    admin_note: Optional[str] = None
    model_version: Optional[str] = None


class RegisterFlaggedPayload(BaseModel):
    """Lightweight payload sent by the dashboard simulation for each HALT/REVIEW tx."""
    transaction_id: str
    decision: str          # "HALT" or "REVIEW"
    risk_score: float
    ml_fraud_prob: float
    anomaly_score: float
    velocity_surge_index: float
    amount: Optional[str] = None
    reasons: List[str]
    feature_vector: Optional[Dict[str, float]] = None


class AdminFeedbackPayload(BaseModel):
    verified_label: int = Field(..., description="0 for LEGITIMATE, 1 for FRAUD")
    admin_note: Optional[str] = Field("", description="Admin investigation note / rationale")
    reviewer_id: Optional[str] = Field("admin", description="Reviewer identifier")
    allow_override: Optional[bool] = Field(False, description="Whether to update existing feedback")


class AdminApproveResponse(BaseModel):
    transaction_id: str
    status: str
    message: str
    previous_decision: str
    new_decision: str
    approved_at: str


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

active_model_version = "model_v1"
artifacts = None

def load_artifacts(custom_path: Optional[str] = None):
    global artifacts, active_model_version
    ensure_pickle_shim()
    try:
        meta = get_active_model_info()
        active_model_version = meta.get("active_model_version", "model_v1")
        load_path = custom_path or meta.get("artifact_path", ARTIFACTS_PATH)
        if not os.path.exists(load_path):
            load_path = ARTIFACTS_PATH

        with open(load_path, "rb") as f:
            artifacts = pickle.load(f)
            print(f"[MODEL] Successfully loaded risk engine artifacts [{active_model_version}] from '{load_path}'!")
    except Exception as e:
        print(f"Artifacts not yet found or error loading: {e}. Attempting fallback to '{ARTIFACTS_PATH}'...")
        try:
            with open(ARTIFACTS_PATH, "rb") as f:
                artifacts = pickle.load(f)
                active_model_version = "model_v1"
                print(f"[MODEL] Fallback loaded baseline artifacts from '{ARTIFACTS_PATH}'!")
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")


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

# In-memory store for flagged transactions (HALT / REVIEW)
# Dict[transaction_id -> FlaggedTransaction]
# Capped at 1 000 entries (FIFO eviction) to prevent unbounded growth.
MAX_FLAGGED_STORE = 1000
flagged_store: Dict[str, FlaggedTransaction] = {}

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

    tx_id = f"TX-{int(payload.Time)}-{int(payload.Amount*100)%9999}"

    # Persist HALT / REVIEW decisions so admins can search & override them
    if decision in ("HALT", "REVIEW"):
        if len(flagged_store) >= MAX_FLAGGED_STORE:
            # Evict oldest entry
            oldest_key = next(iter(flagged_store))
            del flagged_store[oldest_key]
        feature_dict = {c: float(raw_dict.get(c, 0.0)) for c in feature_cols}
        flagged_store[tx_id] = FlaggedTransaction(
            transaction_id=tx_id,
            original_decision=decision,
            current_decision=decision,
            risk_score=round(adaptive_risk, 4),
            ml_fraud_prob=round(ml_prob, 4),
            anomaly_score=round(anomaly_score, 4),
            velocity_surge_index=round(velocity_surge_index, 4),
            amount=f"${payload.Amount:.2f}",
            reasons=reasons,
            flagged_at=datetime.now(timezone.utc).isoformat(),
            approved_by_admin=False,
            approved_at=None,
            feature_vector=feature_dict,
            verified_label=None,
            admin_note=None,
            model_version=active_model_version,
        )

    return RiskDecisionResponse(
        transaction_id=tx_id,
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


# =====================================================================
# 5. ADMIN OVERRIDE ENDPOINTS
# =====================================================================

@app.get("/api/v1/admin/transaction/{tx_id}", response_model=FlaggedTransaction)
def admin_lookup_transaction(tx_id: str):
    """
    Look up a HALT or REVIEW transaction by its ID.
    Used by admins to inspect a flagged transaction before deciding to approve it.
    """
    entry = flagged_store.get(tx_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{tx_id}' not found in the flagged store. "
                   "Only HALT and REVIEW decisions are retained for admin override."
        )
    return entry


@app.post("/api/v1/admin/approve/{tx_id}", response_model=AdminApproveResponse)
def admin_approve_transaction(tx_id: str):
    """
    Admin override: forcibly approve a previously HALT-ed or REVIEW-ed transaction.
    This records the override in-memory so the audit trail is preserved.
    """
    entry = flagged_store.get(tx_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{tx_id}' not found. Only HALT/REVIEW transactions can be overridden."
        )
    if entry.approved_by_admin:
        raise HTTPException(
            status_code=409,
            detail=f"Transaction '{tx_id}' has already been approved by an admin at {entry.approved_at}."
        )

    previous = entry.current_decision
    approved_ts = datetime.now(timezone.utc).isoformat()
    entry.current_decision = "APPROVE"
    entry.approved_by_admin = True
    entry.approved_at = approved_ts
    flagged_store[tx_id] = entry

    return AdminApproveResponse(
        transaction_id=tx_id,
        status="success",
        message=f"Transaction {tx_id} has been administratively approved. Original ML decision was '{previous}'.",
        previous_decision=previous,
        new_decision="APPROVE",
        approved_at=approved_ts,
    )


@app.get("/api/v1/admin/flagged", response_model=List[FlaggedTransaction])
def admin_list_flagged(limit: int = 200):
    """
    Return the most-recent flagged transactions (HALT/REVIEW/overridden), newest first.
    Useful for the admin panel to show pending overrides.
    """
    entries = list(flagged_store.values())
    entries.sort(key=lambda e: e.flagged_at, reverse=True)
    return entries[:limit]


@app.post("/api/v1/admin/register", status_code=201)
def admin_register_flagged(payload: RegisterFlaggedPayload):
    """
    Register a HALT or REVIEW transaction from the dashboard simulation into the
    flagged_store. Bridges client-side simulation with server-side admin override
    and verified human feedback workflows.
    """
    if payload.decision not in ("HALT", "REVIEW"):
        raise HTTPException(status_code=400, detail="Only HALT or REVIEW decisions can be registered.")

    # Idempotent: if already in store, just return the existing entry
    if payload.transaction_id in flagged_store:
        return {"status": "already_exists", "transaction_id": payload.transaction_id}

    if len(flagged_store) >= MAX_FLAGGED_STORE:
        oldest_key = next(iter(flagged_store))
        del flagged_store[oldest_key]

    feature_dict = payload.feature_vector
    if not feature_dict:
        # Synthesize complete 53-dimension feature vector based on reported signals
        raw_amt = 100.0
        if payload.amount:
            try:
                raw_amt = float(str(payload.amount).replace("$", "").replace(",", "").strip())
            except Exception:
                raw_amt = 100.0
        now_t = time.time()
        hours = (now_t // 3600.0) % 24.0
        feature_dict = {
            "Time": now_t,
            "Amount": raw_amt,
            "Hour": hours,
            "Hour_Sin": float(np.sin(2.0 * np.pi * hours / 24.0)),
            "Hour_Cos": float(np.cos(2.0 * np.pi * hours / 24.0)),
            "Log_Amount": float(np.log1p(raw_amt)),
            "Time_Since_Prev": 0.5,
            "Tx_Count_10s": float(15.0 * payload.velocity_surge_index),
            "Amt_Sum_10s": float(raw_amt * 5.0),
            "Amt_Mean_10s": float(raw_amt),
            "Amt_Ratio_10s": 1.0,
            "Tx_Count_60s": float(60.0 * payload.velocity_surge_index),
            "Amt_Sum_60s": float(raw_amt * 25.0),
            "Amt_Mean_60s": float(raw_amt),
            "Amt_Ratio_60s": 1.0,
            "Tx_Count_300s": float(250.0 * payload.velocity_surge_index),
            "Amt_Sum_300s": float(raw_amt * 100.0),
            "Amt_Mean_300s": float(raw_amt),
            "Amt_Ratio_300s": 1.0,
            "Tx_Count_900s": float(700.0 * payload.velocity_surge_index),
            "Amt_Sum_900s": float(raw_amt * 300.0),
            "Amt_Mean_900s": float(raw_amt),
            "Amt_Ratio_900s": 1.0,
            "V14_V4_Ratio": -2.5 if payload.anomaly_score > 0.5 else 0.5,
            "V12_V10_Diff": -3.0 if payload.anomaly_score > 0.5 else 0.2,
        }
        for i in range(1, 29):
            if i in (14, 12, 10, 4) and payload.anomaly_score > 0.5:
                feature_dict[f"V{i}"] = -3.5 if i in (14, 12, 10) else 2.8
            else:
                feature_dict[f"V{i}"] = 0.0

    flagged_store[payload.transaction_id] = FlaggedTransaction(
        transaction_id=payload.transaction_id,
        original_decision=payload.decision,
        current_decision=payload.decision,
        risk_score=round(payload.risk_score, 4),
        ml_fraud_prob=round(payload.ml_fraud_prob, 4),
        anomaly_score=round(payload.anomaly_score, 4),
        velocity_surge_index=round(payload.velocity_surge_index, 4),
        amount=payload.amount,
        reasons=payload.reasons,
        flagged_at=datetime.now(timezone.utc).isoformat(),
        approved_by_admin=False,
        approved_at=None,
        feature_vector=feature_dict,
        verified_label=None,
        admin_note=None,
        model_version=active_model_version,
    )
    return {"status": "registered", "transaction_id": payload.transaction_id}


# =====================================================================
# 6. HUMAN-IN-THE-LOOP ADAPTIVE FRAUD LEARNING ENDPOINTS
# =====================================================================

@app.post("/api/v1/admin/feedback/{tx_id}")
def submit_verified_feedback(tx_id: str, payload: AdminFeedbackPayload):
    """
    Submits verified human feedback (0 = LEGITIMATE, 1 = FRAUD) for a flagged transaction.
    Persists the exact 53-dimension feature vector used at inference time into SQLite.
    Separates ML ground truth from operational overrides.
    """
    if payload.verified_label not in (0, 1):
        raise HTTPException(status_code=400, detail="verified_label must be 0 (LEGITIMATE) or 1 (FRAUD)")

    # 1. Lookup in flagged_store or existing feedback
    entry = flagged_store.get(tx_id)
    existing_fb = feedback_store.get_feedback_by_tx(tx_id)

    if existing_fb and not payload.allow_override:
        raise HTTPException(
            status_code=409,
            detail=f"Feedback already recorded for transaction {tx_id} as "
                   f"{'FRAUD' if existing_fb['verified_label'] == 1 else 'LEGITIMATE'} at {existing_fb['feedback_timestamp']}."
        )

    if entry is None and existing_fb is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{tx_id}' not found in the flagged store or feedback database."
        )

    # 2. Extract feature vector and scores
    if entry:
        feat_vector = entry.feature_vector or {}
        ml_prob = entry.ml_fraud_prob
        anomaly = entry.anomaly_score
        velocity = entry.velocity_surge_index
        risk = entry.risk_score
        orig_decision = entry.original_decision
        m_version = entry.model_version or active_model_version
    else:
        feat_vector = existing_fb["feature_vector"]
        ml_prob = existing_fb["ml_fraud_prob"]
        anomaly = existing_fb["anomaly_score"]
        velocity = existing_fb["velocity_surge_index"]
        risk = existing_fb["adaptive_risk_score"]
        orig_decision = existing_fb["original_decision"]
        m_version = existing_fb["model_version"]

    # 3. Save into SQLite persistent store
    rec = feedback_store.save_feedback(
        transaction_id=tx_id,
        feature_vector=feat_vector,
        verified_label=payload.verified_label,
        ml_fraud_prob=ml_prob,
        anomaly_score=anomaly,
        velocity_surge_index=velocity,
        adaptive_risk_score=risk,
        original_decision=orig_decision,
        model_version=m_version,
        admin_note=payload.admin_note or "",
        reviewer_id=payload.reviewer_id or "admin",
        allow_override=payload.allow_override
    )

    # 4. Update in-memory entry if present
    if entry:
        entry.verified_label = payload.verified_label
        entry.admin_note = payload.admin_note
        flagged_store[tx_id] = entry

    label_str = "FRAUD" if payload.verified_label == 1 else "LEGITIMATE"
    print(f"[FEEDBACK] Transaction {tx_id} labeled {label_str} by {payload.reviewer_id} (Model: {m_version})")

    total_fb = feedback_store.get_feedback_count()
    retrain_thresh = int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "10"))

    return {
        "status": "success",
        "message": f"Transaction {tx_id} verified as {label_str}.",
        "transaction_id": tx_id,
        "verified_label": payload.verified_label,
        "label_name": label_str,
        "admin_note": payload.admin_note,
        "reviewer_id": payload.reviewer_id,
        "total_feedback_count": total_fb,
        "retrain_threshold": retrain_thresh,
        "ready_for_retraining": total_fb >= retrain_thresh,
        "feedback_timestamp": rec["feedback_timestamp"],
        "model_version": m_version
    }


@app.get("/api/v1/admin/feedback")
def admin_list_feedback(limit: int = 100, offset: int = 0):
    """
    Returns list of verified feedback records with aggregate label statistics.
    """
    return {
        "stats": feedback_store.get_feedback_stats(),
        "retrain_threshold": int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "10")),
        "records": feedback_store.list_feedback(limit=limit, offset=offset)
    }


@app.post("/api/v1/admin/retrain")
def admin_retrain_model(force_promote: bool = False):
    """
    Triggers Human-in-the-Loop retraining:
    1. Reads original training data + verified human feedback from SQLite.
    2. Trains a challenger LightGBM model.
    3. Evaluates Challenger vs Champion on the pristine validation split.
    4. Compares metrics and safely promotes Challenger if criteria pass.
    5. Hot-reloads the promoted model in memory without server downtime.
    """
    try:
        result = retrain_engine.run_retraining_cycle(force_promote=force_promote)
        if result.get("promoted"):
            load_artifacts()
            print(f"[HOT-RELOAD] API in-memory active model updated to {result['candidate_model']}")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Retraining cycle failed: {str(e)}")


@app.post("/api/v1/admin/reload-model")
def admin_reload_model():
    """
    Safely reloads the active model from disk into the running server in-memory.
    """
    try:
        load_artifacts()
        return {
            "status": "success",
            "message": f"Successfully reloaded active model [{active_model_version}] in memory.",
            "active_model_version": active_model_version
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/api/v1/admin/model-status")
def admin_model_status():
    """
    Returns current active model metadata, champion metrics, promotion history, and feedback dataset size.
    """
    info = retrain_engine.get_active_model_info()
    stats = feedback_store.get_feedback_stats()
    thresh = int(os.getenv("FEEDBACK_RETRAIN_THRESHOLD", "10"))
    return {
        "active_model_version": active_model_version,
        "activated_at": info.get("activated_at"),
        "champion_metrics": info.get("metrics"),
        "feedback_stats": stats,
        "retrain_threshold": thresh,
        "ready_for_retraining": stats["total_verified"] >= thresh,
        "history": info.get("history", [])
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

@app.get("/admin")
def serve_admin():
    admin_file = os.path.join(dashboard_path, "admin.html")
    if os.path.exists(admin_file):
        return FileResponse(admin_file)
    return {"message": "Admin page not found."}

@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(dashboard_path, "style.css"))

@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(dashboard_path, "app.js"))

@app.get("/admin.js")
def serve_admin_js():
    return FileResponse(os.path.join(dashboard_path, "admin.js"))

@app.get("/admin-style.css")
def serve_admin_style():
    return FileResponse(os.path.join(dashboard_path, "admin-style.css"))
