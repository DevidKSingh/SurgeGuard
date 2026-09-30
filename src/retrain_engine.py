"""
retrain_engine.py - Retraining, Validation, and Champion/Challenger Model Management.
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import os
import sys
import time
import json
import types
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

from feedback_store import feedback_store

load_dotenv()

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))

_env_model_dir = os.getenv("MODEL_DIR")
if _env_model_dir and os.path.isabs(_env_model_dir):
    MODEL_DIR = _env_model_dir
elif _env_model_dir and os.path.exists(_env_model_dir):
    MODEL_DIR = os.path.abspath(_env_model_dir)
elif os.path.exists(os.path.join(PROJECT_DIR, "models")):
    MODEL_DIR = os.path.join(PROJECT_DIR, "models")
elif os.path.exists("models"):
    MODEL_DIR = os.path.abspath("models")
else:
    MODEL_DIR = os.path.join(PROJECT_DIR, "models")

ACTIVE_MODEL_JSON = os.path.join(MODEL_DIR, "active_model.json")
MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", "0.75"))
MIN_PR_AUC = float(os.getenv("MIN_PR_AUC", "0.40"))
MODEL_AUTO_PROMOTION = os.getenv("MODEL_AUTO_PROMOTION", "true").lower() == "true"


# ── Pure-NumPy Metric Functions (Zero external dependency & AppControl safe) ──

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    npos = y_true_s.sum()
    nneg = len(y_true_s) - npos
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    tpr = np.concatenate([[0.0], tps / max(npos, 1)])
    fpr = np.concatenate([[0.0], fps / max(nneg, 1)])
    return float(np.trapezoid(tpr, fpr))


def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    npos = y_true_s.sum()
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    precision = tps / (tps + fps)
    recall = tps / max(npos, 1)
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum(precision * (recall - recall_prev)))


def precision_recall_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    desc_idx = np.argsort(y_score)[::-1]
    y_true_s = y_true[desc_idx]
    thresholds = y_score[desc_idx]
    npos = y_true_s.sum()
    tps = np.cumsum(y_true_s)
    fps = np.cumsum(1 - y_true_s)
    precision = np.concatenate([tps / (tps + fps), [1.0]])
    recall    = np.concatenate([tps / max(npos, 1), [0.0]])
    return precision, recall, thresholds


class IsolationForest:
    """Lightweight numpy Z-score anomaly detector matching IsolationForest API."""
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


def ensure_pickle_shim():
    """Ensures IsolationForest can be deserialized across modules and workers without side-effects."""
    for mod_name in ("__main__", "__mp_main__", "ml_pipeline"):
        shim = sys.modules.get(mod_name)
        if shim is None or not hasattr(shim, "IsolationForest"):
            shim = types.ModuleType(mod_name)
            shim.IsolationForest = IsolationForest
            sys.modules[mod_name] = shim
        else:
            shim.IsolationForest = IsolationForest


def resolve_model_artifact_path(artifact_path: Optional[str] = None) -> str:
    """Robustly resolves the full path to a model artifact across directory structures."""
    candidates = []
    if artifact_path:
        candidates.extend([
            artifact_path,
            os.path.join(MODEL_DIR, artifact_path),
            os.path.join(MODEL_DIR, os.path.basename(artifact_path)),
            os.path.join(PROJECT_DIR, "models", os.path.basename(artifact_path)),
            os.path.join(PROJECT_DIR, artifact_path),
            os.path.join(os.getcwd(), artifact_path),
            os.path.join(os.getcwd(), "models", os.path.basename(artifact_path)),
        ])
    candidates.extend([
        os.path.join(MODEL_DIR, "risk_engine_artifacts.pkl"),
        os.path.join(PROJECT_DIR, "models", "risk_engine_artifacts.pkl"),
        os.path.join(PROJECT_DIR, "models", "model_v3.pkl"),
        "models/risk_engine_artifacts.pkl",
        "risk_engine_artifacts.pkl",
    ])
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Model artifact '{artifact_path}' not found in candidate paths: {candidates}")


def resolve_train_data_path() -> str:
    """Robustly locates train_cleaned.csv."""
    candidates = [
        os.path.join(PROJECT_DIR, "train_cleaned.csv"),
        "train_cleaned.csv",
        os.path.join(os.getcwd(), "train_cleaned.csv"),
        os.path.join(PROJECT_DIR, "..", "train_cleaned.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"train_cleaned.csv not found in candidate paths: {candidates}")


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes causal features strictly using past information (< current transaction time).
    Exact identical implementation to ml_pipeline.py to guarantee zero feature definition divergence.
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
    
    windows = [10, 60, 300, 900]
    for w in windows:
        left_idx = np.searchsorted(times, times - w, side="left")
        count = indices - left_idx
        amt_sum = amount_cumsum[indices] - amount_cumsum[left_idx]
        amt_mean = np.divide(amt_sum, count, out=np.zeros_like(amt_sum), where=count > 0)
        
        df[f"Tx_Count_{w}s"] = count
        df[f"Amt_Sum_{w}s"] = amt_sum
        df[f"Amt_Mean_{w}s"] = amt_mean
        
        amt_ratio = np.divide(amounts, amt_mean, out=np.ones_like(amounts), where=amt_mean > 0)
        df[f"Amt_Ratio_{w}s"] = amt_ratio
    
    df["V14_V4_Ratio"] = df["V14"] / (np.abs(df["V4"]) + 1e-5)
    df["V12_V10_Diff"] = df["V12"] - df["V10"]
    
    return df


def get_active_model_info() -> Dict[str, Any]:
    """Reads active model metadata from active_model.json."""
    if os.path.exists(ACTIVE_MODEL_JSON):
        try:
            with open(ACTIVE_MODEL_JSON, "r") as f:
                info = json.load(f)
                if "artifact_path" in info:
                    try:
                        info["artifact_path"] = resolve_model_artifact_path(info["artifact_path"])
                    except Exception:
                        pass
                return info
        except Exception as e:
            print(f"Warning reading {ACTIVE_MODEL_JSON}: {e}")
    
    # Fallback default
    try:
        resolved_default = resolve_model_artifact_path("risk_engine_artifacts.pkl")
    except Exception:
        resolved_default = os.path.join(MODEL_DIR, "risk_engine_artifacts.pkl")

    return {
        "active_model_version": "model_v1",
        "artifact_path": resolved_default,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {"roc_auc": 0.9667, "pr_auc": 0.6295},
        "feedback_samples_used": 0,
        "history": []
    }


def load_model_artifact(artifact_path: str) -> Dict[str, Any]:
    """Loads a serialized model artifact dictionary."""
    ensure_pickle_shim()
    resolved_path = resolve_model_artifact_path(artifact_path)
    with open(resolved_path, "rb") as f:
        return pickle.load(f)


# Cache for engineered training dataframe to avoid re-reading 100MB CSV repeatedly
_CACHED_TRAIN_FEATS: Optional[pd.DataFrame] = None

def get_base_training_data() -> Tuple[pd.DataFrame, pd.DataFrame, list, list]:
    """
    Loads train_cleaned.csv and extracts features with chronological 80/20 split.
    Uses memory cache so subsequent retraining calls run in seconds.
    """
    global _CACHED_TRAIN_FEATS
    if _CACHED_TRAIN_FEATS is None:
        data_path = resolve_train_data_path()
        print(f"[RETRAIN] Loading base training data from {data_path}...")
        train_df = pd.read_csv(data_path).sort_values("Time").reset_index(drop=True)
        _CACHED_TRAIN_FEATS = extract_features(train_df)
    
    split_idx = int(len(_CACHED_TRAIN_FEATS) * 0.8)
    train_split = _CACHED_TRAIN_FEATS.iloc[:split_idx].copy()
    val_split = _CACHED_TRAIN_FEATS.iloc[split_idx:].copy()
    
    drop_cols = ["Class"]
    feature_cols = [c for c in _CACHED_TRAIN_FEATS.columns if c not in drop_cols]
    v_cols = [f"V{i}" for i in range(1, 29)]
    
    return train_split, val_split, feature_cols, v_cols


def run_retraining_cycle(force_promote: bool = False) -> Dict[str, Any]:
    """
    Executes the complete Human-in-the-Loop Adaptive Retraining cycle:
    1. Loads base historical training data + pristine validation split.
    2. Reads accumulated verified human feedback from SQLite.
    3. Merges feedback feature vectors into training split (Zero Leakage).
    4. Evaluates Active Champion model on validation split.
    5. Trains Challenger LightGBM model on enriched training data.
    6. Evaluates Challenger model on validation split.
    7. Compares Champion vs Challenger metrics.
    8. If Challenger satisfies validation criteria, promotes it and updates active_model.json.
    """
    t0 = time.time()
    ensure_pickle_shim()
    
    # 1. Active model info
    active_info = get_active_model_info()
    current_champion_version = active_info.get("active_model_version", "model_v1")
    current_artifact_path = active_info.get("artifact_path", "risk_engine_artifacts.pkl")
    
    print(f"\n[RETRAIN] Current Active Champion: {current_champion_version}")
    
    # 2. Load historical splits
    train_split, val_split, feature_cols, v_cols = get_base_training_data()
    X_tr = train_split[feature_cols].copy()
    y_tr = train_split["Class"].copy()
    X_val = val_split[feature_cols].copy()
    y_val = val_split["Class"].copy()
    
    # 3. Read accumulated human feedback
    feedback_records = feedback_store.get_training_records()
    num_feedback = len(feedback_records)
    print(f"[RETRAIN] Collected {num_feedback} verified feedback records from database.")
    
    # Merge feedback into training split
    if num_feedback > 0:
        feedback_rows = []
        feedback_labels = []
        for rec in feedback_records:
            fv = rec["feature_vector"]
            # Extract in exact feature_cols order with default 0.0
            row = [float(fv.get(col, 0.0)) for col in feature_cols]
            feedback_rows.append(row)
            feedback_labels.append(int(rec["verified_label"]))
        
        fb_df = pd.DataFrame(feedback_rows, columns=feature_cols)
        fb_series = pd.Series(feedback_labels, name="Class")
        
        X_tr = pd.concat([X_tr, fb_df], ignore_index=True)
        y_tr = pd.concat([y_tr, fb_series], ignore_index=True)
        print(f"[RETRAIN] Enriched training set: {len(X_tr):,} samples ({num_feedback} from human feedback).")
    else:
        print("[RETRAIN] No feedback records available; training on baseline data.")

    # 4. Evaluate Champion on validation set
    champion_artifact = load_model_artifact(current_artifact_path)
    champion_model = champion_artifact["lgb_model"]
    champion_preds = champion_model.predict(X_val)
    champion_roc_auc = roc_auc_score(y_val, champion_preds)
    champion_pr_auc = average_precision_score(y_val, champion_preds)
    
    print(f"[VALIDATION] Champion ({current_champion_version}) — ROC-AUC: {champion_roc_auc:.4f}, PR-AUC: {champion_pr_auc:.4f}")
    
    # 5. Train Challenger LightGBM model
    next_ver_num = int(current_champion_version.replace("model_v", "").split("_")[0]) + 1
    challenger_version = f"model_v{next_ver_num}"
    print(f"[RETRAIN] Training challenger {challenger_version}...")
    
    neg_count = (y_tr == 0).sum()
    pos_count = (y_tr == 1).sum()
    scale_pos_weight = neg_count / max(1, pos_count)
    
    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "scale_pos_weight": scale_pos_weight * 0.25,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4
    }
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    challenger_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=400,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # 6. Evaluate Challenger
    challenger_preds = challenger_model.predict(X_val, num_iteration=challenger_model.best_iteration)
    challenger_roc_auc = roc_auc_score(y_val, challenger_preds)
    challenger_pr_auc = average_precision_score(y_val, challenger_preds)
    
    print(f"[VALIDATION] Challenger ({challenger_version}) — ROC-AUC: {challenger_roc_auc:.4f}, PR-AUC: {challenger_pr_auc:.4f}")
    
    # 7. Check validation criteria
    # Criteria:
    # 1. Challenger ROC-AUC >= MIN_ROC_AUC
    # 2. Challenger PR-AUC >= MIN_PR_AUC
    # 3. Challenger PR-AUC does not suffer catastrophic regression vs Champion (>= Champion PR-AUC - 0.05)
    meets_min_roc = challenger_roc_auc >= MIN_ROC_AUC
    meets_min_pr = challenger_pr_auc >= MIN_PR_AUC
    not_regressed = challenger_pr_auc >= (champion_pr_auc - 0.05)
    
    criteria_passed = (meets_min_roc and meets_min_pr and not_regressed)
    should_promote = (criteria_passed and MODEL_AUTO_PROMOTION) or force_promote
    
    # Calibrate tiers for challenger
    precisions, recalls, thresholds = precision_recall_curve(y_val, challenger_preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    opt_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    tau_review = float(max(0.05, opt_threshold * 0.45))
    tau_halt = float(opt_threshold)
    
    # Keep baseline anomaly detector
    iso_forest = champion_artifact.get("iso_forest")
    
    challenger_artifact_payload = {
        "lgb_model": challenger_model,
        "iso_forest": iso_forest,
        "feature_cols": feature_cols,
        "v_cols": v_cols,
        "opt_threshold": opt_threshold,
        "tiers": {"approve_max": tau_review, "halt_min": tau_halt}
    }
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    candidate_artifact_file = os.path.join(MODEL_DIR, f"{challenger_version}.pkl")
    with open(candidate_artifact_file, "wb") as f:
        pickle.dump(challenger_artifact_payload, f)
    
    now_iso = datetime.now(timezone.utc).isoformat()
    duration_s = round(time.time() - t0, 2)
    
    history_entry = {
        "challenger_version": challenger_version,
        "champion_version": current_champion_version,
        "evaluated_at": now_iso,
        "champion_metrics": {
            "roc_auc": round(champion_roc_auc, 4),
            "pr_auc": round(champion_pr_auc, 4)
        },
        "challenger_metrics": {
            "roc_auc": round(challenger_roc_auc, 4),
            "pr_auc": round(challenger_pr_auc, 4)
        },
        "feedback_samples": num_feedback,
        "promoted": should_promote,
        "reason": "Passed all validation criteria" if criteria_passed else (
            f"Failed criteria (Challenger PR-AUC: {challenger_pr_auc:.4f}, Min: {MIN_PR_AUC})"
        )
    }
    
    history = active_info.get("history", [])
    history.append(history_entry)
    
    if should_promote:
        print(f"[PROMOTION] {challenger_version} promoted to active champion!")
        active_meta = {
            "active_model_version": challenger_version,
            "artifact_path": candidate_artifact_file,
            "activated_at": now_iso,
            "metrics": {
                "roc_auc": round(challenger_roc_auc, 4),
                "pr_auc": round(challenger_pr_auc, 4)
            },
            "previous_model": current_champion_version,
            "feedback_samples_used": num_feedback,
            "history": history
        }
        with open(ACTIVE_MODEL_JSON, "w") as f:
            json.dump(active_meta, f, indent=2)
        
        # Also update risk_engine_artifacts.pkl in MODEL_DIR for legacy/predict.py compatibility
        try:
            target_legacy = os.path.join(MODEL_DIR, "risk_engine_artifacts.pkl")
            with open(target_legacy, "wb") as f:
                pickle.dump(challenger_artifact_payload, f)
            if os.path.abspath(target_legacy) != os.path.abspath("risk_engine_artifacts.pkl"):
                try:
                    with open("risk_engine_artifacts.pkl", "wb") as f:
                        pickle.dump(challenger_artifact_payload, f)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning updating risk_engine_artifacts.pkl: {e}")
        
        return {
            "status": "success",
            "candidate_model": challenger_version,
            "promoted": True,
            "previous_model": current_champion_version,
            "champion_metrics": {
                "roc_auc": round(champion_roc_auc, 4),
                "pr_auc": round(champion_pr_auc, 4)
            },
            "challenger_metrics": {
                "roc_auc": round(challenger_roc_auc, 4),
                "pr_auc": round(challenger_pr_auc, 4)
            },
            "feedback_samples": num_feedback,
            "duration_seconds": duration_s,
            "activated_at": now_iso
        }
    else:
        print(f"[REJECTED] {challenger_version} failed validation. {current_champion_version} remains active.")
        active_info["history"] = history
        with open(ACTIVE_MODEL_JSON, "w") as f:
            json.dump(active_info, f, indent=2)
        
        return {
            "status": "success",
            "candidate_model": challenger_version,
            "promoted": False,
            "reason": history_entry["reason"],
            "previous_model": current_champion_version,
            "champion_metrics": {
                "roc_auc": round(champion_roc_auc, 4),
                "pr_auc": round(champion_pr_auc, 4)
            },
            "challenger_metrics": {
                "roc_auc": round(challenger_roc_auc, 4),
                "pr_auc": round(challenger_pr_auc, 4)
            },
            "feedback_samples": num_feedback,
            "duration_seconds": duration_s
        }
