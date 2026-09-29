"""
feedback_store.py - Persistent SQLite storage for Human-in-the-Loop verified feedback.
Securing the Surge: Protecting Digital Transactions During Peak E-Commerce Events
"""

import os
import json
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = os.getenv("FEEDBACK_DB_PATH", "feedback.db")


class FeedbackStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("FEEDBACK_DB_PATH", DEFAULT_DB_PATH)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes the SQLite feedback_records table if it doesn't exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    timestamp REAL,
                    feature_vector TEXT NOT NULL,
                    ml_fraud_prob REAL,
                    anomaly_score REAL,
                    velocity_surge_index REAL,
                    adaptive_risk_score REAL,
                    original_decision TEXT,
                    verified_label INTEGER NOT NULL CHECK (verified_label IN (0, 1)),
                    admin_note TEXT,
                    reviewer_id TEXT DEFAULT 'admin',
                    feedback_timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_tx_id 
                ON feedback_records (transaction_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_timestamp 
                ON feedback_records (feedback_timestamp)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_feedback(
        self,
        transaction_id: str,
        feature_vector: Dict[str, float],
        verified_label: int,
        ml_fraud_prob: float,
        anomaly_score: float,
        velocity_surge_index: float,
        adaptive_risk_score: float,
        original_decision: str,
        model_version: str,
        admin_note: str = "",
        reviewer_id: str = "admin",
        timestamp: Optional[float] = None,
        allow_override: bool = False
    ) -> Dict[str, Any]:
        """
        Persists a verified human feedback record into SQLite.
        verified_label: 0 for LEGITIMATE, 1 for FRAUD.
        Stores full 53-dimension feature vector as JSON to guarantee zero feature leakage.
        """
        if verified_label not in (0, 1):
            raise ValueError(f"verified_label must be 0 (LEGITIMATE) or 1 (FRAUD), got {verified_label}")

        now_iso = datetime.now(timezone.utc).isoformat()
        feat_json = json.dumps(feature_vector)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if allow_override:
                cursor.execute("""
                    INSERT INTO feedback_records (
                        transaction_id, timestamp, feature_vector, ml_fraud_prob,
                        anomaly_score, velocity_surge_index, adaptive_risk_score,
                        original_decision, verified_label, admin_note, reviewer_id,
                        feedback_timestamp, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(transaction_id) DO UPDATE SET
                        verified_label = excluded.verified_label,
                        admin_note = excluded.admin_note,
                        reviewer_id = excluded.reviewer_id,
                        feedback_timestamp = excluded.feedback_timestamp,
                        model_version = excluded.model_version
                """, (
                    transaction_id, timestamp or 0.0, feat_json, ml_fraud_prob,
                    anomaly_score, velocity_surge_index, adaptive_risk_score,
                    original_decision, verified_label, admin_note, reviewer_id,
                    now_iso, model_version
                ))
            else:
                cursor.execute("""
                    INSERT INTO feedback_records (
                        transaction_id, timestamp, feature_vector, ml_fraud_prob,
                        anomaly_score, velocity_surge_index, adaptive_risk_score,
                        original_decision, verified_label, admin_note, reviewer_id,
                        feedback_timestamp, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction_id, timestamp or 0.0, feat_json, ml_fraud_prob,
                    anomaly_score, velocity_surge_index, adaptive_risk_score,
                    original_decision, verified_label, admin_note, reviewer_id,
                    now_iso, model_version
                ))
            conn.commit()
        finally:
            conn.close()

        return self.get_feedback_by_tx(transaction_id)

    def get_feedback_by_tx(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a feedback record by transaction_id."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM feedback_records WHERE transaction_id = ?",
                (transaction_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["feature_vector"] = json.loads(res["feature_vector"])
            except Exception:
                pass
            return res
        finally:
            conn.close()

    def get_feedback_count(self) -> int:
        """Returns the total number of feedback records stored."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback_records")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Returns aggregate metrics on human feedback labels."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback_records")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback_records WHERE verified_label = 0")
            legitimate = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM feedback_records WHERE verified_label = 1")
            fraud = cursor.fetchone()[0]

            cursor.execute("SELECT model_version, COUNT(*) FROM feedback_records GROUP BY model_version")
            by_version = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_verified": total,
                "verified_legitimate": legitimate,
                "verified_fraud": fraud,
                "by_model_version": by_version
            }
        finally:
            conn.close()

    def list_feedback(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Returns paginated feedback records ordered newest first."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, transaction_id, timestamp, ml_fraud_prob, anomaly_score,
                       velocity_surge_index, adaptive_risk_score, original_decision,
                       verified_label, admin_note, reviewer_id, feedback_timestamp,
                       model_version
                FROM feedback_records
                ORDER BY id DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_training_records(self) -> List[Dict[str, Any]]:
        """
        Returns all feedback records with deserialized feature vectors for retraining.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT transaction_id, feature_vector, verified_label,
                       feedback_timestamp, model_version
                FROM feedback_records
                ORDER BY id ASC
            """)
            records = []
            for row in cursor.fetchall():
                try:
                    feat_dict = json.loads(row["feature_vector"])
                    records.append({
                        "transaction_id": row["transaction_id"],
                        "feature_vector": feat_dict,
                        "verified_label": row["verified_label"],
                        "feedback_timestamp": row["feedback_timestamp"],
                        "model_version": row["model_version"]
                    })
                except Exception as e:
                    print(f"Warning: Failed to parse feature vector for {row['transaction_id']}: {e}")
            return records
        finally:
            conn.close()


# Singleton instance for import
feedback_store = FeedbackStore()
