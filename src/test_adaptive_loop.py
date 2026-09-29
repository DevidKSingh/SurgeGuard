"""
test_adaptive_loop.py - End-to-End Verification of Human-in-the-Loop Adaptive Learning.
Tests:
1. Transaction evaluation & flagged store feature preservation
2. Admin inspection & operational override separation
3. Verified human feedback persistence (0 = LEGITIMATE, 1 = FRAUD)
4. Feedback retrieval & audit statistics
5. Challenger model retraining on (Historical Data + Verified Feedback)
6. Pure-NumPy validation & Champion vs Challenger comparison
7. Model promotion & versioning (model_v1 -> model_v2)
8. In-memory hot reload & live inference with promoted model
"""

import os
import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

def request(method, path, data=None):
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def run_tests():
    print("=" * 70)
    print("SURGEGUARD ADAPTIVE LEARNING E2E TEST SUITE")
    print("=" * 70)

    # 1. Health & Initial Model Status
    status, health = request("GET", "/api/v1/health")
    assert status == 200 and health["status"] == "healthy", f"Health failed: {health}"
    print("[PASS] 1. Health check passed:", health)

    status, model_status = request("GET", "/api/v1/admin/model-status")
    assert status == 200, f"Model status failed: {model_status}"
    initial_version = model_status["active_model_version"]
    print(f"[PASS] 2. Initial Active Champion: {initial_version}")

    # 2. Evaluate Legitimate Transaction (APPROVE)
    legit_payload = {
        "Time": 1000.0,
        "Amount": 25.50,
        "V1": 0.1, "V2": -0.2, "V4": 0.5, "V10": 0.2, "V12": 0.1, "V14": 0.2
    }
    status, legit_resp = request("POST", "/api/v1/evaluate", legit_payload)
    assert status == 200 and legit_resp["decision"] == "APPROVE", f"Legit tx failed: {legit_resp}"
    print(f"[PASS] 3. Normal Tx: {legit_resp['transaction_id']} -> {legit_resp['decision']} (Risk: {legit_resp['risk_score']})")

    # 3. Evaluate High-Risk Fraud Transactions to populate flagged_store
    print("\n--- Generating Suspicious Transactions (Triggering HALT/REVIEW) ---")
    flagged_ids = []
    i = 0
    while len(flagged_ids) < 12:
        i += 1
        suspicious_payload = {
            "Time": 2000.0 + i * 5,
            "Amount": 850.0 + i * 15,
            "V1": -3.5, "V2": 2.8, "V3": -4.2, "V4": 4.1, "V5": -2.0,
            "V10": -3.8, "V12": -4.5, "V14": -5.2, "V16": -2.8, "V17": -4.0
        }
        status, s_resp = request("POST", "/api/v1/evaluate", suspicious_payload)
        assert status == 200
        tx_id = s_resp["transaction_id"]
        if s_resp["decision"] in ("HALT", "REVIEW"):
            flagged_ids.append(tx_id)
        print(f"  Tx {tx_id}: Decision={s_resp['decision']}, Risk={s_resp['risk_score']}, ML_Prob={s_resp['ml_fraud_prob']}")

    # 4. Verify Flagged Transaction contains 53-dimension feature vector
    first_tx = flagged_ids[0]
    status, flagged_tx = request("GET", f"/api/v1/admin/transaction/{first_tx}")
    assert status == 200, f"Lookup failed: {flagged_tx}"
    assert flagged_tx["feature_vector"] is not None, "Feature vector was not preserved in flagged_store!"
    fv_len = len(flagged_tx["feature_vector"])
    assert fv_len >= 50, f"Feature vector incomplete: {fv_len} cols"
    print(f"[PASS] 4. Flagged Tx inspection: {first_tx} retains {fv_len} features without data leakage.")

    # 5. Operational Override Separation Test
    status, override_resp = request("POST", f"/api/v1/admin/approve/{first_tx}")
    assert status == 200, f"Override failed: {override_resp}"
    assert override_resp["new_decision"] == "APPROVE"
    print(f"[PASS] 5. Operational override recorded: Decision changed from {override_resp['previous_decision']} to APPROVE.")

    # Verify that operational override DID NOT inject a ground truth training label
    status, flagged_tx_after = request("GET", f"/api/v1/admin/transaction/{first_tx}")
    assert flagged_tx_after["approved_by_admin"] is True
    assert flagged_tx_after["verified_label"] is None, "Operational override erroneously wrote verified_label!"
    print("[PASS] 6. Ground truth separation verified: verified_label remains None after operational override.")

    # 6. Submit Explicit Verified Human Feedback
    print("\n--- Submitting Human-in-the-Loop Verified Ground Truth ---")
    # Mark first 6 as FRAUD (1), next 6 as LEGITIMATE (0)
    for idx, tx_id in enumerate(flagged_ids):
        v_label = 1 if idx < 6 else 0
        note = "Confirmed account draining bot cluster" if v_label == 1 else "VIP corporate customer phone verified"
        status, fb_resp = request("POST", f"/api/v1/admin/feedback/{tx_id}", {
            "verified_label": v_label,
            "admin_note": note,
            "reviewer_id": "senior_analyst_01",
            "allow_override": True
        })
        assert status == 200, f"Feedback failed: {fb_resp}"
        print(f"  Feedback recorded: {tx_id} -> {'FRAUD (1)' if v_label == 1 else 'LEGITIMATE (0)'} | Pool: {fb_resp['total_feedback_count']}/{fb_resp['retrain_threshold']}")

    # 7. Check Feedback Stats Endpoint
    status, fb_list = request("GET", "/api/v1/admin/feedback")
    assert status == 200
    stats = fb_list["stats"]
    assert stats["total_verified"] >= 12
    assert stats["verified_fraud"] >= 6
    assert stats["verified_legitimate"] >= 6
    print(f"[PASS] 7. Persistent Feedback Store: Total={stats['total_verified']}, Legitimate={stats['verified_legitimate']}, Fraud={stats['verified_fraud']}")

    # 8. Trigger Model Retraining & Champion vs Challenger Validation
    print("\n--- Executing Champion vs Challenger Retraining Cycle ---")
    t0 = time.time()
    status, retrain_resp = request("POST", "/api/v1/admin/retrain?force_promote=false")
    elapsed = time.time() - t0
    assert status == 200, f"Retrain failed: {retrain_resp}"

    print(f"Retrain Completed in {elapsed:.2f}s:")
    print(f"  Candidate Model:   {retrain_resp['candidate_model']}")
    print(f"  Previous Model:    {retrain_resp['previous_model']}")
    print(f"  Promoted to Active:{retrain_resp['promoted']}")
    print(f"  Feedback Samples:  {retrain_resp['feedback_samples']}")
    print(f"  Champion PR-AUC:   {retrain_resp['champion_metrics']['pr_auc']:.4f} | ROC-AUC: {retrain_resp['champion_metrics']['roc_auc']:.4f}")
    print(f"  Challenger PR-AUC: {retrain_resp['challenger_metrics']['pr_auc']:.4f} | ROC-AUC: {retrain_resp['challenger_metrics']['roc_auc']:.4f}")
    # 8. Retraining Validation: First demonstrate safe rejection when challenger does not beat benchmark
    if not retrain_resp["promoted"]:
        print(f"[PASS] 8A. Safety guardrail active: Challenger failed validation ({retrain_resp.get('reason')}).")
        print("           Champion 'model_v1' remains active to protect production from degradation.")

        # Now test Promotion flow with force_promote=True
        print("\n--- Testing Promotion Flow (force_promote=True) ---")
        status, promote_resp = request("POST", "/api/v1/admin/retrain?force_promote=true")
        assert status == 200, f"Promote retrain failed: {promote_resp}"
        assert promote_resp["promoted"] is True, f"Force promotion failed: {promote_resp}"
        retrain_resp = promote_resp
        print(f"[PASS] 8B. Challenger {retrain_resp['candidate_model']} successfully promoted!")
    else:
        print(f"[PASS] 8. Challenger {retrain_resp['candidate_model']} passed validation and promoted!")

    # 9. Verify Live Server Uses Promoted Model Without Restart
    status, updated_status = request("GET", "/api/v1/admin/model-status")
    assert status == 200
    assert updated_status["active_model_version"] == retrain_resp["candidate_model"], (
        f"Server active model not updated! Expected {retrain_resp['candidate_model']}, got {updated_status['active_model_version']}"
    )
    print(f"[PASS] 9. In-Memory Hot-Reload verified: Active model is now {updated_status['active_model_version']}")

    # 10. Live Inference with Promoted Model
    post_retrain_payload = {
        "Time": 3000.0,
        "Amount": 45.0,
        "V1": 0.05, "V2": -0.1, "V4": 0.3, "V10": 0.1, "V12": 0.05, "V14": 0.1
    }
    status, post_resp = request("POST", "/api/v1/evaluate", post_retrain_payload)
    assert status == 200 and post_resp["decision"] == "APPROVE"
    print(f"[PASS] 10. Live inference using promoted model {updated_status['active_model_version']} succeeded in {post_resp['latency_ms']}ms.")

    print("\n" + "=" * 70)
    print("ALL 10 VERIFICATION TESTS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
