#!/usr/bin/env python3
"""End-to-End Smoke Test for CDM-1.

1. Starts API service / initializes test environment.
2. Creates a tenant API key.
3. Submits an operational task through POST /v1/agent/runs.
4. Observes tool calls, policy gate evaluations, and state transitions.
5. Verifies decision records and HMAC provenance cryptographic signatures in database/audit log.
6. Prints PASS or FAIL with exit code 0 or 1.
"""

import sys
import os
import json
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app, KEY_MANAGER
from cdm.execution.audit import AuditTrailManager
from cdm.inference.provenance import PredictionProvenance


def main():
    print("=" * 70)
    print("CDM-1 END-TO-END VERIFICATION SMOKE TEST")
    print("=" * 70)

    api_url = os.environ.get("CDM_API_URL")
    if api_url:
        import requests
        class HttpClientWrapper:
            def __init__(self, base_url):
                self.base_url = base_url.rstrip("/")
            def get(self, path, **kwargs):
                return requests.get(f"{self.base_url}{path}", **kwargs)
            def post(self, path, **kwargs):
                return requests.post(f"{self.base_url}{path}", **kwargs)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        client_context = HttpClientWrapper(api_url)
    else:
        client_context = TestClient(app)

    success = True
    with client_context as client:
        # 1. Readiness probe check
        print("\n[Step 1] Checking service readiness probe (/v1/ready)...")
        ready_resp = client.get("/v1/ready")
        if ready_resp.status_code == 200:
            ready_data = ready_resp.json()
            print(f"    [+] /v1/ready status: {ready_data.get('status')} (mode={ready_data.get('mode')})")
        else:
            print(f"    [-] /v1/ready returned HTTP {ready_resp.status_code}: {ready_resp.text}")
            success = False

        # 2. Key Generation
        print("\n[Step 2] Generating Tenant API Key via PersistentApiKeyManager...")
        tenant_id = "tenant_smoke_test"
        raw_key, key_hash = KEY_MANAGER.generate_key(tenant_id=tenant_id, tier="developer")
        print(f"    [+] Generated API Key: {raw_key[:12]}... (hash={key_hash[:16]}...)")
        headers = {"Authorization": f"Bearer {raw_key}"}

        # 3. Direct Decision Call
        print("\n[Step 3] Submitting operational decision request to POST /v1/decisions...")
        dec_payload = {
            "text": "Customer order #8841 arrived with damaged display. Customer requests replacement authorization.",
            "structured_features": {
                "monetary_value": 45.0,
                "account_tenure_months": 14,
                "urgency_level": "medium",
                "domain": "customer_support",
            },
            "task": "binary",
        }
        dec_resp = client.post("/v1/decisions", headers=headers, json=dec_payload)
        if dec_resp.status_code == 200:
            dec_data = dec_resp.json()
            print(f"    [+] Decision ID: {dec_data.get('decision_id')}")
            print(f"    [+] Decision:    {dec_data.get('decision')} (conf={dec_data.get('confidence')})")
            print(f"    [+] Score:       {dec_data.get('score')} | Risk: {dec_data.get('risk')}")
        else:
            print(f"    [-] POST /v1/decisions failed ({dec_resp.status_code}): {dec_resp.text}")
            success = False

        # 4. Agent Control Loop Run
        print("\n[Step 4] Submitting multi-turn agent task to POST /v1/agent/runs...")
        run_payload = {
            "goal": "Verify damaged return and issue replacement voucher",
            "task_description": "Validate support ticket RMA-9912 and generate dispatch confirmation.",
            "step_budget": 5,
            "cost_budget_usd": 1.0,
            "tenant_policies": ["max_refund_usd: 100.0"],
            "auto_run": True,
        }
        run_resp = client.post("/v1/agent/runs", headers=headers, json=run_payload)
        if run_resp.status_code == 200:
            run_data = run_resp.json()
            run_id = run_data["run_id"]
            print(f"    [+] Agent Run ID:     {run_id}")
            print(f"    [+] Final Status:     {run_data.get('status')}")
            print(f"    [+] Steps Executed:   {run_data.get('step_count')}")
            print(f"    [+] Accumulated Cost: ${run_data.get('accumulated_cost_usd'):.4f}")

            # Check history for tool calls and policy gate evaluations
            history = run_data.get("history", [])
            print(f"    [+] Recorded Steps in Run History: {len(history)}")
            for step in history:
                print(f"        - Step {step.get('step')}: Action='{step.get('action')}' | Gate='{step.get('gate_decision')}' | Success={step.get('success')}")
        else:
            print(f"    [-] POST /v1/agent/runs failed ({run_resp.status_code}): {run_resp.text}")
            success = False
            run_id = None

        # 5. Database & HMAC Provenance Verification
        if run_id:
            print("\n[Step 5] Cryptographic HMAC Provenance Verification on Audit Trail...")
            audit_dir = os.environ.get("CDM_AUDIT_DIR", "experiments/agent_runs")
            audit_mgr = AuditTrailManager(audit_dir=audit_dir)
            valid, issues = audit_mgr.verify_run_audit(run_id)
            if valid:
                print(f"    [+] HMAC Signature Verification: PASSED (All step signatures verified)")
            else:
                print(f"    [-] HMAC Signature Verification: FAILED (Issues: {issues})")
                success = False

            print("\n[Step 6] Full Provenance Chain:")
            history_records = audit_mgr.get_run_history(run_id)
            for record in history_records:
                prov = record.provenance_record
                print(f"    --- Step {record.step_number} Provenance ---")
                print(f"        Execution ID:   {prov.get('execution_id')}")
                print(f"        Tenant ID:      {prov.get('tenant_id')}")
                print(f"        Key ID:         {prov.get('key_id')}")
                print(f"        Model:          {prov.get('model_id')} (v{prov.get('model_version')})")
                print(f"        Payload Hash:   {prov.get('prediction_digest')}")
                print(f"        HMAC Signature: {prov.get('provenance_signature')}")
                print(f"        Timestamp:      {prov.get('timestamp')}")

    print("\n" + "=" * 70)
    if success:
        print("RESULT: PASS — All end-to-end smoke tests and HMAC verifications succeeded.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("RESULT: FAIL — One or more smoke test checks failed.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
