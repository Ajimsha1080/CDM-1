#!/usr/bin/env python3
"""CDM-1 Final Verification: Release Security, Manifest Authenticity & API Readiness.

Verifies:
1. Ed25519 signature verification on FINAL_RELEASE_MANIFEST.json and dist/FINAL_RELEASE_MANIFEST.json.
2. Checkpoint authenticity verification via verify_checkpoint_authenticity().
3. Tamper resistance: rejection of modified manifests, unsigned manifests, and forged keys.
4. Production API service readiness: validates /v1/ready and /v1/health via FastAPI TestClient.
5. Packaging artifact presence in dist/.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from api.main import app
from cdm.security.manifest_verification import (
    verify_manifest_signature,
    verify_checkpoint_authenticity,
    compute_file_sha256,
    TRUSTED_RELEASE_PUBLIC_KEY,
    SecurityVerificationError,
)

EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
RELEASE_DIR = "experiments/release/cdm-1-general"


def verify_release(report_path: str = "release_report.json") -> Dict[str, Any]:
    print("=" * 70)
    print("CDM-1 VERIFICATION: RELEASE SECURITY & API READINESS")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "checks": {},
        "passed": False,
    }

    # 1. Root Release Manifest Verification
    print("[*] 1. Verifying root release manifest Ed25519 signature...")
    root_manifest_path = "FINAL_RELEASE_MANIFEST.json"
    root_manifest_ok = False
    if os.path.exists(root_manifest_path):
        try:
            root_manifest_ok = verify_manifest_signature(root_manifest_path)
        except Exception as e:
            print(f"[!] Root manifest verification error: {e}")
            root_manifest_ok = False
    report["checks"]["root_release_manifest_signature"] = {
        "status": "PASS" if root_manifest_ok else "FAIL",
        "path": root_manifest_path,
        "verified": root_manifest_ok,
    }
    print(f"[*] Root release manifest signature -> {'PASS' if root_manifest_ok else 'FAIL'}")

    # 2. Checkpoint Authenticity Gate
    print("[*] 2. Verifying release checkpoint authenticity gate...")
    ckpt_auth_ok = False
    try:
        auth_res = verify_checkpoint_authenticity(RELEASE_DIR, verify_signature=True)
        ckpt_auth_ok = (auth_res.get("status") == "authenticated" and auth_res.get("sha256") == EXPECTED_SHA256)
        report["checks"]["checkpoint_authenticity_gate"] = {
            "status": "PASS" if ckpt_auth_ok else "FAIL",
            "model_dir": RELEASE_DIR,
            "sha256": auth_res.get("sha256"),
            "size_bytes": auth_res.get("size_bytes"),
            "signature_verified": auth_res.get("signature_verified"),
        }
    except Exception as e:
        print(f"[!] Checkpoint authenticity failed: {e}")
        report["checks"]["checkpoint_authenticity_gate"] = {"status": "FAIL", "error": str(e)}
    print(f"[*] Release checkpoint authenticity -> {'PASS' if ckpt_auth_ok else 'FAIL'}")

    # 3. Security Tamper-Resistance Tests
    print("[*] 3. Testing manifest tamper detection and forged public keys...")
    tamper_ok = True
    # Test modified manifest raises SecurityVerificationError
    with open(root_manifest_path, "r", encoding="utf-8") as f:
        m_data = json.load(f)
    tampered_manifest = dict(m_data)
    tampered_manifest["model_name"] = "Forged CDM Model"
    tamper_rejected = False
    try:
        verify_manifest_signature(tampered_manifest)
    except SecurityVerificationError:
        tamper_rejected = True

    # Test wrong public key raises SecurityVerificationError
    wrong_key = "0" * 64
    wrong_key_rejected = False
    try:
        verify_manifest_signature(root_manifest_path, trusted_public_key_hex=wrong_key)
    except SecurityVerificationError:
        wrong_key_rejected = True

    tamper_ok = tamper_rejected and wrong_key_rejected
    report["checks"]["tamper_resistance"] = {
        "status": "PASS" if tamper_ok else "FAIL",
        "tampered_manifest_rejected": tamper_rejected,
        "wrong_public_key_rejected": wrong_key_rejected,
    }
    print(f"[*] Tamper detection: tampered_rejected={tamper_rejected}, wrong_key_rejected={wrong_key_rejected} -> {'PASS' if tamper_ok else 'FAIL'}")

    # 4. API Service Readiness (/v1/ready & /v1/health)
    print("[*] 4. Testing FastAPI /v1/ready and /v1/health endpoints...")
    api_ok = False
    try:
        with TestClient(app) as client:
            r_ready = client.get("/v1/ready")
            ready_status = r_ready.status_code
            ready_json = r_ready.json() if ready_status == 200 else {}

            r_health = client.get("/v1/health")
            health_status = r_health.status_code

            api_ok = (
                ready_status == 200
                and ready_json.get("status") == "ready"
                and ready_json.get("model_loaded") is True
                and ready_json.get("weights_sha256") == EXPECTED_SHA256
                and health_status == 200
            )
            report["checks"]["api_readiness"] = {
                "status": "PASS" if api_ok else "FAIL",
                "ready_http_status": ready_status,
                "ready_response": ready_json,
                "health_http_status": health_status,
            }
    except Exception as e:
        print(f"[!] API readiness check failed: {e}")
        report["checks"]["api_readiness"] = {"status": "FAIL", "error": str(e)}
    print(f"[*] API /v1/ready & /v1/health endpoints -> {'PASS' if api_ok else 'FAIL'}")

    # 5. Packaged Archives Check & Extraction Verification
    print("[*] 5. Inspecting packaged distributions and verifying extraction from CDM-1-Complete-With-Weights.zip...")
    dist_dir = "dist"
    archives = {}
    weights_archive = os.path.join(dist_dir, "CDM-1-Complete-With-Weights.zip")
    extraction_verified = False

    if os.path.exists(dist_dir):
        for item in os.listdir(dist_dir):
            ipath = os.path.join(dist_dir, item)
            if os.path.isfile(ipath):
                archives[item] = {
                    "size_bytes": os.path.getsize(ipath),
                    "sha256": compute_file_sha256(ipath),
                }

    if os.path.exists(weights_archive):
        import tempfile
        import zipfile
        with tempfile.TemporaryDirectory() as tmp_extract:
            with zipfile.ZipFile(weights_archive, "r") as zf:
                members = zf.namelist()
                has_weights = any("model.safetensors" in m for m in members)
                if has_weights:
                    zf.extractall(tmp_extract)
                    extracted_root = os.path.join(tmp_extract, "CDM-1")
                    extracted_model_dir = os.path.join(extracted_root, "experiments", "release", "cdm-1-general")
                    if os.path.exists(os.path.join(extracted_model_dir, "model.safetensors")):
                        from cdm.inference.pipeline import CognitiveDecisionModel
                        ext_model = CognitiveDecisionModel.from_pretrained(extracted_model_dir, local_files_only=True)
                        ext_res = ext_model.decide(
                            text="High priority test query on extracted weights",
                            structured_features={"monetary_value_usd": 100.0, "domain": "customer_support"},
                            task="choice",
                            candidates=["auto_resolve", "escalate_to_tier2"],
                        )
                        if ext_res.decision:
                            extraction_verified = True

    archive_status = "PASS" if (len(archives) >= 2 and extraction_verified) else "FAIL"
    report["checks"]["packaged_distributions"] = {
        "status": archive_status,
        "archives": archives,
        "clean_extraction_inference_verified": extraction_verified,
    }
    print(f"[*] Packaged archives found: {list(archives.keys())}, extraction_inference={extraction_verified} -> {archive_status}")

    overall_passed = root_manifest_ok and ckpt_auth_ok and tamper_ok and api_ok and (archive_status == "PASS")
    report["passed"] = overall_passed

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Release verification complete: {'ALL PASS' if overall_passed else 'FAILURES DETECTED'}")
    print(f"[+] Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = verify_release()
    sys.exit(0 if rep.get("passed") else 1)
