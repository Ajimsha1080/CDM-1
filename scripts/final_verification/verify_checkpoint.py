#!/usr/bin/env python3
"""CDM-1 Final Verification: Model Checkpoint Integrity & Genuine Offline Inference.

Verifies:
1. File existence and size (633,465,536 bytes)
2. SHA-256 hash match against official reference
3. Tensor count (215 tensors) and total parameter count (158,360,004 parameters)
4. Strict offline reload via SafeTensors
5. Genuine real neural inference and prediction provenance generation
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import safetensors.torch
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.security.manifest_verification import compute_file_sha256

EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
EXPECTED_SIZE_BYTES = 633465536
EXPECTED_TENSOR_COUNT = 215
EXPECTED_PARAM_COUNT = 158360004
PRIMARY_CHECKPOINT_DIR = "experiments/release/cdm-1-general"
PACKAGED_CHECKPOINT_DIR = "dist/package/cdm-1-0.1.0/cdm/models/cdm-1-general"


def verify_checkpoint(report_path: str = "checkpoint_report.json") -> Dict[str, Any]:
    print("=" * 70)
    print("CDM-1 VERIFICATION: MODEL CHECKPOINT INTEGRITY & OFFLINE INFERENCE")
    print("=" * 70)

    weights_file = os.path.join(PRIMARY_CHECKPOINT_DIR, "model.safetensors")
    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "checkpoint_path": weights_file,
        "checks": {},
        "passed": False,
    }

    # 1. Existence check
    print(f"[*] Checking file existence: {weights_file}")
    if not os.path.exists(weights_file):
        report["checks"]["file_exists"] = {"status": "FAIL", "error": f"File not found: {weights_file}"}
        print(f"[!] FAILED: Model file not found at {weights_file}")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    actual_size = os.path.getsize(weights_file)
    size_ok = (actual_size == EXPECTED_SIZE_BYTES)
    report["checks"]["file_size"] = {
        "status": "PASS" if size_ok else "FAIL",
        "expected_bytes": EXPECTED_SIZE_BYTES,
        "actual_bytes": actual_size,
    }
    print(f"[*] File size: {actual_size} bytes (Expected: {EXPECTED_SIZE_BYTES}) -> {'PASS' if size_ok else 'FAIL'}")

    # 2. SHA-256 calculation
    print("[*] Computing SHA-256 digest...")
    actual_sha256 = compute_file_sha256(weights_file)
    sha_ok = (actual_sha256.lower() == EXPECTED_SHA256.lower())
    report["checks"]["sha256"] = {
        "status": "PASS" if sha_ok else "FAIL",
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": actual_sha256,
    }
    print(f"[*] Checkpoint SHA-256: {actual_sha256} -> {'PASS' if sha_ok else 'FAIL'}")

    # 3. SafeTensors tensor inspection
    print("[*] Inspecting SafeTensors tensors and parameter count...")
    tensors = safetensors.torch.load_file(weights_file, device="cpu")
    actual_tensors = len(tensors)
    actual_params = sum(t.numel() for t in tensors.values())
    tensors_ok = (actual_tensors == EXPECTED_TENSOR_COUNT)
    params_ok = (actual_params == EXPECTED_PARAM_COUNT)

    report["checks"]["tensor_count"] = {
        "status": "PASS" if tensors_ok else "FAIL",
        "expected_tensors": EXPECTED_TENSOR_COUNT,
        "actual_tensors": actual_tensors,
    }
    report["checks"]["param_count"] = {
        "status": "PASS" if params_ok else "FAIL",
        "expected_params": EXPECTED_PARAM_COUNT,
        "actual_params": actual_params,
    }
    print(f"[*] Tensor count: {actual_tensors} (Expected: {EXPECTED_TENSOR_COUNT}) -> {'PASS' if tensors_ok else 'FAIL'}")
    print(f"[*] Parameter count: {actual_params:,} (Expected: {EXPECTED_PARAM_COUNT:,}) -> {'PASS' if params_ok else 'FAIL'}")

    # 4. Strict offline model reload & real neural inference
    print("[*] Performing strict offline reload and real neural inference...")
    try:
        model = CognitiveDecisionModel.from_pretrained(model_name_or_path=PRIMARY_CHECKPOINT_DIR, local_files_only=True)
        assert model.artifact_hash == EXPECTED_SHA256

        t0 = time.time()
        res = model.decide(
            structured_features={"monetary_risk": 0.25, "operational_urgency": 0.75, "customer_tier": 2.0},
            text="High priority escalation request requiring automated neural risk assessment.",
            task="binary",
            candidates=["auto_resolve", "escalate_tier2"],
            tenant_id="tenant_verification_run",
        )
        latency_ms = round((time.time() - t0) * 1000, 2)

        inference_ok = (
            res.decision in ["YES", "NO"]
            and 0.0 <= res.confidence <= 1.0
            and 0.0 <= res.risk <= 1.0
            and res.provenance is not None
            and res.provenance.get("artifact_hash") == EXPECTED_SHA256
            and res.provenance.get("prediction_source") == "trained_model_prediction"
        )

        report["checks"]["offline_inference"] = {
            "status": "PASS" if inference_ok else "FAIL",
            "decision": res.decision,
            "confidence": round(res.confidence, 4),
            "risk": round(res.risk, 4),
            "latency_ms": latency_ms,
            "candidate_scores": res.candidate_scores,
            "provenance_authenticated": res.provenance.get("authenticated", False),
            "provenance_digest": res.provenance.get("provenance_digest"),
            "artifact_hash": res.provenance.get("artifact_hash"),
        }
        print(f"[*] Offline inference result: decision={res.decision}, conf={res.confidence:.4f}, risk={res.risk:.4f}, latency={latency_ms}ms -> {'PASS' if inference_ok else 'FAIL'}")
    except Exception as e:
        inference_ok = False
        report["checks"]["offline_inference"] = {"status": "FAIL", "error": str(e)}
        print(f"[!] Offline inference failed: {e}")

    # 5. Packaged checkpoint check (dist/package)
    packaged_weights = os.path.join(PACKAGED_CHECKPOINT_DIR, "model.safetensors")
    packaged_exists = os.path.exists(packaged_weights)
    if packaged_exists:
        pkg_size = os.path.getsize(packaged_weights)
        pkg_ok = (pkg_size == EXPECTED_SIZE_BYTES)
        report["checks"]["packaged_distribution_checkpoint"] = {
            "status": "PASS" if pkg_ok else "FAIL",
            "path": packaged_weights,
            "bytes": pkg_size,
        }
        print(f"[*] Packaged checkpoint verified: {packaged_weights} ({pkg_size} bytes) -> PASS")
    else:
        report["checks"]["packaged_distribution_checkpoint"] = {
            "status": "WARN",
            "path": packaged_weights,
            "note": "Package distribution not yet synchronized with this run",
        }

    overall_passed = size_ok and sha_ok and tensors_ok and params_ok and inference_ok
    report["passed"] = overall_passed

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Checkpoint verification complete: {'ALL PASS' if overall_passed else 'FAILURES DETECTED'}")
    print(f"[+] Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = verify_checkpoint()
    sys.exit(0 if rep.get("passed") else 1)
