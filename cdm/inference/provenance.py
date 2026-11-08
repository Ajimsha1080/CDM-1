"""Prediction Provenance and Cryptographic Verification for CDM-1 Inference Runtime (V10).
Binds model ID, model version, physical loaded artifact SHA-256, execution ID, tenant ID, task,
and deterministic prediction output digest to eliminate caller-controlled provenance.
"""

import os
import time
import uuid
import json
import hmac
import hashlib
from typing import Dict, Any, Tuple, Optional

# Process-private secret generated when process starts; cannot be forged by external callers
_RUNTIME_PROVENANCE_SECRET = os.urandom(32)


class PredictionProvenance:
    """Manages generation and cryptographic verification of prediction provenance."""

    @classmethod
    def compute_prediction_digest(cls, prediction_output: Any) -> str:
        """Computes a deterministic SHA-256 digest of prediction outputs."""
        if isinstance(prediction_output, dict):
            # Strip internal provenance or metadata before computing digest of decision outputs
            clean_output = {
                k: v for k, v in prediction_output.items()
                if k not in ("provenance", "provenance_signature", "timestamp")
            }
            payload = json.dumps(clean_output, sort_keys=True, separators=(",", ":")).encode("utf-8")
        elif isinstance(prediction_output, (str, bytes)):
            payload = prediction_output.encode("utf-8") if isinstance(prediction_output, str) else prediction_output
        else:
            payload = str(prediction_output).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def create(
        cls,
        model_id: str,
        model_version: str,
        artifact_hash: str,
        execution_id: Optional[str] = None,
        tenant_id: str = "default_tenant",
        task: str = "binary",
        prediction_output: Optional[Any] = None,
        estimated_risk: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Creates an authenticated provenance record signed by the trusted inference runtime,
        cryptographically bound to model ID, model version, loaded artifact hash, execution ID,
        tenant ID, task, and the prediction output digest.
        """
        exec_id = execution_id or f"exec_{uuid.uuid4().hex}"
        timestamp = time.time()
        pred_digest = cls.compute_prediction_digest(prediction_output) if prediction_output is not None else ""

        canonical_data = (
            f"{model_id}:{model_version}:{artifact_hash}:{exec_id}:{tenant_id}:{task}:{pred_digest}:{timestamp:.4f}"
        )
        signature = hmac.new(
            _RUNTIME_PROVENANCE_SECRET, canonical_data.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return {
            "model_id": model_id,
            "model_version": model_version,
            "artifact_hash": artifact_hash,
            "execution_id": exec_id,
            "tenant_id": tenant_id,
            "task": task,
            "prediction_digest": pred_digest,
            "timestamp": timestamp,
            "estimated_risk": estimated_risk,
            "prediction_source": "trained_model_prediction",
            "provenance_signature": signature,
        }

    @classmethod
    def validate(
        cls,
        provenance_dict: Optional[Dict[str, Any]],
        expected_artifact_hash: Optional[str] = None,
        expected_tenant_id: Optional[str] = None,
        expected_task: Optional[str] = None,
        prediction_output: Optional[Any] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Validates prediction provenance against cryptographic signature, artifact hash,
        tenant ID, task, and prediction output digest.
        Returns:
            Tuple of (is_valid, classification_source, diagnostic_details)
        """
        if not provenance_dict or not isinstance(provenance_dict, dict):
            return False, "deterministic_heuristic", {"reason": "No prediction provenance provided."}

        # Check required fields
        model_id = provenance_dict.get("model_id")
        model_version = provenance_dict.get("model_version")
        artifact_hash = provenance_dict.get("artifact_hash")
        execution_id = provenance_dict.get("execution_id")
        tenant_id = provenance_dict.get("tenant_id", "default_tenant")
        task = provenance_dict.get("task", "binary")
        pred_digest = provenance_dict.get("prediction_digest", "")
        timestamp = provenance_dict.get("timestamp")
        signature = provenance_dict.get("provenance_signature")

        if not all([model_id, model_version, artifact_hash, execution_id, timestamp, signature]):
            return False, "unverified_caller_input", {
                "reason": "Provenance record is missing required cryptographic identity fields.",
                "missing_fields": [
                    k for k in ["model_id", "model_version", "artifact_hash", "execution_id", "timestamp", "provenance_signature"]
                    if not provenance_dict.get(k)
                ],
            }

        # 1. Verify cryptographic signature against runtime session secret
        try:
            canonical_data = (
                f"{model_id}:{model_version}:{artifact_hash}:{execution_id}:{tenant_id}:{task}:{pred_digest}:{float(timestamp):.4f}"
            )
            expected_sig = hmac.new(
                _RUNTIME_PROVENANCE_SECRET, canonical_data.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_sig):
                return False, "unverified_caller_input", {
                    "reason": "Cryptographic provenance signature mismatch: forged or tampered provenance.",
                }
        except Exception as e:
            return False, "unverified_caller_input", {
                "reason": f"Provenance signature verification error: {e}",
            }

        # 2. Verify artifact hash matches known loaded model weights if provided
        if expected_artifact_hash and artifact_hash.lower() != expected_artifact_hash.lower():
            return False, "unverified_caller_input", {
                "reason": f"Artifact hash mismatch: expected {expected_artifact_hash}, got {artifact_hash}",
            }

        # 3. Verify tenant isolation if expected tenant provided
        if expected_tenant_id and tenant_id != expected_tenant_id:
            return False, "unverified_caller_input", {
                "reason": f"Tenant mismatch in prediction provenance: expected {expected_tenant_id}, got {tenant_id}",
            }

        # 4. Verify task if expected task provided
        if expected_task and task != expected_task:
            return False, "unverified_caller_input", {
                "reason": f"Task mismatch in prediction provenance: expected {expected_task}, got {task}",
            }

        # 5. Verify prediction digest if prediction output provided
        if prediction_output is not None and pred_digest:
            current_digest = cls.compute_prediction_digest(prediction_output)
            if current_digest != pred_digest:
                return False, "unverified_caller_input", {
                    "reason": f"Prediction output digest mismatch: outputs have been modified after inference.",
                }

        return True, "trained_model_prediction", {
            "model_id": model_id,
            "model_version": model_version,
            "artifact_hash": artifact_hash,
            "execution_id": execution_id,
            "tenant_id": tenant_id,
            "task": task,
            "prediction_digest": pred_digest,
            "verified_at": time.time(),
        }
