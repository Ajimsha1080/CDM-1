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


def _load_runtime_secrets() -> Tuple[bytes, str, Dict[str, bytes]]:
    """Loads HMAC secret and key ID from environment or persistent secret store.
    Supports key rotation across processes and restarts.
    """
    key_id = os.environ.get("CDM_PROVENANCE_KEY_ID", "cdm-prov-v1")
    env_secret = os.environ.get("CDM_PROVENANCE_SECRET")

    key_store: Dict[str, bytes] = {}

    if env_secret:
        active_secret = env_secret.encode("utf-8") if len(env_secret) >= 32 else hashlib.sha256(env_secret.encode("utf-8")).digest()
    else:
        key_file = os.environ.get("CDM_PROVENANCE_KEY_FILE", os.path.join("experiments", "provenance_secret.key"))
        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    active_secret = f.read()
            except Exception:
                active_secret = os.urandom(32)
        else:
            active_secret = os.urandom(32)
            try:
                os.makedirs(os.path.dirname(os.path.abspath(key_file)), exist_ok=True)
                with open(key_file, "wb") as f:
                    f.write(active_secret)
            except Exception:
                pass

    key_store[key_id] = active_secret

    # Check for additional keys in rotation keyring
    keyring_json = os.environ.get("CDM_PROVENANCE_KEY_RING")
    if keyring_json:
        try:
            parsed = json.loads(keyring_json)
            for kid, val in parsed.items():
                if isinstance(val, str):
                    key_store[kid] = bytes.fromhex(val) if len(val) == 64 else val.encode("utf-8")
        except Exception:
            pass

    return active_secret, key_id, key_store


_ACTIVE_SECRET, _ACTIVE_KEY_ID, _KEY_STORE = _load_runtime_secrets()


def reload_runtime_secrets():
    """Refreshes runtime secrets from environment/disk (useful for rotation and restarts)."""
    global _ACTIVE_SECRET, _ACTIVE_KEY_ID, _KEY_STORE
    _ACTIVE_SECRET, _ACTIVE_KEY_ID, _KEY_STORE = _load_runtime_secrets()


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
        key_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates an authenticated provenance record signed by the trusted inference runtime,
        cryptographically bound to key ID, model ID, model version, loaded artifact hash, execution ID,
        tenant ID, task, and the prediction output digest.
        """
        active_kid = key_id or _ACTIVE_KEY_ID
        signing_secret = _KEY_STORE.get(active_kid, _ACTIVE_SECRET)

        exec_id = execution_id or f"exec_{uuid.uuid4().hex}"
        timestamp = time.time()
        pred_digest = cls.compute_prediction_digest(prediction_output) if prediction_output is not None else ""

        canonical_data = (
            f"{active_kid}:{model_id}:{model_version}:{artifact_hash}:{exec_id}:{tenant_id}:{task}:{pred_digest}:{timestamp:.4f}"
        )
        signature = hmac.new(
            signing_secret, canonical_data.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return {
            "key_id": active_kid,
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
        record_kid = provenance_dict.get("key_id", _ACTIVE_KEY_ID)

        if not all([model_id, model_version, artifact_hash, execution_id, timestamp, signature]):
            return False, "unverified_caller_input", {
                "reason": "Provenance record is missing required cryptographic identity fields.",
                "missing_fields": [
                    k for k in ["model_id", "model_version", "artifact_hash", "execution_id", "timestamp", "provenance_signature"]
                    if not provenance_dict.get(k)
                ],
            }

        # 1. Verify cryptographic signature against key store secret
        secret_to_use = _KEY_STORE.get(record_kid, _ACTIVE_SECRET)
        try:
            # First try canonical data with key_id
            canonical_data = (
                f"{record_kid}:{model_id}:{model_version}:{artifact_hash}:{execution_id}:{tenant_id}:{task}:{pred_digest}:{float(timestamp):.4f}"
            )
            expected_sig = hmac.new(
                secret_to_use, canonical_data.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            sig_matches = hmac.compare_digest(signature, expected_sig)
            if not sig_matches:
                # Fallback to legacy format without key_id for backward compatibility
                legacy_canonical = (
                    f"{model_id}:{model_version}:{artifact_hash}:{execution_id}:{tenant_id}:{task}:{pred_digest}:{float(timestamp):.4f}"
                )
                legacy_expected = hmac.new(
                    secret_to_use, legacy_canonical.encode("utf-8"), hashlib.sha256
                ).hexdigest()
                sig_matches = hmac.compare_digest(signature, legacy_expected)

            if not sig_matches:
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
            "key_id": record_kid,
            "model_id": model_id,
            "model_version": model_version,
            "artifact_hash": artifact_hash,
            "execution_id": execution_id,
            "tenant_id": tenant_id,
            "task": task,
            "prediction_digest": pred_digest,
            "verified_at": time.time(),
        }
