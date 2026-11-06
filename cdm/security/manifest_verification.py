"""Cryptographic Release Manifest Verification & Checkpoint Authenticity Gate.
Provides HMAC-SHA256 digital signature signing and verification for release manifests,
strict offline model artifact hash checking, and tamper detection against modified weights or manifests.
"""

import os
import json
import hmac
import hashlib
from typing import Dict, Any, Optional, Union


DEFAULT_TRUSTED_SIGNING_KEY = b"cdm-v8-trusted-release-root-2026-c9a7"
KEY_ID = "cdm-release-root-2026"


class SecurityVerificationError(RuntimeError):
    """Raised when release manifest or model artifact tampering/corruption is detected."""
    pass


def get_canonical_manifest_bytes(manifest_dict: Dict[str, Any]) -> bytes:
    """Serializes canonical manifest fields for cryptographic signing."""
    # Canonical subset of fields that define release integrity
    canonical_payload = {
        "release_version": manifest_dict.get("release_version"),
        "model_name": manifest_dict.get("model_name"),
        "artifact": manifest_dict.get("artifact"),
        "packaged_files": manifest_dict.get("packaged_files"),
    }
    return json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest_dict: Dict[str, Any], key: Optional[bytes] = None) -> Dict[str, Any]:
    """Appends an HMAC-SHA256 digital signature to the release manifest."""
    signing_key = key or os.environ.get("CDM_RELEASE_SIGNING_KEY", "").encode("utf-8") or DEFAULT_TRUSTED_SIGNING_KEY
    canonical_bytes = get_canonical_manifest_bytes(manifest_dict)
    signature = hmac.new(signing_key, canonical_bytes, hashlib.sha256).hexdigest()

    signed_manifest = dict(manifest_dict)
    signed_manifest["manifest_signature"] = {
        "algorithm": "HMAC-SHA256",
        "key_id": KEY_ID,
        "signature": signature,
    }
    return signed_manifest


def verify_manifest_signature(
    manifest_data: Union[str, Dict[str, Any]], key: Optional[bytes] = None
) -> bool:
    """Verifies that the release manifest has not been modified or tampered with."""
    if isinstance(manifest_data, str):
        if not os.path.isfile(manifest_data):
            raise FileNotFoundError(f"Manifest file not found: {manifest_data}")
        with open(manifest_data, "r", encoding="utf-8") as f:
            manifest_dict = json.load(f)
    else:
        manifest_dict = manifest_data

    sig_block = manifest_dict.get("manifest_signature")
    if not sig_block or not isinstance(sig_block, dict):
        raise SecurityVerificationError(
            "Release manifest lacks required cryptographic signature block ('manifest_signature')."
        )

    expected_sig = sig_block.get("signature")
    if not expected_sig:
        raise SecurityVerificationError("Manifest signature field is empty.")

    signing_key = key or os.environ.get("CDM_RELEASE_SIGNING_KEY", "").encode("utf-8") or DEFAULT_TRUSTED_SIGNING_KEY
    canonical_bytes = get_canonical_manifest_bytes(manifest_dict)
    computed_sig = hmac.new(signing_key, canonical_bytes, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected_sig.lower(), computed_sig.lower()):
        raise SecurityVerificationError(
            f"Release manifest signature mismatch! The manifest has been modified or tampered with. "
            f"(Expected: {expected_sig[:12]}..., Computed: {computed_sig[:12]}...)"
        )

    return True


def compute_file_sha256(path: str) -> str:
    """Streams file from disk and computes SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint_authenticity(
    model_dir: str,
    verify_signature: bool = True,
    key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Authenticates model checkpoint directory:
    1. Verifies release manifest exists and its digital signature is intact.
    2. Validates physical weights file exists, verifies size, and calculates actual streaming SHA-256.
    3. Guarantees tamper detection on both manifest and physical SafeTensors weights.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model checkpoint directory not found: {model_dir}")

    manifest_path = os.path.join(model_dir, "release_manifest.json")
    if not os.path.isfile(manifest_path):
        raise SecurityVerificationError(
            f"Authentication failed: release manifest missing at {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Cryptographic signature check on the manifest
    if verify_signature:
        verify_manifest_signature(manifest, key=key)

    trusted_hash = (
        manifest.get("artifact", {}).get("sha256")
        or manifest.get("packaged_files", {}).get("model.safetensors", {}).get("sha256")
    )
    if not trusted_hash:
        raise SecurityVerificationError(
            "Authentication failed: release manifest lacks trusted SHA-256 for model.safetensors"
        )

    expected_size = (
        manifest.get("artifact", {}).get("file_size_bytes")
        or manifest.get("packaged_files", {}).get("model.safetensors", {}).get("size_bytes")
    )

    weights_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Authentication failed: physical checkpoint weights missing at {weights_path}"
        )

    actual_size = os.path.getsize(weights_path)
    if expected_size and actual_size != expected_size:
        raise SecurityVerificationError(
            f"Checkpoint size mismatch: expected {expected_size} bytes, got {actual_size} bytes."
        )

    actual_hash = compute_file_sha256(weights_path)
    if actual_hash.lower() != trusted_hash.lower():
        raise SecurityVerificationError(
            f"Model checkpoint tampering detected! Physical SHA-256 does not match signed manifest. "
            f"Expected: {trusted_hash}, Got: {actual_hash}"
        )

    return {
        "status": "authenticated",
        "model_dir": model_dir,
        "sha256": actual_hash,
        "size_bytes": actual_size,
        "signature_verified": verify_signature,
        "release_version": manifest.get("release_version", "1.0.0"),
    }
