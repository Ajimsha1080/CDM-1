"""Cryptographic Release Manifest Verification & Checkpoint Authenticity Gate.
Implements Ed25519 asymmetric digital signature verification for release manifests,
strict offline model artifact hash checking, and tamper detection against modified weights or manifests.
Private signing keys are kept strictly outside the repository; only the trusted public verification key
is embedded in the application runtime.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, Union

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


# Embedded trusted public verification key (32 bytes raw Ed25519 public key in hex)
TRUSTED_RELEASE_PUBLIC_KEY = "6765c115e21dd648b1ac1d12a3183357e407deeb6d554a5caad1b8ab9dc0334d"
KEY_ID = "cdm-release-public-v9"


class SecurityVerificationError(RuntimeError):
    """Raised when release manifest or model artifact tampering, forgery, or corruption is detected."""
    pass


def get_canonical_manifest_bytes(manifest_dict: Dict[str, Any]) -> bytes:
    """Serializes canonical manifest fields for cryptographic signing."""
    canonical_payload = {
        "release_version": manifest_dict.get("release_version"),
        "model_name": manifest_dict.get("model_name"),
        "artifact": manifest_dict.get("artifact"),
        "packaged_files": manifest_dict.get("packaged_files"),
    }
    return json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(
    manifest_dict: Dict[str, Any],
    private_key_hex: Optional[str] = None,
) -> Dict[str, Any]:
    """Signs the release manifest using an Ed25519 private key.
    The private key must be provided via parameter or environment variable CDM_RELEASE_SIGNING_PRIVATE_KEY.
    Private keys are NEVER hardcoded or embedded in the codebase.
    """
    priv_hex = private_key_hex or os.environ.get("CDM_RELEASE_SIGNING_PRIVATE_KEY")
    if not priv_hex:
        raise ValueError(
            "Cannot sign release manifest: Ed25519 private key not provided. "
            "Set CDM_RELEASE_SIGNING_PRIVATE_KEY in environment or pass private_key_hex explicitly."
        )

    try:
        priv_bytes = bytes.fromhex(priv_hex.strip())
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Ed25519 private key format: {e}")

    canonical_bytes = get_canonical_manifest_bytes(manifest_dict)
    sig_bytes = private_key.sign(canonical_bytes)

    signed_manifest = dict(manifest_dict)
    signed_manifest["manifest_signature"] = {
        "algorithm": "Ed25519",
        "key_id": KEY_ID,
        "public_key": TRUSTED_RELEASE_PUBLIC_KEY,
        "signature": sig_bytes.hex(),
    }
    return signed_manifest


def verify_manifest_signature(
    manifest_data: Union[str, Dict[str, Any]],
    trusted_public_key_hex: Optional[str] = None,
) -> bool:
    """Verifies that the release manifest is signed with a trusted Ed25519 key and has not been altered."""
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
            "Release manifest is unsigned: missing required 'manifest_signature' block."
        )

    algorithm = sig_block.get("algorithm")
    if algorithm != "Ed25519":
        raise SecurityVerificationError(
            f"Unsupported manifest signature algorithm: '{algorithm}'. Expected 'Ed25519'."
        )

    signature_hex = sig_block.get("signature")
    if not signature_hex:
        raise SecurityVerificationError("Manifest signature field is empty.")

    pub_hex = trusted_public_key_hex or os.environ.get("CDM_RELEASE_PUBLIC_KEY") or TRUSTED_RELEASE_PUBLIC_KEY
    try:
        pub_bytes = bytes.fromhex(pub_hex.strip())
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
    except Exception as e:
        raise SecurityVerificationError(f"Invalid Ed25519 public verification key: {e}")

    canonical_bytes = get_canonical_manifest_bytes(manifest_dict)
    try:
        sig_bytes = bytes.fromhex(signature_hex)
        public_key.verify(sig_bytes, canonical_bytes)
    except InvalidSignature:
        raise SecurityVerificationError(
            "Ed25519 signature verification failed: the release manifest has been modified, forged, or corrupted."
        )
    except Exception as e:
        raise SecurityVerificationError(f"Ed25519 manifest signature verification error: {e}")

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
    trusted_public_key_hex: Optional[str] = None,
) -> Dict[str, Any]:
    """Authenticates model checkpoint directory:
    1. Verifies release manifest exists and its Ed25519 asymmetric signature is valid.
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

    # 1. Asymmetric Ed25519 signature check on the manifest
    if verify_signature:
        verify_manifest_signature(manifest, trusted_public_key_hex=trusted_public_key_hex)

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
        "signature_algorithm": "Ed25519",
        "release_version": manifest.get("release_version", "1.0.0"),
    }
