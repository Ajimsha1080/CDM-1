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
TRUSTED_RELEASE_PUBLIC_KEY = "4784afcf98a872ec65cc951f8803c6619e777923b559196f4cd9297bbadf56a5"
KEY_ID = "cdm-release-public-v10"


class SecurityVerificationError(RuntimeError):
    """Raised when release manifest or model artifact tampering, forgery, or corruption is detected."""
    pass


def get_canonical_manifest_bytes(manifest_dict: Dict[str, Any]) -> bytes:
    """Serializes canonical manifest fields for cryptographic signing."""
    canonical_payload = {
        "release_version": manifest_dict.get("release_version"),
        "model_name": manifest_dict.get("model_name"),
        "architecture": manifest_dict.get("architecture"),
        "artifact": manifest_dict.get("artifact"),
        "packaged_files": manifest_dict.get("packaged_files"),
    }
    return json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(
    manifest_dict: Dict[str, Any],
    private_key_hex: Optional[str] = None,
) -> Dict[str, Any]:
    """Signs the release manifest using an Ed25519 private key.
    The private key must come from an environment variable (CDM_RELEASE_PRIVATE_KEY) only,
    or explicitly passed in tests. Private keys are NEVER hardcoded or embedded in the codebase.
    """
    priv_hex = private_key_hex or os.environ.get("CDM_RELEASE_PRIVATE_KEY") or os.environ.get("CDM_RELEASE_SIGNING_PRIVATE_KEY")
    if not priv_hex:
        raise ValueError(
            "Cannot sign release manifest: Ed25519 private key not provided. "
            "Set CDM_RELEASE_PRIVATE_KEY in environment or pass private_key_hex explicitly."
        )

    try:
        priv_bytes = bytes.fromhex(priv_hex.strip())
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
    except Exception as e:
        raise ValueError(f"Invalid Ed25519 private key format: {e}")

    canonical_bytes = get_canonical_manifest_bytes(manifest_dict)
    sig_bytes = private_key.sign(canonical_bytes)

    # Derive public key hex from the private key
    pub_hex = private_key.public_key().public_bytes_raw().hex()

    signed_manifest = dict(manifest_dict)
    signed_manifest["manifest_signature"] = {
        "algorithm": "Ed25519",
        "key_id": KEY_ID,
        "public_key": pub_hex,
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
    2. Validates EVERY file listed in packaged_files: checks size_bytes and computes actual SHA-256.
    3. Guarantees no unlisted extra files exist in the model directory (excluding release_manifest.json).
    4. Guarantees no missing files from packaged_files.
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

    packaged_files = manifest.get("packaged_files")
    artifact = manifest.get("artifact")

    all_expected = dict(packaged_files) if isinstance(packaged_files, dict) else {}
    if isinstance(artifact, dict) and artifact.get("file_name"):
        art_name = artifact["file_name"]
        if art_name not in all_expected:
            all_expected[art_name] = {
                "size_bytes": artifact.get("file_size_bytes"),
                "sha256": artifact.get("sha256"),
            }

    if not all_expected:
        raise SecurityVerificationError(
            "Authentication failed: release manifest lacks required 'packaged_files' or 'artifact' declarations."
        )

    # 2. Check for unlisted extra files and missing files
    ignored_patterns = {"release_manifest.json", ".DS_Store", "Thumbs.db"}
    actual_files = set(
        f for f in os.listdir(model_dir)
        if f not in ignored_patterns and not f.startswith(".") and os.path.isfile(os.path.join(model_dir, f))
    )
    expected_files = set(all_expected.keys())

    extra_files = actual_files - expected_files
    if extra_files:
        raise SecurityVerificationError(
            f"Authentication failed: unlisted extra file(s) found in model directory: {sorted(list(extra_files))}"
        )

    missing_files = expected_files - actual_files
    if missing_files:
        raise FileNotFoundError(
            f"Authentication failed: required packaged file(s) missing: {sorted(list(missing_files))}"
        )

    # 3. Verify EVERY file in all_expected
    file_hashes = {}
    for file_name, file_meta in all_expected.items():
        file_path = os.path.join(model_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Packaged file not found: {file_path}")

        expected_size = file_meta.get("size_bytes")
        actual_size = os.path.getsize(file_path)
        if expected_size is not None and actual_size != expected_size:
            raise SecurityVerificationError(
                f"Checkpoint size mismatch for '{file_name}': expected {expected_size} bytes, got {actual_size} bytes."
            )

        expected_hash = file_meta.get("sha256")
        if not expected_hash:
            raise SecurityVerificationError(f"Manifest missing expected SHA-256 for '{file_name}'.")

        actual_hash = compute_file_sha256(file_path)
        if actual_hash.lower() != expected_hash.lower():
            raise SecurityVerificationError(
                f"File tampering detected for '{file_name}'! Physical SHA-256 does not match signed manifest. "
                f"Expected: {expected_hash}, Got: {actual_hash}"
            )
        file_hashes[file_name] = actual_hash

    model_sha256 = None
    if isinstance(artifact, dict) and artifact.get("sha256"):
        model_sha256 = artifact.get("sha256")
    elif "model.safetensors" in file_hashes:
        model_sha256 = file_hashes["model.safetensors"]

    return {
        "status": "authenticated",
        "model_dir": model_dir,
        "files_verified": list(all_expected.keys()),
        "file_hashes": file_hashes,
        "sha256": model_sha256,
        "signature_verified": verify_signature,
        "signature_algorithm": "Ed25519",
        "release_version": manifest.get("release_version", "1.0.0"),
    }


def verify_release_manifest_mode(
    release_dir: str,
    mode: str = "full_model",
    trusted_public_key_hex: Optional[str] = None,
    verify_offline_inference: bool = True,
) -> Dict[str, Any]:
    """Verification with two explicit modes:
    Mode A ('software_only'):
      Verifies configuration, schema, tokenizer, and non-model packaged files.
      May succeed without model weights, but strictly reports that the neural model is absent.
    Mode B ('full_model'):
      MANDATORY full release verification.
      Verifies checkpoint presence, exact file size, SHA-256, tensor metadata,
      architecture compatibility, model version, and offline checkpoint loading.
      Strictly raises SecurityVerificationError or FileNotFoundError if any model artifact is missing or invalid.
    """
    if mode not in ("full_model", "software_only"):
        raise ValueError(f"Unknown verification mode '{mode}'. Choose 'full_model' or 'software_only'.")

    if not os.path.isdir(release_dir):
        raise FileNotFoundError(f"Release directory not found: {release_dir}")

    manifest_path = os.path.join(release_dir, "release_manifest.json")
    if not os.path.isfile(manifest_path):
        raise SecurityVerificationError(f"Release manifest missing at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Asymmetric Ed25519 signature check
    verify_manifest_signature(manifest, trusted_public_key_hex=trusted_public_key_hex)

    # 2. Check architecture sync with config.json
    config_path = os.path.join(release_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Required configuration file missing: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    arch = manifest.get("architecture", {})
    mismatches = []
    if arch.get("language_projection_dim") != cfg.get("language_encoder", {}).get("projection_dim"):
        mismatches.append("language_projection_dim")
    if arch.get("structured_encoder_dim") != cfg.get("structured_encoder", {}).get("projection_dim"):
        mismatches.append("structured_encoder_dim")
    if arch.get("cross_modal_fusion_dim") != cfg.get("fusion", {}).get("fusion_dim"):
        mismatches.append("cross_modal_fusion_dim")
    if arch.get("predictive_network_heads") != cfg.get("predictive", {}).get("num_outcome_classes"):
        mismatches.append("predictive_network_heads")
    if arch.get("decision_network_heads") != cfg.get("decision", {}).get("num_priority_classes"):
        mismatches.append("decision_network_heads")

    if mismatches:
        raise SecurityVerificationError(f"Architecture mismatch between config.json and manifest in: {mismatches}")

    # 3. Check packaged files
    packaged_files = manifest.get("packaged_files", {})
    if not packaged_files:
        raise SecurityVerificationError("Release manifest lacks required 'packaged_files' block.")

    file_hashes = {}
    model_absent = False

    for file_name, meta in packaged_files.items():
        file_path = os.path.join(release_dir, file_name)
        if not os.path.isfile(file_path):
            if file_name == "model.safetensors" and mode == "software_only":
                model_absent = True
                continue
            raise FileNotFoundError(f"Required release artifact missing: {file_name}")

        # In software_only mode, allow model.safetensors to be either absent or an unhydrated LFS pointer
        if file_name == "model.safetensors" and mode == "software_only":
            actual_size = os.path.getsize(file_path)
            expected_size = meta.get("size_bytes")
            if actual_size != expected_size:
                model_absent = True
                continue

        actual_size = os.path.getsize(file_path)
        expected_size = meta.get("size_bytes")
        if expected_size is not None and actual_size != expected_size:
            raise SecurityVerificationError(
                f"Size mismatch for '{file_name}': expected {expected_size} bytes, got {actual_size} bytes."
            )

        actual_hash = compute_file_sha256(file_path)
        expected_hash = meta.get("sha256", "").lower()
        if actual_hash.lower() != expected_hash:
            raise SecurityVerificationError(
                f"File tampering detected for '{file_name}'! SHA-256 does not match manifest. Expected {expected_hash}, got {actual_hash}"
            )
        file_hashes[file_name] = actual_hash

    # In full_model mode, model.safetensors MUST exist and be the actual trained binary
    weights_path = os.path.join(release_dir, "model.safetensors")
    if mode == "full_model":
        if not os.path.isfile(weights_path):
            raise FileNotFoundError("Full model-release verification failed: model.safetensors is absent.")
        if os.path.getsize(weights_path) < 1024:
            raise SecurityVerificationError(
                f"Full model verification failed: '{weights_path}' is an unhydrated Git LFS pointer ({os.path.getsize(weights_path)} bytes), "
                "not the genuine trained neural model binary."
            )

        # Tensor architecture and parameter validation using safetensors
        import safetensors.torch
        with safetensors.safe_open(weights_path, framework="pt") as sf:
            keys = list(sf.keys())
        if len(keys) != 215:
            raise SecurityVerificationError(
                f"Incompatible tensor architecture: expected 215 tensor keys, found {len(keys)} keys."
            )

        # Offline inference test
        if verify_offline_inference:
            from cdm.inference.pipeline import CognitiveDecisionModel
            model = CognitiveDecisionModel.from_pretrained(release_dir, local_files_only=True)
            if getattr(model, "is_fallback", False):
                raise SecurityVerificationError("Offline model loading returned fallback instead of trained weights.")
            test_res = model.decide(
                text="Verification dispute claim test",
                structured_features={"monetary_value_usd": 100.0, "urgency_level": "medium"},
                task="binary",
            )
            if test_res.decision not in ("YES", "NO", "APPROVE", "REJECT", "ABSTAIN"):
                raise SecurityVerificationError(f"Unexpected decision output: {test_res.decision}")

    return {
        "status": "authenticated",
        "mode": mode,
        "model_absent": model_absent,
        "files_verified": list(file_hashes.keys()),
        "file_hashes": file_hashes,
        "release_version": manifest.get("release_version", "1.0.0"),
        "signature_verified": True,
    }
