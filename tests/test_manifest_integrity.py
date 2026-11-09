"""Tests for release manifest integrity, Ed25519 digital signature, and multi-file tamper detection.
"""

import os
import json
import pytest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519

from cdm.security.manifest_verification import (
    sign_manifest,
    verify_manifest_signature,
    verify_checkpoint_authenticity,
    compute_file_sha256,
    SecurityVerificationError,
    TRUSTED_RELEASE_PUBLIC_KEY,
)


@pytest.fixture
def mock_release_dir(tmp_path):
    """Creates a temporary valid release directory signed with a fresh Ed25519 keypair."""
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_hex = priv.private_bytes_raw().hex()
    pub_hex = priv.public_key().public_bytes_raw().hex()

    files = {
        "config.json": json.dumps({"language_encoder": {"projection_dim": 256}}).encode("utf-8"),
        "feature_schema.json": json.dumps({"numeric_features": ["val"]}).encode("utf-8"),
        "tokenizer_config.json": json.dumps({"model_type": "modernbert"}).encode("utf-8"),
        "tokenizer.json": json.dumps({"version": "1.0"}).encode("utf-8"),
        "model.safetensors": b"FAKETENSORS_DATA_12345",
    }

    packaged_files = {}
    for filename, content in files.items():
        fpath = tmp_path / filename
        fpath.write_bytes(content)
        packaged_files[filename] = {
            "size_bytes": len(content),
            "sha256": compute_file_sha256(str(fpath)),
        }

    manifest = {
        "release_version": "1.0.0",
        "model_name": "Test-Model",
        "architecture": {
            "language_projection_dim": 256,
            "structured_encoder_dim": 256,
        },
        "packaged_files": packaged_files,
        "artifact": {
            "file_name": "model.safetensors",
            "sha256": packaged_files["model.safetensors"]["sha256"],
            "file_size_bytes": packaged_files["model.safetensors"]["size_bytes"],
        },
    }

    signed_manifest = sign_manifest(manifest, private_key_hex=priv_hex)
    manifest_path = tmp_path / "release_manifest.json"
    manifest_path.write_text(json.dumps(signed_manifest, indent=2), encoding="utf-8")

    return tmp_path, pub_hex, priv_hex


def test_manifest_verification_success(mock_release_dir):
    release_dir, pub_hex, _ = mock_release_dir
    res = verify_checkpoint_authenticity(str(release_dir), trusted_public_key_hex=pub_hex)
    assert res["status"] == "authenticated"
    assert len(res["files_verified"]) == 5


@pytest.mark.parametrize(
    "tampered_file",
    ["config.json", "feature_schema.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"],
)
def test_manifest_fails_on_file_content_tampering(mock_release_dir, tampered_file):
    release_dir, pub_hex, _ = mock_release_dir
    target = release_dir / tampered_file
    orig_content = target.read_bytes()
    target.write_bytes(orig_content + b"_CORRUPTED")

    with pytest.raises(SecurityVerificationError, match=rf"(Checkpoint size mismatch|tampering detected) for '{tampered_file}'"):
        verify_checkpoint_authenticity(str(release_dir), trusted_public_key_hex=pub_hex)


def test_manifest_fails_on_unlisted_extra_file(mock_release_dir):
    release_dir, pub_hex, _ = mock_release_dir
    extra = release_dir / "unlisted_trojan.exe"
    extra.write_text("malicious payload", encoding="utf-8")

    with pytest.raises(SecurityVerificationError, match="unlisted extra file"):
        verify_checkpoint_authenticity(str(release_dir), trusted_public_key_hex=pub_hex)


def test_manifest_fails_on_missing_file(mock_release_dir):
    release_dir, pub_hex, _ = mock_release_dir
    (release_dir / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="required packaged file.*missing"):
        verify_checkpoint_authenticity(str(release_dir), trusted_public_key_hex=pub_hex)


def test_manifest_fails_on_forged_signature(mock_release_dir):
    release_dir, pub_hex, _ = mock_release_dir
    manifest_path = release_dir / "release_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Tamper with architecture without re-signing
    data["architecture"]["language_projection_dim"] = 9999
    manifest_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SecurityVerificationError, match="Ed25519 signature verification failed"):
        verify_checkpoint_authenticity(str(release_dir), trusted_public_key_hex=pub_hex)


def test_manifest_signing_requires_private_key(monkeypatch):
    monkeypatch.delenv("CDM_RELEASE_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("CDM_RELEASE_SIGNING_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="Ed25519 private key not provided"):
        sign_manifest({"release_version": "1.0.0"})


def test_actual_release_directory_verifies():
    """Verify that experiments/release/cdm-1-general verifies cleanly against the embedded public key."""
    actual_release_dir = Path(__file__).resolve().parent.parent / "experiments" / "release" / "cdm-1-general"
    if (actual_release_dir / "model.safetensors").exists():
        res = verify_checkpoint_authenticity(str(actual_release_dir))
        assert res["status"] == "authenticated"
        assert "model.safetensors" in res["files_verified"]
        assert "config.json" in res["files_verified"]
        assert "feature_schema.json" in res["files_verified"]
        assert "tokenizer_config.json" in res["files_verified"]
