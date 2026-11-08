"""Verification test ensuring every packaged release file strictly matches its signed manifest SHA-256."""

import os
import json
import hashlib
import pytest
from cdm.security.manifest_verification import compute_file_sha256, verify_manifest_signature


def test_packaged_files_sha256_match_manifest():
    """Verify that every file listed in release_manifest.json packaged_files exists and matches exact SHA-256."""
    release_dir = "experiments/release/cdm-1-general"
    manifest_path = os.path.join(release_dir, "release_manifest.json")
    assert os.path.isfile(manifest_path), f"Missing release manifest at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify Ed25519 digital signature of the manifest itself
    assert verify_manifest_signature(manifest) is True

    # 2. Verify all packaged files
    packaged_files = manifest.get("packaged_files", {})
    assert len(packaged_files) > 0, "No packaged_files declared in release manifest"

    for rel_path, meta in packaged_files.items():
        full_path = os.path.join(release_dir, rel_path)
        assert os.path.isfile(full_path), f"Packaged file not found on disk: {full_path}"

        expected_size = meta.get("size_bytes")
        if expected_size is not None:
            actual_size = os.path.getsize(full_path)
            assert actual_size == expected_size, f"Size mismatch for {rel_path}: expected {expected_size}, got {actual_size}"

        expected_hash = meta.get("sha256")
        assert expected_hash, f"Missing sha256 for packaged file {rel_path}"
        actual_hash = compute_file_sha256(full_path)
        assert actual_hash.lower() == expected_hash.lower(), (
            f"SHA-256 mismatch for {rel_path}! Expected {expected_hash}, got {actual_hash}"
        )
