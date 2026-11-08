#!/usr/bin/env python3
"""CI check: Recomputes hashes and sizes for release manifest files,
verifies Ed25519 digital signature, and validates architecture synchronization.
Exits with code 0 on success, code 1 on mismatch or tampering.
"""

import os
import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cdm.security.manifest_verification import (
    compute_file_sha256,
    verify_manifest_signature,
    TRUSTED_RELEASE_PUBLIC_KEY,
)

RELEASE_DIR = ROOT_DIR / "experiments" / "release" / "cdm-1-general"


def main():
    manifest_path = RELEASE_DIR / "release_manifest.json"
    if not manifest_path.exists():
        print(f"[-] FATAL: Release manifest missing at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify Ed25519 Signature
    print(f"[*] Verifying Ed25519 signature of {manifest_path.name}...")
    try:
        verify_manifest_signature(manifest, trusted_public_key_hex=TRUSTED_RELEASE_PUBLIC_KEY)
        print("    [+] Ed25519 signature is VALID.")
    except Exception as e:
        print(f"[-] FATAL: Ed25519 signature verification failed: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Check architecture sync with config.json
    config_path = RELEASE_DIR / "config.json"
    if config_path.exists():
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
            print(f"[-] FATAL: Architecture mismatch between config.json and manifest in: {mismatches}", file=sys.stderr)
            sys.exit(1)
        print("    [+] Architecture fields match config.json exactly.")

    # 3. Recompute hashes and sizes for packaged files
    packaged_files = manifest.get("packaged_files", {})
    if not packaged_files:
        print("[-] FATAL: No packaged_files in release manifest", file=sys.stderr)
        sys.exit(1)

    errors = []
    checked_count = 0

    for file_name, meta in packaged_files.items():
        file_path = RELEASE_DIR / file_name
        if not file_path.exists():
            if file_name == "model.safetensors":
                print(f"    [!] Note: {file_name} not present in local workspace (download required for offline weights inference).")
                continue
            else:
                errors.append(f"Required release file missing: {file_name}")
                continue

        actual_size = file_path.stat().st_size
        expected_size = meta.get("size_bytes")
        if expected_size is not None and actual_size != expected_size:
            errors.append(f"Size mismatch for {file_name}: expected {expected_size}, got {actual_size}")

        actual_hash = compute_file_sha256(str(file_path))
        expected_hash = meta.get("sha256", "").lower()
        if actual_hash.lower() != expected_hash:
            errors.append(f"SHA-256 mismatch for {file_name}:\n  expected {expected_hash}\n  actual   {actual_hash}")
        else:
            print(f"    [+] {file_name} verified ({actual_size} bytes, SHA-256: {actual_hash[:16]}...)")
        checked_count += 1

    if errors:
        print("\n[-] MANIFEST INTEGRITY CHECK FAILED:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[+] Manifest integrity check PASSED ({checked_count} files verified, signature verified).")
    sys.exit(0)


if __name__ == "__main__":
    main()
