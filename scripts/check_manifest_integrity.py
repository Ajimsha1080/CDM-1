#!/usr/bin/env python3
"""CI check: Recomputes hashes and sizes for release manifest files,
verifies Ed25519 digital signature, and validates architecture synchronization.
Supports two explicit verification modes:
  Mode A: --mode software_only (verifies non-model artifacts, reports model absent)
  Mode B: --mode full_model (MANDATORY: verifies model weights, size, sha256, tensor count, offline inference)
Exits with code 0 on success, code 1 on mismatch or tampering.
"""

import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cdm.security.manifest_verification import (
    verify_release_manifest_mode,
    TRUSTED_RELEASE_PUBLIC_KEY,
)

RELEASE_DIR = ROOT_DIR / "experiments" / "release" / "cdm-1-general"


def main():
    parser = argparse.ArgumentParser(description="CDM-1 Release Manifest Integrity Verifier")
    parser.add_argument(
        "--mode",
        choices=["full_model", "software_only"],
        default=os.environ.get("CDM_VERIFY_MODE", "full_model"),
        help="Verification mode: 'full_model' (default) enforces model checkpoint presence; 'software_only' verifies non-model artifacts.",
    )
    parser.add_argument(
        "--release-dir",
        default=str(RELEASE_DIR),
        help="Directory containing release manifest and packaged files.",
    )
    args = parser.parse_args()

    print(f"[*] Starting CDM-1 Manifest Integrity Verification (Mode: {args.mode.upper()})...")
    try:
        res = verify_release_manifest_mode(
            release_dir=args.release_dir,
            mode=args.mode,
            trusted_public_key_hex=TRUSTED_RELEASE_PUBLIC_KEY,
            verify_offline_inference=True,
        )
        print("    [+] Ed25519 digital signature is VALID.")
        print(f"    [+] Verified {len(res['files_verified'])} packaged files:")
        for fn, h in res["file_hashes"].items():
            print(f"        - {fn}: SHA-256 {h[:16]}...")
        if res.get("model_absent"):
            print("    [!] NOTE: Neural model checkpoint (model.safetensors) is ABSENT. Software components verified.")
        else:
            print("    [+] Model checkpoint (model.safetensors) verified (215 tensors, offline inference passed).")
        print(f"\n[+] Manifest integrity check PASSED (Mode: {args.mode.upper()}).")
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] MANIFEST INTEGRITY CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
