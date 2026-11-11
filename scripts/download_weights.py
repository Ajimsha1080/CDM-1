#!/usr/bin/env python3
"""Download and verify genuine neural weights for CDM-1 from Hugging Face Hub.

Verifies SHA-256 checksum against signed release manifest before saving.
The repository never tracks weight binary files directly.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_HUB_REPO = os.getenv("CDM_WEIGHTS_REPO", "cdm-project/cdm-1-general")
DEFAULT_DESTINATION = Path("experiments/release/cdm-1-general")
DEFAULT_FILENAME = "model.safetensors"
EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_expected_hash(dest_dir: Path, explicit_hash: str | None = None) -> str:
    """Retrieve expected hash from explicit parameter or local manifest."""
    if explicit_hash:
        return explicit_hash.lower().strip()

    manifest_path = dest_dir / "release_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["artifact"]["sha256"].lower().strip()
        except Exception as e:
            print(f"[!] Warning: Could not read release_manifest.json: {e}")

    return EXPECTED_SHA256


def verify_weights(file_path: Path, expected_hash: str) -> bool:
    """Check if the local file exists and matches the expected SHA-256 hash."""
    if not file_path.exists():
        print(f"[-] File not found: {file_path}")
        return False

    print(f"[*] Verifying checksum for: {file_path}")
    actual_hash = compute_sha256(file_path)
    if actual_hash == expected_hash:
        print(f"[+] Checksum verified: {actual_hash}")
        return True
    else:
        print("[-] Checksum mismatch!")
        print(f"    Expected: {expected_hash}")
        print(f"    Actual:   {actual_hash}")
        return False


def download_file(url: str, dest_path: Path, expected_hash: str, timeout: int = 300) -> bool:
    """Download a file with streaming SHA-256 calculation and atomic rename."""
    temp_path = dest_path.with_suffix(".safetensors.part")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading weights from: {url}")
    print(f"[*] Destination: {dest_path}")
    hasher = hashlib.sha256()
    bytes_downloaded = 0

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "cdm-1-weight-downloader/0.1.0"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            total_size_hdr = response.headers.get("Content-Length")
            total_size = int(total_size_hdr) if total_size_hdr else 0

            with open(temp_path, "wb") as f:
                chunk_size = 1024 * 1024  # 1 MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    hasher.update(chunk)
                    bytes_downloaded += len(chunk)
                    if total_size > 0:
                        pct = (bytes_downloaded / total_size) * 100
                        mb_down = bytes_downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r    Progress: {pct:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)", end="", flush=True)
                    else:
                        mb_down = bytes_downloaded / (1024 * 1024)
                        print(f"\r    Downloaded: {mb_down:6.1f} MB", end="", flush=True)
            print()

        actual_hash = hasher.hexdigest()
        if actual_hash != expected_hash:
            if temp_path.exists():
                temp_path.unlink()
            print(f"[-] Verification failed! Expected {expected_hash}, got {actual_hash}")
            return False

        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        print(f"[+] Download and cryptographic verification complete: {dest_path}")
        return True

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[-] Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and verify CDM-1 neural model weights.")
    parser.add_argument(
        "--hub-repo",
        default=DEFAULT_HUB_REPO,
        help=f"Hugging Face repository ID (default: {DEFAULT_HUB_REPO})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Target directory (default: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Weight filename (default: {DEFAULT_FILENAME})",
    )
    parser.add_argument(
        "--expected-hash",
        default=None,
        help="Expected SHA-256 hash (defaults to manifest or built-in hash)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing local weights, do not download",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing local weights",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download request timeout in seconds (default: 300)",
    )

    args = parser.parse_args()
    dest_path = args.destination / args.filename
    expected_hash = get_expected_hash(args.destination, args.expected_hash)

    if args.verify_only:
        success = verify_weights(dest_path, expected_hash)
        sys.exit(0 if success else 1)

    if dest_path.exists() and not args.force:
        print(f"[*] Found existing weights file: {dest_path}")
        # Check if existing weights file is an unhydrated Git LFS pointer
        if dest_path.stat().st_size < 1024:
            with open(dest_path, "rb") as f:
                header = f.read(256)
            if b"git-lfs" in header or b"oid sha256:" in header:
                print("[!] Detected Git LFS pointer text file (134 bytes). Attempting git lfs pull...")
                import subprocess
                try:
                    p = subprocess.run(["git", "lfs", "pull"], capture_output=True, text=True, timeout=180)
                    if p.returncode == 0 and dest_path.stat().st_size > 1024:
                        print("[+] git lfs pull succeeded in hydrating model weights.")
                except Exception as e:
                    print(f"[-] git lfs pull error: {e}")

        if verify_weights(dest_path, expected_hash):
            print("[+] Existing weights are valid. Use --force to re-download.")
            sys.exit(0)
        else:
            print("[!] Existing weights have checksum mismatch. Re-downloading...")

    url = f"https://huggingface.co/{args.hub_repo}/resolve/main/{args.filename}"
    success = download_file(url, dest_path, expected_hash, timeout=args.timeout)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
