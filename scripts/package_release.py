"""Packaging & Distribution Utility for CDM-1 (Cross-Platform).
Creates portable zip archives with configurable destination, SHA-256 release manifests,
and strict exclusion of credentials, caches, and runtime databases.
"""

import os
import sys
import zipfile
import shutil
import hashlib
import json
import argparse
from typing import Dict, Any

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "dist",
    "build",
}

EXCLUDE_EXTS = {
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".env",
    ".pem",
    ".key",
}

EXCLUDE_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    "experiments/api_keys.sqlite",
    "experiments/api_usage_meter.sqlite",
}


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_zip(zip_path: str, include_weights: bool = False) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    print(f"Building {zip_path} (include_weights={include_weights})...")
    manifest_entries = {}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in EXCLUDE_EXTS:
                    continue
                if file.endswith(".zip"):
                    continue
                if ext == ".safetensors" and not include_weights:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ROOT_DIR).replace("\\", "/")

                if rel_path in EXCLUDE_FILENAMES or file.startswith(".env"):
                    continue

                zf.write(full_path, arcname=f"CDM-1/{rel_path}")
                manifest_entries[rel_path] = {
                    "size_bytes": os.path.getsize(full_path),
                }

    file_size = os.path.getsize(zip_path)
    file_sha256 = compute_sha256(zip_path)
    size_mb = file_size / (1024 * 1024)
    print(f"Completed {zip_path}: {size_mb:.2f} MB (SHA-256: {file_sha256[:16]}...)")

    return {
        "file_name": os.path.basename(zip_path),
        "file_path": zip_path,
        "size_mb": round(size_mb, 2),
        "size_bytes": file_size,
        "sha256": file_sha256,
        "include_weights": include_weights,
        "packaged_file_count": len(manifest_entries),
    }


def main():
    parser = argparse.ArgumentParser(description="Package CDM-1 Release Archives")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("CDM_RELEASE_DIR", os.path.join(ROOT_DIR, "dist")),
        help="Directory where packaged archives will be written (default: dist/ or $CDM_RELEASE_DIR)",
    )
    parser.add_argument(
        "--copy-downloads",
        action="store_true",
        help="Also copy output archives to user's standard Downloads folder if on Windows/macOS",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    proj_zip = os.path.join(out_dir, "CDM-1-Project.zip")
    weights_zip = os.path.join(out_dir, "CDM-1-Complete-With-Weights.zip")

    m1 = build_zip(proj_zip, include_weights=False)
    m2 = build_zip(weights_zip, include_weights=True)

    release_manifest = {
        "release_version": "1.0.0",
        "packages": [m1, m2],
    }

    manifest_path = os.path.join(out_dir, "release_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(release_manifest, f, indent=2)
    print(f"Release manifest saved to: {manifest_path}")

    # Copy to workspace root for convenience
    shutil.copy2(proj_zip, os.path.join(ROOT_DIR, "CDM-1-Project.zip"))
    shutil.copy2(weights_zip, os.path.join(ROOT_DIR, "CDM-1-Complete-With-Weights.zip"))

    if args.copy_downloads:
        user_dl = os.path.expanduser("~/Downloads")
        if os.path.exists(user_dl):
            shutil.copy2(proj_zip, os.path.join(user_dl, "CDM-1-Project.zip"))
            shutil.copy2(weights_zip, os.path.join(user_dl, "CDM-1-Complete-With-Weights.zip"))
            print(f"Copied archives to user Downloads: {user_dl}")


if __name__ == "__main__":
    main()
