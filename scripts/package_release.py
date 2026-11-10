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
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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

    # P0: Strictly fail packaging if weights requested but checkpoint is absent
    if include_weights:
        rel_ckpt = os.path.join(ROOT_DIR, "experiments", "release", "cdm-1-general", "model.safetensors")
        chk_ckpt = os.path.join(ROOT_DIR, "experiments", "checkpoints", "cdm-1-general", "model.safetensors")
        if not (os.path.exists(rel_ckpt) or os.path.exists(chk_ckpt)):
            raise FileNotFoundError(
                "CRITICAL PACKAGING FAILURE: Cannot package weights archive. "
                "SafeTensors checkpoint missing from both experiments/release and experiments/checkpoints."
            )

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

                if file == "cognitive_validation_results.json":
                    with open(full_path, "r", encoding="utf-8") as f_in:
                        cog_data = json.load(f_in)
                    if not include_weights:
                        cog_data["model_artifacts"]["checkpoint_exists"] = False
                        cog_data["model_artifacts"]["checkpoint_size_bytes"] = 0
                        cog_data["model_artifacts"]["checkpoint_sha256"] = None
                        cog_data["model_artifacts"]["deterministic_inference_verified"] = False
                        cog_data["model_artifacts"]["distribution_mode"] = "software_only_without_weights"
                    else:
                        cog_data["model_artifacts"]["checkpoint_exists"] = True
                        cog_data["model_artifacts"]["distribution_mode"] = "full_model_with_weights"
                    payload_bytes = json.dumps(cog_data, indent=2).encode("utf-8")
                    zf.writestr(f"CDM-1/{rel_path}", payload_bytes)
                    manifest_entries[rel_path] = {"size_bytes": len(payload_bytes)}
                    continue

                zf.write(full_path, arcname=f"CDM-1/{rel_path}")
                manifest_entries[rel_path] = {
                    "size_bytes": os.path.getsize(full_path),
                }

    # Verify actual archive contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        if include_weights:
            has_weights = any(n.endswith("model.safetensors") for n in namelist)
            if not has_weights:
                raise RuntimeError(
                    f"CRITICAL PACKAGING INTEGRITY FAILURE: model.safetensors missing from generated archive: {zip_path}"
                )

    file_size = os.path.getsize(zip_path)
    file_sha256 = compute_sha256(zip_path)
    size_mb = file_size / (1024 * 1024)
    print(f"Completed {zip_path}: {size_mb:.2f} MB (SHA-256: {file_sha256[:16]}...)")

    # Use portable relative path instead of absolute local filesystem paths
    portable_rel_path = os.path.relpath(zip_path, ROOT_DIR).replace("\\", "/")

    return {
        "file_name": os.path.basename(zip_path),
        "file_path": portable_rel_path,
        "size_mb": round(size_mb, 2),
        "size_bytes": file_size,
        "sha256": file_sha256,
        "include_weights": include_weights,
        "packaged_file_count": len(manifest_entries),
    }


def verify_clean_extraction_and_inference(zip_path: str) -> None:
    """P0 Requirement 7: Extract the generated archive into a clean directory and run inference."""
    import tempfile
    print(f"Verifying clean extraction and offline inference from: {zip_path}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        extracted_root = os.path.join(tmpdir, "CDM-1")
        if extracted_root not in sys.path:
            sys.path.insert(0, extracted_root)
        model_dir = os.path.join(extracted_root, "experiments", "release", "cdm-1-general")
        if not os.path.exists(model_dir):
            model_dir = os.path.join(extracted_root, "experiments", "checkpoints", "cdm-1-general")

        weights_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Verification failure: model.safetensors missing in extracted directory: {model_dir}")

        # Run standalone offline inference using the packaged model
        from cdm.inference.pipeline import CognitiveDecisionModel
        offline_model = CognitiveDecisionModel.from_pretrained(model_dir, local_files_only=True)
        res = offline_model.decide(
            text="Emergency ICU triage: acute coronary syndrome presenting with diaphoresis and hypotension.",
            structured_features={"monetary_value_usd": 5000.0, "urgency_level": "critical", "domain": "customer_support"},
            task="binary",
        )
        assert res.decision in {"APPROVE", "REJECT", "YES", "NO", "ABSTAIN"}, f"Unexpected decision from extracted archive: {res.decision}"
        print(f"Verification successful: Clean extraction and offline inference passed! (Decision={res.decision}, Conf={res.confidence:.4f})")


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

    # Verify extraction portability and offline inference
    verify_clean_extraction_and_inference(weights_zip)

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
