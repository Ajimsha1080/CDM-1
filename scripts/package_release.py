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
    """Requirement 4 & 5: Extract the generated archive into a clean directory,
    recalculate checkpoint SHA-256 and size, confirm manifest match, verify full_model mode,
    and execute predictive, decision, candidate scoring, and CognitiveEngine integration.
    """
    import tempfile
    print(f"\n[*] Verifying clean extraction and offline inference from: {zip_path}...")

    EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
    EXPECTED_SIZE = 633465536

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            # 1. Verify ZIP contains model.safetensors
            matching = [n for n in namelist if n.endswith("experiments/release/cdm-1-general/model.safetensors")]
            if not matching:
                raise RuntimeError(f"CRITICAL: model.safetensors is missing from zip archive: {zip_path}")
            print(f"    [+] model.safetensors verified inside archive: {matching[0]}")
            zf.extractall(tmpdir)

        extracted_root = os.path.join(tmpdir, "CDM-1")
        if extracted_root not in sys.path:
            sys.path.insert(0, extracted_root)

        model_dir = os.path.join(extracted_root, "experiments", "release", "cdm-1-general")
        weights_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(weights_file):
            raise FileNotFoundError(f"Verification failure: model.safetensors missing in extracted directory: {model_dir}")

        # 2. Recalculate extracted checkpoint size and SHA-256
        extracted_size = os.path.getsize(weights_file)
        extracted_sha256 = compute_sha256(weights_file)
        print(f"    [+] Extracted model size:   {extracted_size} bytes (matches manifest: {extracted_size == EXPECTED_SIZE})")
        print(f"    [+] Extracted model SHA256: {extracted_sha256} (matches manifest: {extracted_sha256 == EXPECTED_SHA256})")

        assert extracted_size == EXPECTED_SIZE, f"Extracted size mismatch: expected {EXPECTED_SIZE}, got {extracted_size}"
        assert extracted_sha256 == EXPECTED_SHA256, f"Extracted SHA mismatch: expected {EXPECTED_SHA256}, got {extracted_sha256}"

        # 3. Verify release manifest under full_model mode
        from cdm.security.manifest_verification import verify_release_manifest_mode, TRUSTED_RELEASE_PUBLIC_KEY
        print("    [*] Verifying extracted release manifest in full_model mode...")
        verify_release_manifest_mode(
            release_dir=model_dir,
            mode="full_model",
            trusted_public_key_hex=TRUSTED_RELEASE_PUBLIC_KEY,
            verify_offline_inference=True,
        )
        print("    [+] Extracted release manifest authenticated with Ed25519 signature under full_model mode.")

        # 4. Actual inference validation: Predictive, Decision, Candidate Scoring, and CognitiveEngine
        print("    [*] Running end-to-end cognitive runtime validation on extracted release...")
        from cdm.cognitive.engine import CognitiveEngine
        from cdm.inference.pipeline import CognitiveDecisionModel

        # Standalone decision model test
        offline_model = CognitiveDecisionModel.from_pretrained(model_dir, local_files_only=True)
        res_standalone = offline_model.decide(
            text="High-value subscription renewal payment processing gateway timeout.",
            structured_features={"domain": 0, "channel": 1, "customer_tier": 4, "urgency_level": 3},
            candidates=["act_issue_refund", "act_auto_resolve", "act_escalate_tier2"],
            task="choice",
        )
        assert res_standalone.used_neural_inference is True
        print(f"    [+] Extracted standalone inference passed: selected={res_standalone.decision}, conf={res_standalone.confidence:.4f}")

        # CognitiveEngine integrated test
        engine = CognitiveEngine.from_pretrained(model_dir, local_files_only=True)
        cog_res = engine.process(
            text="High-value subscription renewal payment processing gateway timeout.",
            candidates=["act_issue_refund", "act_auto_resolve", "act_escalate_tier2"],
            prohibited_actions=["act_auto_resolve"],
            structured_features={"domain": 0, "channel": 1, "customer_tier": 4, "urgency_level": 3},
            tenant_id="tenant-extracted-verify",
        )
        assert cog_res.used_neural_inference is True
        assert cog_res.decision != "act_auto_resolve"
        assert cog_res.candidate_scores["act_auto_resolve"] == 0.0
        assert cog_res.neural_score is not None
        assert cog_res.heuristic_score is None
        print(f"    [+] Extracted CognitiveEngine integration passed: decision={cog_res.decision}, neural_score={cog_res.neural_score}, prohibited_score={cog_res.candidate_scores.get('act_auto_resolve')}")

    print(f"[*] Complete release packaging verification PASSED for: {zip_path}\n")


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
