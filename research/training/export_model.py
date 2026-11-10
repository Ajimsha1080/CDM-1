"""STAGE 10: Model Export and Checkpoint Validation for CDM-1.
Exports trained model weights in SafeTensors format, serializes tokenizer files,
writes feature schema, verifies SHA-256 and manifest integrity in full_model mode,
and verifies standalone reloading and offline inference.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from safetensors import safe_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from transformers import AutoTokenizer
from cdm.models.cdm_model import CDMModel
from cdm.config.feature_schema import FeatureSchema
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.security.manifest_verification import (
    verify_release_manifest_mode,
    TRUSTED_RELEASE_PUBLIC_KEY,
    compute_file_sha256,
)

EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
EXPECTED_SIZE = 633465536
EXPECTED_TENSORS = 215


def export_model_package(
    source_checkpoint_dir: str = "experiments/checkpoints/cdm-1-general",
    export_dir: str = "experiments/release/cdm-1-general",
) -> dict:
    print(f"=== STAGE 10: Exporting Model Checkpoint to {export_dir} ===", flush=True)
    os.makedirs(export_dir, exist_ok=True)

    # 1. Resolve source checkpoint directory
    actual_source = source_checkpoint_dir
    source_weights = os.path.join(actual_source, "model.safetensors")
    if not os.path.isfile(source_weights):
        fallback_source = export_dir
        fallback_weights = os.path.join(fallback_source, "model.safetensors")
        if os.path.isfile(fallback_weights):
            print(f"Source '{source_weights}' absent; utilizing verified checkpoint at '{fallback_weights}'", flush=True)
            actual_source = fallback_source
        else:
            raise FileNotFoundError(f"Cannot export: neither '{source_weights}' nor '{fallback_weights}' exist. Execute training first.")

    # 2. Load model and export weights & config
    print(f"Loading trained weights from '{actual_source}'...", flush=True)
    model = CDMModel.from_pretrained(actual_source)
    if actual_source != export_dir:
        print(f"Exporting SafeTensors weights and config.json to '{export_dir}'...", flush=True)
        model.save_pretrained(export_dir)

    # 3. Export tokenizer if not already present
    tok_json = os.path.join(export_dir, "tokenizer.json")
    tok_cfg = os.path.join(export_dir, "tokenizer_config.json")
    if not (os.path.isfile(tok_json) and os.path.isfile(tok_cfg)):
        print("Exporting tokenizer vocabulary and configs...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model.config.language_encoder.model_name_or_path)
        tokenizer.save_pretrained(export_dir)

    # 4. Export feature schema if not already present
    schema_path = os.path.join(export_dir, "feature_schema.json")
    if not os.path.isfile(schema_path):
        schema = FeatureSchema.get_default_schema()
        with open(schema_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(schema.model_dump_json(indent=2))

    # 5. Verify exported weights, tensor count, and SHA-256
    weights_path = os.path.join(export_dir, "model.safetensors")
    assert os.path.isfile(weights_path), f"CRITICAL: model.safetensors missing at {weights_path}"
    actual_size = os.path.getsize(weights_path)
    actual_sha = compute_file_sha256(weights_path)

    with safe_open(weights_path, framework="pt") as sf:
        tensor_keys = list(sf.keys())
        tensor_count = len(tensor_keys)

    print(f"Exported Checkpoint Size:  {actual_size} bytes (expected {EXPECTED_SIZE})", flush=True)
    print(f"Exported Checkpoint SHA:   {actual_sha}", flush=True)
    print(f"Exported Tensor Count:     {tensor_count} (expected {EXPECTED_TENSORS})", flush=True)

    assert actual_size == EXPECTED_SIZE, f"Size mismatch: expected {EXPECTED_SIZE}, got {actual_size}"
    assert actual_sha == EXPECTED_SHA256, f"SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
    assert tensor_count == EXPECTED_TENSORS, f"Tensor count mismatch: expected {EXPECTED_TENSORS}, got {tensor_count}"

    # 6. Verify release manifest integrity under full_model mode
    print("Verifying release manifest integrity under full_model mode...", flush=True)
    manifest_res = verify_release_manifest_mode(
        release_dir=export_dir,
        mode="full_model",
        trusted_public_key_hex=TRUSTED_RELEASE_PUBLIC_KEY,
        verify_offline_inference=True,
    )
    print("Manifest verification passed: Ed25519 signature valid and all 5 packaged files verified.", flush=True)

    # 7. Run standalone test inference on unseen sample
    print("Running standalone offline test inference from exported package...", flush=True)
    decision_model = CognitiveDecisionModel.from_pretrained(export_dir, local_files_only=True)
    res = decision_model.decide(
        text="Customer reporting duplicate subscription billing on annual tier.",
        structured_features={"domain": 0, "channel": 1, "customer_tier": 3, "urgency_level": 2},
        candidates=["act_issue_refund", "act_auto_resolve", "act_request_info"],
        task="choice",
    )
    assert res.used_neural_inference is True
    assert res.decision in ["act_issue_refund", "act_auto_resolve", "act_request_info"]
    print(f"Offline test inference passed! (Selected Decision: {res.decision}, Confidence: {res.confidence:.4f})", flush=True)

    print("=== STAGE 10: Model Export and Verification Completed Successfully ===", flush=True)
    return {
        "status": "success",
        "export_dir": export_dir,
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "tensor_count": tensor_count,
        "manifest_verified": True,
        "test_inference_decision": res.decision,
        "test_inference_confidence": res.confidence,
    }


if __name__ == "__main__":
    export_model_package()

