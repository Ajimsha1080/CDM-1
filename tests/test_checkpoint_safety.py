"""Tests for checkpoint safety, strict validation, and offline loading."""

import os
import tempfile
import json
import pytest
from cdm.models.cdm_model import CDMModel
from cdm.inference.pipeline import CognitiveDecisionModel


def test_missing_directory_raises_error():
    with pytest.raises(FileNotFoundError):
        CDMModel.from_pretrained("non_existent_directory_xyz")


def test_missing_weights_raises_filenotfound():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config.json but no model.safetensors
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": "CDM-1"}, f)

        with pytest.raises(FileNotFoundError, match="Missing model.safetensors"):
            CDMModel.from_pretrained(tmpdir)


def test_corrupt_empty_weights_raises_valueerror():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"model_name": "CDM-1"}, f)

        # Create 0-byte file
        with open(os.path.join(tmpdir, "model.safetensors"), "wb") as f:
            pass

        with pytest.raises(ValueError, match="is empty"):
            CDMModel.from_pretrained(tmpdir)


def test_trained_checkpoint_loads_strictly_offline():
    release_path = "experiments/release/cdm-1-general"
    assert os.path.exists(release_path), "Release checkpoint directory must exist!"

    model = CDMModel.from_pretrained(release_path, strict=True)
    counts = model.count_parameters()
    assert counts["total_parameters"] == 158_360_004

    # Verify offline inference engine loads without network
    runtime = CognitiveDecisionModel.from_pretrained(release_path, local_files_only=True)
    assert runtime.model is not None
    assert runtime.schema is not None


def test_missing_backbone_with_local_files_only_raises_clear_error():
    from cdm.models.language_encoder import LanguageEncoder
    from cdm.config.model_config import LanguageEncoderConfig

    cfg = LanguageEncoderConfig(
        model_name_or_path="nonexistent-org/missing-backbone-model-xyz",
        local_files_only=True,
    )
    with pytest.raises(RuntimeError, match="missing from local files/cache with local_files_only=True"):
        LanguageEncoder(cfg)


def test_software_only_mode_works_without_weights():
    """Verify software-only mode initializes and functions without neural weights."""
    with tempfile.TemporaryDirectory() as empty_dir:
        runtime = CognitiveDecisionModel.from_pretrained(
            empty_dir,
            allow_fallback=True,
            production_mode=False,
        )
        assert runtime.is_fallback is True
        assert runtime.loading_status["mode"] == "development_rule_fallback"

        # Verify inference execution without weights
        result = runtime.predict(
            text="Customer requests refund for damaged order",
            structured_features={"amount": 45.0, "risk_score": 0.1},
            task="binary",
        )
        assert result.decision in ("YES", "NO")
        assert result.used_neural_inference is False
        assert result.neural_score is None
        assert result.data["prediction_source"] == "rule_based_fallback"


def test_production_mode_cannot_silently_use_fallback():
    """Verify production mode strictly rejects silent fallbacks when weights are absent."""
    with tempfile.TemporaryDirectory() as empty_dir:
        # 1. Pipeline level check
        with pytest.raises(FileNotFoundError, match="Production mode strictly requires"):
            CognitiveDecisionModel.from_pretrained(
                empty_dir,
                allow_fallback=True,  # Overridden by production_mode=True
                production_mode=True,
            )

        # 2. Environment variable enforcement
        os.environ["CDM_PRODUCTION_MODE"] = "1"
        try:
            with pytest.raises(FileNotFoundError, match="Production mode strictly requires"):
                CognitiveDecisionModel.from_pretrained(
                    empty_dir,
                    allow_fallback=True,
                )
        finally:
            os.environ.pop("CDM_PRODUCTION_MODE", None)

        # 3. CognitiveEngine level check: strictly fails fast during initialization
        from cdm.cognitive.engine import CognitiveEngine
        with pytest.raises(RuntimeError, match="Production mode startup failure"):
            CognitiveEngine(decision_model=None, production_mode=True)


