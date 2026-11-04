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
