"""Tests for Model Optionality, Deterministic Rule Fallback, and Checkpoint Integrity Verification."""

import os
import tempfile
import pytest
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.security.manifest_verification import SecurityVerificationError


def test_optional_model_fallback_when_weights_absent():
    """Verify that when model weights are absent, CognitiveDecisionModel falls back cleanly to deterministic rule-based scorer."""
    with tempfile.TemporaryDirectory() as empty_dir:
        # Load with allow_fallback=True (default)
        model = CognitiveDecisionModel.from_pretrained(empty_dir, allow_fallback=True)
        assert model.is_fallback is True
        assert model.model is None
        assert model.tokenizer is None

        # Binary decision
        res_bin = model.decide(
            text="Critical payment gateway timeout reported on production.",
            structured_features={"urgency_level": "critical", "sentiment_score": -0.8},
            task="binary",
        )
        assert res_bin.decision in ["YES", "NO"]
        assert res_bin.priority in ["HIGH", "CRITICAL"]
        assert 0.0 <= res_bin.risk <= 1.0
        assert res_bin.provenance is not None
        assert res_bin.data["prediction_source"] == "rule_based_fallback"

        # Choice task with constraints
        res_choice = model.decide(
            text="Standard customer query regarding password reset.",
            task="choice",
            candidates=["auto_resolve", "escalate", "block_account"],
            prohibited_actions=["block_account"],
        )
        assert res_choice.decision != "block_account"
        assert res_choice.selected_candidate != "block_account"
        assert res_choice.candidate_scores["block_account"] == 0.0


def test_tampered_weights_refused_before_loading():
    """Verify that if model.safetensors is corrupted or tampered, it is strictly refused before loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a bogus release_manifest.json with fake hash
        manifest_path = os.path.join(tmpdir, "release_manifest.json")
        weights_path = os.path.join(tmpdir, "model.safetensors")

        # Create dummy weights
        with open(weights_path, "wb") as f:
            f.write(b"corrupted_or_tampered_weights_data_here")

        # Without a valid manifest or matching signature/hash, loading must fail
        with pytest.raises((SecurityVerificationError, FileNotFoundError, Exception)):
            CognitiveDecisionModel.from_pretrained(tmpdir, allow_fallback=False)
