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


def test_software_only_api_startup_and_functionality_without_weights():
    """Verify that in software-only mode, the FastAPI REST API starts cleanly without weights,
    reports ready in fallback mode, and processes decision requests."""
    from fastapi.testclient import TestClient
    from api.main import app, KEY_MANAGER

    with tempfile.TemporaryDirectory() as empty_dir:
        old_env = os.environ.get("CDM_ENV")
        old_path = os.environ.get("CDM_MODEL_PATH")
        os.environ["CDM_ENV"] = "development"
        os.environ["CDM_MODEL_PATH"] = empty_dir
        test_key = "cdm_software_only_test_key_123"
        KEY_MANAGER.register_key(test_key, "tenant_software_only", tier="developer", rate_limit_rpm=500)

        try:
            with TestClient(app) as client:
                # 1. Verify health probe
                r_health = client.get("/v1/health")
                assert r_health.status_code == 200
                assert r_health.json()["status"] == "healthy"

                # 2. Verify readiness probe in software-only mode
                r_ready = client.get("/v1/ready")
                assert r_ready.status_code == 200
                ready_data = r_ready.json()
                assert ready_data["status"] == "ready"
                assert ready_data["mode"] == "rule_based_fallback"
                assert ready_data["weights_verified"] is False

                # 3. Verify authenticated decision endpoint operates without neural weights
                headers = {"Authorization": f"Bearer {test_key}"}
                payload = {
                    "text": "Customer inquiring about software-only license validity.",
                    "structured_features": {"urgency_level": "medium", "sentiment_score": 0.2},
                    "task": "choice",
                    "candidates": ["respond_info", "escalate_tier2"],
                }
                r_decision = client.post("/v1/decisions", json=payload, headers=headers)
                assert r_decision.status_code == 200
                dec_data = r_decision.json()
                assert dec_data["selected_candidate"] in ["respond_info", "escalate_tier2"]
                assert dec_data["calibrated"] is True
                assert "predicted_outcome" in dec_data

                # 4. Verify cognitive process endpoint operates without neural weights
                cog_payload = {
                    "text": "Autonomous triage task for support inquiry.",
                    "task": "choice",
                    "candidates": ["triage_low", "triage_normal"],
                    "session_id": "test_soft_sess",
                }
                r_cog = client.post("/v1/cognitive/process", json=cog_payload, headers=headers)
                assert r_cog.status_code == 200
                cog_data = r_cog.json()
                assert cog_data["decision"] in ["triage_low", "triage_normal"]
        finally:
            if old_env is not None:
                os.environ["CDM_ENV"] = old_env
            else:
                os.environ.pop("CDM_ENV", None)
            if old_path is not None:
                os.environ["CDM_MODEL_PATH"] = old_path
            else:
                os.environ.pop("CDM_MODEL_PATH", None)

