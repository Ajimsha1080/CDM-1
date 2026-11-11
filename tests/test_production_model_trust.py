"""Unit & Integration Tests for Production Model Trust (Gate 1).
Validates:
1. Production mode rejects directly supplied unverified model objects.
2. Rejection of randomly initialized in-memory model instances.
3. Rejection of checkpoints with invalid or tampered SHA-256 hashes.
4. Independent verification by CognitiveEngine.verify_model_trust.
5. Acceptance of genuine authenticated release checkpoints in production.
"""

import os
import pytest
import torch

from cdm.cognitive.engine import CognitiveEngine
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.models.cdm_model import CDMModel
from cdm.config.model_config import CDMModelConfig


def test_production_mode_rejects_none_model():
    """Verify production mode rejects None decision model."""
    with pytest.raises(RuntimeError, match="Production model trust failure"):
        CognitiveEngine(decision_model=None, production_mode=True)


def test_production_mode_rejects_fallback_scorer():
    """Verify production mode rejects fallback rule scorer."""
    fallback_model = CognitiveDecisionModel(model=None, is_fallback=True)
    with pytest.raises(RuntimeError, match="Fallback rule scorer cannot be used in production mode"):
        CognitiveEngine(decision_model=fallback_model, production_mode=True)


def test_production_mode_rejects_direct_model_injection():
    """Verify production mode rejects directly injected, unverified model instances."""
    # Create an unverified in-memory model without cryptographic checkpoint loading
    cfg = CDMModelConfig()
    cfg.language_encoder.local_files_only = True
    raw_model = CDMModel(cfg)
    injected_cdm = CognitiveDecisionModel(model=raw_model, is_fallback=False)

    # Directly supplied instance has authenticated == False
    assert injected_cdm.loading_status.get("authenticated") is False

    with pytest.raises(RuntimeError, match="Directly supplied unverified model object rejected"):
        CognitiveEngine(decision_model=injected_cdm, production_mode=True)


def test_production_mode_rejects_random_initialization():
    """Verify production mode rejects randomly initialized model with spoofed status."""
    cfg = CDMModelConfig()
    cfg.language_encoder.local_files_only = True
    raw_model = CDMModel(cfg)
    spoofed_cdm = CognitiveDecisionModel(model=raw_model, is_fallback=False)
    # Spoof status dictionary but with bad hash
    spoofed_cdm.loading_status = {
        "loaded": True,
        "authenticated": True,
        "signature_verified": True,
        "artifact_hash": "deadbeef" * 8,
    }

    with pytest.raises(RuntimeError, match="does not match expected release hash"):
        CognitiveEngine(decision_model=spoofed_cdm, production_mode=True)


def test_production_mode_setter_enforces_trust():
    """Verify runtime property setter enforces production model trust."""
    # Start engine in dev mode, enable prod, attempt to set unverified model
    engine = CognitiveEngine(production_mode=False)
    engine.production_mode = True

    cfg = CDMModelConfig()
    cfg.language_encoder.local_files_only = True
    raw_model = CDMModel(cfg)
    unverified_cdm = CognitiveDecisionModel(model=raw_model, is_fallback=False)

    with pytest.raises(RuntimeError, match="Directly supplied unverified model object rejected"):
        engine.decision_model = unverified_cdm


def test_production_mode_accepts_genuine_checkpoint():
    """Verify production mode successfully initializes with authenticated genuine checkpoint."""
    checkpoint_dir = "experiments/release/cdm-1-general"
    if not os.path.exists(checkpoint_dir):
        pytest.skip("Genuine release checkpoint directory not found.")

    engine = CognitiveEngine.from_pretrained(
        model_name_or_path=checkpoint_dir,
        production_mode=True,
    )
    health = engine.get_health_status()
    assert health["production_mode"] is True
    assert health["status"] == "healthy"
    assert health["model"]["authenticated"] is True
    assert health["model"]["signature_verified"] is True
    assert health["model"]["artifact_hash"] == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
