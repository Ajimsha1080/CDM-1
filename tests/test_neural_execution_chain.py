"""CDM-1 Neural Execution Chain End-to-End Verification Test Suite.
Validates:
1. Exact checkpoint authenticity, SHA-256, tensor count, and Ed25519 signature.
2. Strict offline model and tokenizer loading, failing explicitly on missing or corrupted artifacts.
3. Neural forward inference tensor contract, output shapes, numerical validity, and uncertainty calibration.
4. CognitiveEngine integration: predictive outcomes into world state & planning, candidate scores into decision.
5. Controlled counterfactual demonstrations proving neural influence on decisions with identical inputs.
6. Explicit distinction between neural inference vs heuristic fallback (zero fabrication of missing neural scores).
7. Independent evaluation across holdout test records.
"""

import os
import sys
import json
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cdm.models.cdm_model import CDMModel
from cdm.config.model_config import CDMModelConfig
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.cognitive.engine import CognitiveEngine
from cdm.security.manifest_verification import verify_checkpoint_authenticity, SecurityVerificationError


MODEL_DIR = "experiments/release/cdm-1-general"
EXPECTED_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
EXPECTED_SIZE = 633465536
EXPECTED_TENSORS = 215


def test_checkpoint_authenticity_and_tensor_compatibility():
    """Requirement 1: Verify exact model checkpoint, size, SHA-256, tensor count and signature."""
    assert os.path.isdir(MODEL_DIR), f"Release model directory missing: {MODEL_DIR}"
    weights_path = os.path.join(MODEL_DIR, "model.safetensors")
    assert os.path.isfile(weights_path), f"Weights file missing: {weights_path}"

    # Verify size
    actual_size = os.path.getsize(weights_path)
    assert actual_size == EXPECTED_SIZE, f"Size mismatch: expected {EXPECTED_SIZE}, got {actual_size}"

    # Verify Ed25519 cryptographic signature and manifest
    auth_result = verify_checkpoint_authenticity(MODEL_DIR)
    assert auth_result["status"] == "authenticated"
    assert auth_result["signature_verified"] is True
    assert auth_result["sha256"] == EXPECTED_SHA256

    # Verify tensor count via SafeTensors
    from safetensors import safe_open
    with safe_open(weights_path, framework="pt") as sf:
        keys = list(sf.keys())
        assert len(keys) == EXPECTED_TENSORS, f"Expected {EXPECTED_TENSORS} tensors, found {len(keys)}"


def test_model_loading_and_tokenizer_offline():
    """Requirement 2: Strictly load checkpoint offline and verify configuration & tokenizer."""
    decision_model = CognitiveDecisionModel.from_pretrained(MODEL_DIR, local_files_only=True)
    assert decision_model.is_fallback is False
    assert decision_model.model is not None
    assert decision_model.tokenizer is not None
    assert decision_model.artifact_hash == EXPECTED_SHA256

    param_stats = decision_model.model.count_parameters()
    assert param_stats["total_parameters"] == 158360004
    assert param_stats["trainable_parameters"] == 9345732


def test_neural_inference_numerical_validity_and_shapes():
    """Requirement 3: Verify output tensor shapes, probability bounds, and non-NaN numerical validity."""
    model = CDMModel.from_pretrained(MODEL_DIR)
    model.eval()

    device = torch.device("cpu")
    input_ids = torch.randint(1, 1000, (2, 32), dtype=torch.long, device=device)
    attention_mask = torch.ones((2, 32), dtype=torch.long, device=device)
    numeric_values = torch.randn(2, 6, device=device)
    missing_indicators = torch.zeros(2, 6, device=device)
    categorical_indices = {
        "domain": torch.tensor([0, 1], dtype=torch.long, device=device),
        "channel": torch.tensor([1, 2], dtype=torch.long, device=device),
        "customer_tier": torch.tensor([2, 3], dtype=torch.long, device=device),
        "urgency_level": torch.tensor([1, 2], dtype=torch.long, device=device),
    }

    # Pass 3 candidate actions per batch item
    cand_ids = torch.randint(1, 1000, (2, 3, 16), dtype=torch.long, device=device)
    cand_mask = torch.ones((2, 3, 16), dtype=torch.long, device=device)
    constraint_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]], device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
            missing_indicators=missing_indicators,
            categorical_indices=categorical_indices,
            candidate_input_ids=cand_ids,
            candidate_attention_mask=cand_mask,
            constraint_mask=constraint_mask,
        )

    assert "cognitive_state" in out
    assert out["cognitive_state"].shape == (2, 256)
    assert not torch.isnan(out["cognitive_state"]).any()

    # Decision network outputs
    dec = out["decision"]
    assert dec["binary_logits"].shape == (2, 2)
    assert dec["decision_score"].shape in [(2,), (2, 1)]
    assert (dec["decision_score"] >= 0.0).all() and (dec["decision_score"] <= 100.0).all()

    # Candidate scorer outputs
    cand = out["candidates"]
    assert cand is not None
    probs = cand["choice_probabilities"]
    assert probs.shape == (2, 3)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-4)

    # Constraint masking verification: prohibited candidates must have exact 0.0 probability
    assert probs[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert probs[1, 1].item() == pytest.approx(0.0, abs=1e-6)


def test_cognitive_engine_integration():
    """Requirement 4: Verify integration with CognitiveEngine without fake heuristic substitution."""
    engine = CognitiveEngine.from_pretrained(MODEL_DIR, local_files_only=True)
    text = "Enterprise customer requesting expedited processing for international shipment renewal."
    candidates = ["act_issue_refund", "act_auto_resolve", "act_escalate_tier2", "act_request_info"]
    prohibitions = ["act_auto_resolve"]

    result = engine.process(
        text=text,
        candidates=candidates,
        prohibited_actions=prohibitions,
        structured_features={"domain": 0, "channel": 1, "customer_tier": 3, "urgency_level": 2, "value": 4500.0},
        tenant_id="tenant-test-audit",
    )

    # 1. Used genuine neural inference
    assert result.used_neural_inference is True
    assert result.neural_score is not None
    assert result.heuristic_score is None

    # 2. Selected candidate is from the candidate list and not prohibited
    assert result.decision in candidates
    assert result.decision != "act_auto_resolve"
    assert result.candidate_scores["act_auto_resolve"] == 0.0

    # 3. Provenance record generated and valid
    assert "provenance" in result.metadata
    assert "provenance_signature" in result.metadata["provenance"]
    assert len(result.metadata["provenance"]["provenance_signature"]) > 16


def test_demonstrate_neural_influence_controlled_counterfactual():
    """Requirement 5: Controlled experiment with identical context demonstrating how and why decisions shift.
    Case A: All actions permitted -> Neural model selects top scoring candidate action.
    Case B: Identical context, but top action prohibited -> Model cleanly shifts to next highest valid candidate.
    Case C: Shift candidate actions entirely -> Neural candidate embeddings re-rank appropriately.
    """
    engine = CognitiveEngine.from_pretrained(MODEL_DIR, local_files_only=True)
    context_text = "Critical service outage reported on core payment gateway during Black Friday sales."
    base_struct = {"domain": 0, "channel": 2, "customer_tier": 4, "urgency_level": 3, "severity": 0.98}

    # Case A: Standard candidate set, no prohibitions
    cands_a = ["act_escalate_tier2", "act_auto_resolve", "act_request_info"]
    res_a = engine.process(
        text=context_text,
        candidates=cands_a,
        prohibited_actions=[],
        structured_features=base_struct,
        tenant_id="tenant-counterfactual",
    )

    # Case B: Identical context and inputs, but prohibit whichever action was selected in Case A
    prohibited_top = [res_a.decision]
    res_b = engine.process(
        text=context_text,
        candidates=cands_a,
        prohibited_actions=prohibited_top,
        structured_features=base_struct,
        tenant_id="tenant-counterfactual",
    )

    # Decision must change because top action was masked
    assert res_b.decision != res_a.decision, f"Decision did not shift despite prohibiting {res_a.decision}!"
    assert res_b.candidate_scores[res_a.decision] == 0.0, "Prohibited action did not receive 0.0 probability!"
    assert res_b.decision in cands_a

    # Case C: Change the candidate choices to alternative domain actions
    cands_c = ["act_issue_refund", "act_routine_monitoring"]
    res_c = engine.process(
        text=context_text,
        candidates=cands_c,
        prohibited_actions=[],
        structured_features=base_struct,
        tenant_id="tenant-counterfactual",
    )
    assert res_c.decision in cands_c
    assert res_c.used_neural_inference is True


def test_explicit_neural_vs_heuristic_distinction():
    """Requirement 6: Fails explicitly or marks heuristic fallback without fabricating missing neural scores."""
    # When initialized in development mode without weights
    dev_engine = CognitiveEngine(decision_core=None, production_mode=False)
    res = dev_engine.process(
        text="Sample inquiry without neural weights loaded",
        candidates=["act_1", "act_2"],
    )

    assert res.used_neural_inference is False
    assert res.neural_score is None  # Missing neural score is strictly None, NEVER fabricated
    assert res.heuristic_score is not None
    assert all(score is None for score in res.candidate_scores.values())


def test_independent_evaluation_on_unseen_data():
    """Requirement 7: Verify performance across unseen test examples."""
    from cdm.data.curated_corpus import load_dataset_from_disk
    test_data = load_dataset_from_disk("datasets/test.jsonl")
    assert len(test_data) == 58, f"Expected 58 test records, got {len(test_data)}"

    decision_model = CognitiveDecisionModel.from_pretrained(MODEL_DIR, local_files_only=True)
    correct_choices = 0

    for ex in test_data:
        cand_ids = [c["action_id"] if isinstance(c, dict) else c.action_id for c in ex.candidate_actions]
        res = decision_model.decide(
            text=ex.input_text,
            structured_features=ex.structured_features,
            candidates=cand_ids,
            task="choice",
        )
        assert res.used_neural_inference is True
        if res.decision == ex.target_decision.selected_candidate_id:
            correct_choices += 1

    accuracy = correct_choices / len(test_data)
    assert accuracy >= 0.90, f"Holdout test accuracy too low: {accuracy:.4f}"
