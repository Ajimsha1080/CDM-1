"""Unit tests for CDM-1 Neural Architecture Components.
Validates tensor shapes, forward execution, loss computation, and parameter counting.
"""

import pytest
import torch
from cdm.config.model_config import (
    CDMModelConfig,
    LanguageEncoderConfig,
    StructuredEncoderConfig,
    FusionConfig,
    PredictiveNetworkConfig,
    DecisionNetworkConfig,
)
from cdm.models.structured_encoder import StructuredFeatureEncoder
from cdm.models.fusion import CognitiveFusion, GatedFusion, CrossAttentionFusion
from cdm.models.predictive_network import PredictiveNetwork
from cdm.models.decision_network import DecisionNetwork
from cdm.models.candidate_scorer import CandidateScorer
from cdm.models.cdm_model import CDMModel


def test_structured_feature_encoder():
    cfg = StructuredEncoderConfig(
        num_numeric_features=6,
        categorical_cardinalities={"domain": 8, "channel": 6, "customer_tier": 5, "urgency_level": 4},
        projection_dim=256,
    )
    encoder = StructuredFeatureEncoder(cfg)

    batch_size = 4
    numeric = torch.randn(batch_size, 6)
    missing = torch.zeros(batch_size, 6)
    missing[0, 1] = 1.0  # simulate missing feature
    cat_indices = {
        "domain": torch.tensor([1, 2, 0, 4]),
        "channel": torch.tensor([1, 1, 3, 2]),
        "customer_tier": torch.tensor([2, 1, 0, 3]),
        "urgency_level": torch.tensor([0, 1, 2, 3]),
    }

    out = encoder(numeric, missing, cat_indices)
    assert out.shape == (batch_size, 256)
    assert not torch.isnan(out).any()


def test_fusion_mechanisms():
    batch_size = 4
    dim = 256
    text_emb = torch.randn(batch_size, dim)
    struct_emb = torch.randn(batch_size, dim)

    # Gated fusion
    cfg_gated = FusionConfig(fusion_type="gated", fusion_dim=dim)
    fusion_gated = CognitiveFusion(cfg_gated)
    out_gated = fusion_gated(text_emb, struct_emb)
    assert out_gated.shape == (batch_size, dim)

    # Cross attention fusion
    cfg_ca = FusionConfig(fusion_type="cross_attention", fusion_dim=dim, num_attention_heads=4)
    fusion_ca = CognitiveFusion(cfg_ca)
    out_ca = fusion_ca(text_emb, struct_emb)
    assert out_ca.shape == (batch_size, dim)


def test_predictive_network():
    batch_size = 4
    dim = 256
    net = PredictiveNetwork(input_dim=dim, action_dim=dim)
    state = torch.randn(batch_size, dim)
    action = torch.randn(batch_size, dim)

    out = net(state, action)
    assert out["binary_outcome_logits"].shape == (batch_size, 1)
    assert out["multiclass_outcome_logits"].shape == (batch_size, 4)
    assert out["numeric_outcome"].shape == (batch_size, 1)
    assert out["estimated_risk"].shape == (batch_size, 1)
    assert out["outcome_entropy"].shape == (batch_size,)


def test_decision_network_and_abstention():
    batch_size = 4
    dim = 256
    cfg = DecisionNetworkConfig(abstention_threshold=0.6, uncertainty_threshold=0.5)
    dec = DecisionNetwork(input_dim=dim, config=cfg)
    state = torch.randn(batch_size, dim)

    out = dec(state)
    assert out["binary_logits"].shape == (batch_size, 2)
    assert out["binary_probabilities"].shape == (batch_size, 2)
    assert out["decision_score"].shape == (batch_size,)
    assert out["priority_logits"].shape == (batch_size, 4)
    assert out["risk_logits"].shape == (batch_size, 4)
    assert out["abstain"].shape == (batch_size,)


def test_candidate_scorer_and_constraints():
    batch_size = 2
    num_cand = 3
    dim = 256
    scorer = CandidateScorer(context_dim=dim, candidate_dim=dim)
    ctx = torch.randn(batch_size, dim)
    cands = torch.randn(batch_size, num_cand, dim)

    # Prohibit candidate 2 for sample 0
    constraint_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    out = scorer(ctx, cands, constraint_mask=constraint_mask)
    assert out["raw_scores"].shape == (batch_size, num_cand)
    assert out["choice_probabilities"].shape == (batch_size, num_cand)
    # Check that prohibited candidate has 0 probability
    assert out["choice_probabilities"][0, 2].item() < 1e-5


def test_full_model_parameter_accounting_and_loss():
    # Use small encoder config for lightning-fast unit test
    model_cfg = CDMModelConfig()
    model = CDMModel(model_cfg)

    param_counts = model.count_parameters()
    assert param_counts["total_parameters"] > 140_000_000
    assert "trainable_parameters" in param_counts
    assert "frozen_parameters" in param_counts

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    numeric = torch.randn(batch_size, 6)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        numeric_values=numeric,
    )

    assert "cognitive_state" in outputs
    assert "decision" in outputs
    assert "predictive" in outputs

    # Test multi-task loss calculation
    targets = {
        "binary_target": torch.tensor([1, 0]),
        "priority_target": torch.tensor([2, 1]),
        "score_target": torch.tensor([85.0, 42.0]),
        "outcome_binary_target": torch.tensor([1.0, 0.0]),
        "outcome_multi_target": torch.tensor([0, 2]),
        "outcome_numeric_target": torch.tensor([3.5, 12.0]),
    }

    loss, loss_dict = model.compute_multitask_loss(outputs, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
