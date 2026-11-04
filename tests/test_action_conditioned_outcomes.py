"""Priority 1: Action-Conditioned Outcome Prediction Test Suite.
Validates candidate-specific predictive representations, independent per-candidate
outcome predictions, sensitivity to candidate action changes, and gradient propagation.
"""

import pytest
import torch
import torch.nn.functional as F
from cdm.config.model_config import CDMModelConfig, PredictiveNetworkConfig
from cdm.models.predictive_network import PredictiveNetwork
from cdm.models.cdm_model import CDMModel


def test_action_conditioning_representation_changes():
    """Verify that different candidate actions produce distinct predictive representations."""
    input_dim = 64
    action_dim = 64
    config = PredictiveNetworkConfig(hidden_dim=32, num_outcome_classes=3)
    pred_net = PredictiveNetwork(input_dim=input_dim, action_dim=action_dim, config=config)
    pred_net.eval()

    batch_size = 1
    cognitive_state = torch.randn(batch_size, input_dim)

    # Two distinct candidate actions: Action A vs Action B
    cand_a = torch.randn(batch_size, 1, action_dim)
    cand_b = cand_a + torch.randn(batch_size, 1, action_dim) * 2.0

    with torch.no_grad():
        out_a = pred_net(cognitive_state, candidate_embeddings=cand_a)
        out_b = pred_net(cognitive_state, candidate_embeddings=cand_b)

    rep_a = out_a["candidate_representations"]
    rep_b = out_b["candidate_representations"]

    # Distance must be significant
    diff_norm = torch.norm(rep_a - rep_b).item()
    assert diff_norm > 1e-3, f"Candidate representations are too similar: diff={diff_norm}"

    # Outcome predictions must also differ
    bin_diff = torch.abs(out_a["candidate_binary_outcome_logits"] - out_b["candidate_binary_outcome_logits"]).item()
    num_diff = torch.abs(out_a["candidate_numeric_outcome"] - out_b["candidate_numeric_outcome"]).item()
    risk_diff = torch.abs(out_a["candidate_estimated_risk"] - out_b["candidate_estimated_risk"]).item()

    assert bin_diff > 1e-4 or num_diff > 1e-4 or risk_diff > 1e-4, "Candidate outcomes did not change with action change"


def test_candidate_conditioned_gradient_propagation():
    """Verify that training losses propagate gradients to candidate action projectors."""
    input_dim = 64
    action_dim = 64
    config = PredictiveNetworkConfig(hidden_dim=32, num_outcome_classes=3)
    pred_net = PredictiveNetwork(input_dim=input_dim, action_dim=action_dim, config=config)
    pred_net.train()

    cognitive_state = torch.randn(2, input_dim, requires_grad=True)
    candidate_embeddings = torch.randn(2, 3, action_dim, requires_grad=True)

    out = pred_net(cognitive_state, candidate_embeddings=candidate_embeddings)
    cand_bin_logits = out["candidate_binary_outcome_logits"].squeeze(-1)  # [2, 3]

    target = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = F.binary_cross_entropy_with_logits(cand_bin_logits, target)
    loss.backward()

    # Verify action projector received gradients
    proj_weight_grad = pred_net.action_projector[0].weight.grad
    assert proj_weight_grad is not None, "Action projector weight has no gradient!"
    assert proj_weight_grad.abs().sum().item() > 0.0, "Action projector gradient is zero!"

    # Verify candidate embeddings received gradients
    assert candidate_embeddings.grad is not None, "Candidate embeddings received no gradient!"
    assert candidate_embeddings.grad.abs().sum().item() > 0.0


def test_cdm_model_candidate_conditioned_loss():
    """Verify that CDMModel.compute_multitask_loss correctly computes candidate outcome loss."""
    cfg = CDMModelConfig()
    cfg.fusion.fusion_dim = 64
    cfg.language_encoder.projection_dim = 64
    cfg.structured_encoder.projection_dim = 64
    cfg.predictive.hidden_dim = 32

    model = CDMModel(cfg)
    model.train()

    batch_size = 2
    num_cands = 3
    cog_state = torch.randn(batch_size, 64)
    cand_embeddings = torch.randn(batch_size, num_cands, 64)

    outputs = {
        "cognitive_state": cog_state,
        "decision": {
            "binary_logits": torch.randn(batch_size, 2),
            "priority_logits": torch.randn(batch_size, 4),
            "risk_logits": torch.randn(batch_size, 4),
            "decision_score": torch.randn(batch_size, 1),
        },
        "predictive": {
            "binary_outcome_logits": torch.randn(batch_size, 1),
            "multiclass_outcome_logits": torch.randn(batch_size, 4),
            "numeric_outcome": torch.randn(batch_size, 1),
            "estimated_risk": torch.sigmoid(torch.randn(batch_size, 1)),
            "candidate_binary_outcome_logits": torch.randn(batch_size, num_cands, 1),
            "candidate_estimated_risk": torch.sigmoid(torch.randn(batch_size, num_cands, 1)),
            "candidate_numeric_outcome": torch.randn(batch_size, num_cands, 1),
        },
        "candidates": {
            "constrained_scores": torch.randn(batch_size, num_cands),
        },
    }

    targets = {
        "choice_target": torch.tensor([0, 1]),
        "choice_mask": torch.tensor([1.0, 1.0]),
        "outcome_binary_target": torch.tensor([1.0, 0.0]),
        "outcome_binary_mask": torch.tensor([1.0, 1.0]),
        "risk_prob_target": torch.tensor([0.2, 0.7]),
        "risk_prob_mask": torch.tensor([1.0, 1.0]),
    }

    loss, loss_dict = model.compute_multitask_loss(outputs, targets)
    assert "loss_cand_outcome_binary" in loss_dict
    assert "loss_cand_risk" in loss_dict
    assert loss_dict["loss_cand_outcome_binary"] > 0.0
    assert loss_dict["loss_cand_risk"] > 0.0
    assert not torch.isnan(loss)
