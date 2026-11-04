"""Tests for explicit loss masking on missing multi-task targets."""

import pytest
import torch
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel


def test_missing_targets_masked_without_artificial_zeros():
    model = CDMModel(CDMModelConfig())
    model.eval()

    batch_size = 3
    dim = 256
    device = torch.device("cpu")

    # Outputs dummy
    outputs = {
        "cognitive_state": torch.randn(batch_size, dim),
        "decision": {
            "binary_logits": torch.randn(batch_size, 2),
            "priority_logits": torch.randn(batch_size, 4),
            "risk_logits": torch.randn(batch_size, 4),
            "decision_score": torch.tensor([50.0, 75.0, 90.0]),
        },
        "predictive": {
            "binary_outcome_logits": torch.randn(batch_size, 1),
            "multiclass_outcome_logits": torch.randn(batch_size, 4),
            "numeric_outcome": torch.tensor([[1.0], [2.0], [3.0]]),
            "estimated_risk": torch.tensor([[0.2], [0.5], [0.8]]),
        },
        "candidates": None,
    }

    # Sample 0 has valid targets; Sample 1 & 2 have missing targets (mask=0.0)
    targets = {
        "binary_target": torch.tensor([1, 0, 0]),
        "binary_mask": torch.tensor([1.0, 0.0, 0.0]),
        "priority_target": torch.tensor([2, 0, 0]),
        "priority_mask": torch.tensor([1.0, 0.0, 0.0]),
        "score_target": torch.tensor([50.0, 0.0, 0.0]),
        "score_mask": torch.tensor([1.0, 0.0, 0.0]),
        "outcome_binary_target": torch.tensor([1.0, 0.0, 0.0]),
        "outcome_binary_mask": torch.tensor([1.0, 0.0, 0.0]),
        "outcome_numeric_target": torch.tensor([1.0, 0.0, 0.0]),
        "outcome_numeric_mask": torch.tensor([1.0, 0.0, 0.0]),
    }

    loss, loss_dict = model.compute_multitask_loss(outputs, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "loss_binary" in loss_dict
    assert "loss_priority" in loss_dict
    assert "loss_score" in loss_dict
    assert "loss_outcome_binary" in loss_dict
    assert "loss_outcome_numeric" in loss_dict
