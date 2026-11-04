"""Tests for candidate-conditioned neural training and parameter gradient updates."""

import pytest
import torch
import torch.nn.functional as F
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel


def test_candidate_constraint_masking_zero_probability():
    """Verify that prohibited candidates receive strictly zero selection probability."""
    model = CDMModel(CDMModelConfig())
    model.eval()

    batch_size = 2
    num_cands = 4
    dim = 256

    ctx = torch.randn(batch_size, dim)
    cand_embs = torch.randn(batch_size, num_cands, dim)

    # Prohibit candidate 2 for sample 0, and candidate 1 for sample 1
    constraint_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    existence_mask = torch.ones(batch_size, num_cands)

    out = model.candidate_scorer(
        cognitive_context=ctx,
        candidate_embeddings=cand_embs,
        constraint_mask=constraint_mask,
        existence_mask=existence_mask,
    )

    probs = out["choice_probabilities"]
    assert probs[0, 2].item() == 0.0, "Prohibited candidate must have 0 selection probability!"
    assert probs[1, 1].item() == 0.0, "Prohibited candidate must have 0 selection probability!"
    assert out["selected_candidate_idx"][0].item() != 2
    assert out["selected_candidate_idx"][1].item() != 1


def test_candidate_scorer_receives_gradients_and_updates():
    """Prove that candidate scorer parameters actually update during backprop training."""
    model = CDMModel(CDMModelConfig())
    model.train()

    # Isolate candidate scorer parameters
    scorer_params = list(model.candidate_scorer.parameters())
    optimizer = torch.optim.SGD(scorer_params, lr=0.1)

    initial_weight = model.candidate_scorer.scorer_mlp[0].weight.clone().detach()

    batch_size = 4
    num_cands = 3
    dim = 256

    ctx = torch.randn(batch_size, dim, requires_grad=True)
    cand_embs = torch.randn(batch_size, num_cands, dim, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    out = model.candidate_scorer(
        cognitive_context=ctx,
        candidate_embeddings=cand_embs,
    )

    loss = F.cross_entropy(out["constrained_scores"], targets)
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients are non-zero
    assert model.candidate_scorer.scorer_mlp[0].weight.grad is not None
    grad_norm = torch.norm(model.candidate_scorer.scorer_mlp[0].weight.grad).item()
    assert grad_norm > 0.0, "Candidate scorer gradients must not be zero!"

    optimizer.step()
    updated_weight = model.candidate_scorer.scorer_mlp[0].weight.clone().detach()
    diff = torch.norm(updated_weight - initial_weight).item()

    assert diff > 1e-4, f"Candidate scorer parameters failed to update (delta={diff})!"
