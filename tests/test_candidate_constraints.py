"""Priority 2: Candidate Constraint Safety Test Suite.
Validates robust handling of all-prohibited edge cases, partial prohibitions,
all-valid sets, padded candidates, and explicit pipeline abstention states.
"""

import pytest
import torch
from cdm.models.candidate_scorer import CandidateScorer
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel
from transformers import AutoTokenizer


def test_all_prohibited_candidates():
    scorer = CandidateScorer(context_dim=64, candidate_dim=64, hidden_dim=32)
    scorer.eval()

    batch_size = 2
    num_cands = 4
    context = torch.randn(batch_size, 64)
    cands = torch.randn(batch_size, num_cands, 64)

    # All candidates prohibited
    all_zero_mask = torch.zeros(batch_size, num_cands)

    with torch.no_grad():
        out = scorer(
            cognitive_context=context,
            candidate_embeddings=cands,
            constraint_mask=all_zero_mask,
        )

    # 1. Must detect all_prohibited and abstained
    assert out["abstained"].tolist() == [True, True]
    assert out["all_prohibited"].tolist() == [True, True]

    # 2. Selected candidate index must be sentinel -1, NEVER a prohibited candidate index (0, 1, 2, 3)
    assert out["selected_candidate_idx"].tolist() == [-1, -1]

    # 3. Choice probabilities must be strictly 0.0 (no prohibited candidate gets positive probability)
    assert not torch.isnan(out["choice_probabilities"]).any()
    assert torch.all(out["choice_probabilities"] == 0.0)


def test_all_valid_candidates():
    scorer = CandidateScorer(context_dim=64, candidate_dim=64, hidden_dim=32)
    scorer.eval()

    batch_size = 2
    num_cands = 4
    context = torch.randn(batch_size, 64)
    cands = torch.randn(batch_size, num_cands, 64)
    all_valid_mask = torch.ones(batch_size, num_cands)

    with torch.no_grad():
        out = scorer(
            cognitive_context=context,
            candidate_embeddings=cands,
            constraint_mask=all_valid_mask,
        )

    assert out["abstained"].tolist() == [False, False]
    assert (out["selected_candidate_idx"] >= 0).all()
    assert (out["selected_candidate_idx"] < num_cands).all()

    probs = out["choice_probabilities"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_partially_prohibited_candidates():
    scorer = CandidateScorer(context_dim=64, candidate_dim=64, hidden_dim=32)
    scorer.eval()

    batch_size = 1
    num_cands = 4
    context = torch.randn(batch_size, 64)
    cands = torch.randn(batch_size, num_cands, 64)

    # Candidates 0 and 2 allowed, 1 and 3 prohibited
    partial_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    with torch.no_grad():
        out = scorer(
            cognitive_context=context,
            candidate_embeddings=cands,
            constraint_mask=partial_mask,
        )

    assert out["abstained"].item() is False
    sel_idx = out["selected_candidate_idx"].item()
    # Selected must be either 0 or 2, NEVER 1 or 3
    assert sel_idx in [0, 2]

    probs = out["choice_probabilities"][0]
    # Prohibited indices must have exactly 0 probability
    assert probs[1].item() == 0.0
    assert probs[3].item() == 0.0
    assert abs(probs[0].item() + probs[2].item() - 1.0) < 1e-5


def test_padded_candidates():
    scorer = CandidateScorer(context_dim=64, candidate_dim=64, hidden_dim=32)
    scorer.eval()

    batch_size = 1
    num_cands = 4
    context = torch.randn(batch_size, 64)
    cands = torch.randn(batch_size, num_cands, 64)

    # 2 valid candidates, 2 padded candidates
    existence_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    with torch.no_grad():
        out = scorer(
            cognitive_context=context,
            candidate_embeddings=cands,
            existence_mask=existence_mask,
        )

    sel_idx = out["selected_candidate_idx"].item()
    assert sel_idx in [0, 1]

    probs = out["choice_probabilities"][0]
    assert probs[2].item() == 0.0
    assert probs[3].item() == 0.0
    assert abs(probs[0].item() + probs[1].item() - 1.0) < 1e-5
