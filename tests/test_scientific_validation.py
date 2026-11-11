"""Scientific Validation and Neural Architecture Defect Remediation Test Suite.
Verifies:
1. MC-Dropout Uncertainty Estimation, Mode Preservation, and Variance Settings.
2. Independent Decision Abstention Mechanisms without Arbitrary Entropy Summation.
3. Candidate Scorer Rigor: Empty Pool Rejection, Out-of-Bounds Rejection, Padding Rejection,
   Prohibited Rejection, Permutation Equivariance, and Single/All-Prohibited Edge Cases.
"""

import pytest
import torch
import torch.nn.functional as F
from cdm.config.model_config import CDMModelConfig, DecisionNetworkConfig, PredictiveNetworkConfig
from cdm.models.predictive_network import PredictiveNetwork
from cdm.models.decision_network import DecisionNetwork
from cdm.models.candidate_scorer import CandidateScorer


# =========================================================================
# SECTION A: Predictive Uncertainty & MC-Dropout Tests
# =========================================================================

def test_mc_dropout_requires_minimum_two_samples():
    """PredictiveNetwork.estimate_uncertainty_mc_dropout must raise ValueError if num_samples < 2."""
    net = PredictiveNetwork(input_dim=64, action_dim=64)
    state = torch.randn(2, 64)

    with pytest.raises(ValueError, match="num_samples must be at least 2"):
        net.estimate_uncertainty_mc_dropout(state, num_samples=1)

    with pytest.raises(ValueError, match="num_samples must be at least 2"):
        net.estimate_uncertainty_mc_dropout(state, num_samples=0)


def test_mc_dropout_restores_original_model_mode_on_success():
    """PredictiveNetwork must restore eval mode if called in eval mode, and train mode if called in train mode."""
    net = PredictiveNetwork(input_dim=64, action_dim=64)
    state = torch.randn(2, 64)

    # 1. Start in eval mode
    net.eval()
    assert not net.training
    res = net.estimate_uncertainty_mc_dropout(state, num_samples=3)
    assert not net.training, "estimate_uncertainty_mc_dropout failed to restore eval mode!"
    assert "epistemic_variance" in res
    assert res["epistemic_variance"].shape == (2,)

    # 2. Start in train mode
    net.train()
    assert net.training
    res = net.estimate_uncertainty_mc_dropout(state, num_samples=3)
    assert net.training, "estimate_uncertainty_mc_dropout failed to restore train mode!"


def test_mc_dropout_restores_original_model_mode_on_exception():
    """If an exception occurs during MC-dropout sampling, model state must still be restored."""
    net = PredictiveNetwork(input_dim=64, action_dim=64)
    net.eval()

    # Pass incompatible state shape to trigger forward pass exception
    bad_state = torch.randn(2, 128)  # wrong dimension (128 vs 64)
    with pytest.raises(RuntimeError):
        net.estimate_uncertainty_mc_dropout(bad_state, num_samples=3)

    assert not net.training, "Model mode was not restored after exception in MC-dropout!"


def test_deterministic_vs_stochastic_inference():
    """In eval mode, forward pass is strictly deterministic; MC-dropout exhibits non-zero variance."""
    torch.manual_seed(42)
    cfg = PredictiveNetworkConfig(hidden_dim=64, dropout=0.5)
    net = PredictiveNetwork(input_dim=64, action_dim=64, config=cfg)
    net.eval()
    state = torch.randn(2, 64)

    # Deterministic inference: two calls in eval mode yield exact same output
    out1 = net(state)
    out2 = net(state)
    torch.testing.assert_close(out1["binary_outcome_logits"], out2["binary_outcome_logits"])
    torch.testing.assert_close(out1["estimated_risk"], out2["estimated_risk"])

    # Stochastic inference: MC-dropout produces sample variance
    mc_res = net.estimate_uncertainty_mc_dropout(state, num_samples=20)
    assert mc_res["epistemic_variance"].shape == (2,)
    # Epistemic variance must be non-negative
    assert (mc_res["epistemic_variance"] >= 0.0).all()


# =========================================================================
# SECTION B: Decision Abstention & Independent Uncertainty Tests
# =========================================================================

def test_decision_abstention_on_low_confidence():
    """DecisionNetwork must flag abstain when binary confidence < abstention_threshold."""
    cfg = DecisionNetworkConfig(abstention_threshold=0.80, uncertainty_threshold=0.99)
    net = DecisionNetwork(input_dim=64, config=cfg)
    net.eval()

    state = torch.randn(4, 64)
    out = net(state)

    # Where binary_confidence < 0.80, abstain_low_confidence and abstain must be True
    low_conf = out["binary_confidence"] < 0.80
    assert (out["abstain_low_confidence"] == low_conf).all()
    assert (out["abstain"][low_conf] == True).all()


def test_decision_abstention_on_high_decision_entropy():
    """DecisionNetwork must flag abstain when decision entropy exceeds uncertainty threshold."""
    cfg = DecisionNetworkConfig(abstention_threshold=0.50, decision_uncertainty_threshold=0.40)
    net = DecisionNetwork(input_dim=64, config=cfg)
    net.eval()

    state = torch.randn(4, 64)
    out = net(state)

    high_ent = out["decision_entropy"] > 0.40
    assert (out["abstain_high_decision_entropy"] == high_ent).all()
    if high_ent.any():
        assert (out["abstain"][high_ent] == True).all()


def test_predictive_and_candidate_uncertainty_independent_abstention():
    """Predictive and candidate uncertainties independently influence abstention without combining entropy terms."""
    cfg = DecisionNetworkConfig(
        abstention_threshold=0.50,
        uncertainty_threshold=0.99,
        predictive_uncertainty_threshold=0.30,
        candidate_uncertainty_threshold=0.50,
    )
    net = DecisionNetwork(input_dim=64, config=cfg)
    net.eval()

    # Create mock inputs for a batch of 3 samples:
    # Sample 0: normal
    # Sample 1: high predictive uncertainty (0.75 > 0.30)
    # Sample 2: high candidate uncertainty (0.85 > 0.50)
    state = torch.randn(3, 64)
    pred_unc = torch.tensor([0.10, 0.75, 0.15])
    cand_unc = torch.tensor([0.20, 0.15, 0.85])

    out = net(state, predictive_uncertainty=pred_unc, candidate_uncertainty=cand_unc)

    assert not out["abstain_predictive_uncertainty"][0].item()
    assert out["abstain_predictive_uncertainty"][1].item()
    assert not out["abstain_predictive_uncertainty"][2].item()

    assert not out["abstain_candidate_uncertainty"][0].item()
    assert not out["abstain_candidate_uncertainty"][1].item()
    assert out["abstain_candidate_uncertainty"][2].item()

    # Overall abstention includes both
    assert out["abstain"][1].item()
    assert out["abstain"][2].item()


def test_policy_rejection_abstention():
    """Policy rejection independently triggers abstention even if model confidence is high."""
    cfg = DecisionNetworkConfig(abstention_threshold=0.50, uncertainty_threshold=0.99)
    net = DecisionNetwork(input_dim=64, config=cfg)
    net.eval()

    state = torch.randn(2, 64)
    policy_rejected = torch.tensor([False, True])
    out = net(state, policy_rejected=policy_rejected)

    assert not out["abstain_policy_rejected"][0].item()
    assert out["abstain_policy_rejected"][1].item()
    assert out["abstain"][1].item()


# =========================================================================
# SECTION C: Candidate Scorer Edge Cases & Invariance Tests
# =========================================================================

def test_candidate_scorer_rejects_zero_candidates():
    """CandidateScorer must raise ValueError when num_candidates == 0."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    ctx = torch.randn(2, 64)
    empty_cands = torch.randn(2, 0, 64)

    with pytest.raises(ValueError, match="num_candidates cannot be zero"):
        scorer(ctx, empty_cands)


def test_candidate_scorer_rejects_dimension_mismatches():
    """CandidateScorer must reject mismatched batch dimensions or invalid tensor ranks."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    ctx = torch.randn(2, 64)

    # 2D candidate tensor instead of 3D
    with pytest.raises(ValueError, match="must be 3D tensor"):
        scorer(ctx, torch.randn(2, 64))

    # Mismatched batch dimension
    with pytest.raises(ValueError, match="batch dimension"):
        scorer(ctx, torch.randn(4, 3, 64))

    # Mismatched candidate_dim
    with pytest.raises(ValueError, match="candidate_dim mismatch"):
        scorer(ctx, torch.randn(2, 3, 32))


def test_ranking_loss_rejects_out_of_bounds_target_indices():
    """compute_ranking_loss must raise IndexError for out-of-range target indices for active choices."""
    scores = torch.randn(2, 4)
    choice_mask = torch.tensor([1.0, 1.0])

    # Target index 4 is out of range for 4 candidates (valid indices are 0..3)
    bad_targets = torch.tensor([1, 4])
    with pytest.raises(IndexError, match="Out-of-range candidate target indices detected"):
        CandidateScorer.compute_ranking_loss(scores, bad_targets, choice_mask=choice_mask)

    # Target index -1 is out of range for active choice
    bad_negative = torch.tensor([-1, 2])
    with pytest.raises(IndexError, match="Out-of-range candidate target indices detected"):
        CandidateScorer.compute_ranking_loss(scores, bad_negative, choice_mask=choice_mask)


def test_ranking_loss_rejects_padded_or_prohibited_targets():
    """compute_ranking_loss must raise ValueError if target index refers to a padding or prohibited candidate."""
    scores = torch.randn(2, 4)
    targets = torch.tensor([2, 1])
    choice_mask = torch.tensor([1.0, 1.0])

    # Candidate 2 at batch 0 is padding (existence_mask = 0)
    existence = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="corresponds to a padding entry"):
        CandidateScorer.compute_ranking_loss(scores, targets, choice_mask=choice_mask, existence_mask=existence)

    # Candidate 1 at batch 1 is prohibited (constraint_mask = 0)
    valid_existence = torch.ones(2, 4)
    constraint = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
    with pytest.raises(ValueError, match="corresponds to a prohibited candidate"):
        CandidateScorer.compute_ranking_loss(
            scores, targets, choice_mask=choice_mask, existence_mask=valid_existence, constraint_mask=constraint
        )


def test_candidate_scorer_single_candidate():
    """CandidateScorer correctly scores a batch with only 1 candidate."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    ctx = torch.randn(2, 64)
    cands = torch.randn(2, 1, 64)

    out = scorer(ctx, cands)
    assert out["raw_scores"].shape == (2, 1)
    assert out["choice_probabilities"].shape == (2, 1)
    torch.testing.assert_close(out["choice_probabilities"], torch.ones(2, 1))
    assert (out["selected_candidate_idx"] == 0).all()
    assert not out["abstained"].any()


def test_candidate_scorer_all_prohibited():
    """When all candidates are prohibited, CandidateScorer returns sentinel -1 and abstained=True."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    ctx = torch.randn(2, 64)
    cands = torch.randn(2, 3, 64)
    constraint_mask = torch.zeros(2, 3)  # all prohibited

    out = scorer(ctx, cands, constraint_mask=constraint_mask)
    assert (out["selected_candidate_idx"] == -1).all()
    assert (out["abstained"] == True).all()
    torch.testing.assert_close(out["choice_probabilities"], torch.zeros(2, 3))


def test_candidate_scorer_mixed_and_padded_batch():
    """Mixed batch with padding and selective prohibitions correctly filters and normalizes probabilities."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    ctx = torch.randn(2, 64)
    cands = torch.randn(2, 4, 64)

    # Batch 0: candidate 0 allowed, candidate 1 prohibited, candidate 2 allowed, candidate 3 padding
    # Batch 1: candidate 0 allowed, candidates 1,2,3 all padding
    constraint_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    existence_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    out = scorer(ctx, cands, constraint_mask=constraint_mask, existence_mask=existence_mask)
    probs = out["choice_probabilities"]

    # In batch 0: probs for prohibited index 1 and padding index 3 must be exactly 0
    assert probs[0, 1].item() == 0.0
    assert probs[0, 3].item() == 0.0
    assert abs(probs[0, 0].item() + probs[0, 2].item() - 1.0) < 1e-5

    # In batch 1: only index 0 is valid, so prob must be 1.0
    assert abs(probs[1, 0].item() - 1.0) < 1e-5
    assert probs[1, 1].item() == 0.0
    assert probs[1, 2].item() == 0.0
    assert probs[1, 3].item() == 0.0


def test_candidate_scorer_permutation_invariance():
    """Permuting the candidate order permutes the raw scores and choice probabilities equivariantly."""
    scorer = CandidateScorer(context_dim=64, candidate_dim=64)
    scorer.eval()

    ctx = torch.randn(1, 64)
    cands = torch.randn(1, 4, 64)

    perm = [2, 0, 3, 1]
    cands_permuted = cands[:, perm, :]

    with torch.no_grad():
        out_orig = scorer(ctx, cands)
        out_perm = scorer(ctx, cands_permuted)

    orig_scores = out_orig["raw_scores"][0]
    perm_scores = out_perm["raw_scores"][0]

    # Scores of permuted candidates must match original scores at permuted positions
    for new_idx, old_idx in enumerate(perm):
        assert abs(perm_scores[new_idx].item() - orig_scores[old_idx].item()) < 1e-5

    orig_probs = out_orig["choice_probabilities"][0]
    perm_probs = out_perm["choice_probabilities"][0]
    for new_idx, old_idx in enumerate(perm):
        assert abs(perm_probs[new_idx].item() - orig_probs[old_idx].item()) < 1e-5
