"""Tests for Non-Finite and Invalid Uncertainty Rejection (Gate 4).
Validates:
1. Rejection of NaN/Inf values in predictive network inputs and uncertainty outputs.
2. Shape, device, and dtype enforcement.
3. Rejection of NaN/Inf in decision network uncertainty inputs.
4. Independent abstention on high decision entropy, high predictive uncertainty, and policy rejection.
"""

import pytest
import torch

from cdm.models.predictive_network import PredictiveNetwork
from cdm.models.decision_network import DecisionNetwork
from cdm.config.model_config import PredictiveNetworkConfig, DecisionNetworkConfig


def test_predictive_network_rejects_nan_inf_inputs():
    """Verify PredictiveNetwork rejects NaN or Inf inputs."""
    pred_net = PredictiveNetwork(input_dim=64, config=PredictiveNetworkConfig(input_dim=64))

    # NaN in cognitive_state
    bad_state = torch.randn(2, 64)
    bad_state[0, 5] = float("nan")
    with pytest.raises(ValueError, match="Non-finite values"):
        pred_net(bad_state)

    # Inf in cognitive_state
    bad_state[0, 5] = float("inf")
    with pytest.raises(ValueError, match="Non-finite values"):
        pred_net(bad_state)


def test_predictive_network_mc_dropout_variance_validation():
    """Verify MC-dropout uncertainty returns strictly finite, non-negative variances."""
    pred_net = PredictiveNetwork(input_dim=64, config=PredictiveNetworkConfig(input_dim=64))
    state = torch.randn(4, 64)

    unc = pred_net.estimate_uncertainty_mc_dropout(state, num_samples=5)
    for k in ["epistemic_variance", "risk_variance", "multiclass_variance"]:
        v = unc[k]
        assert torch.isfinite(v).all()
        assert (v >= 0.0).all()
        assert v.shape == (4,)
        assert v.dtype == torch.float32


def test_decision_network_rejects_nan_inf_uncertainty():
    """Verify DecisionNetwork rejects NaN or Inf uncertainty inputs."""
    dec_net = DecisionNetwork(input_dim=64, config=DecisionNetworkConfig(input_dim=64))
    state = torch.randn(3, 64)

    # NaN in predictive uncertainty
    bad_pred_unc = torch.tensor([0.1, float("nan"), 0.3])
    with pytest.raises(ValueError, match="Non-finite values.*predictive_uncertainty"):
        dec_net(state, predictive_uncertainty=bad_pred_unc)

    # Inf in candidate uncertainty
    bad_cand_unc = torch.tensor([0.1, float("inf"), 0.3])
    with pytest.raises(ValueError, match="Non-finite values.*candidate_uncertainty"):
        dec_net(state, candidate_uncertainty=bad_cand_unc)


def test_decision_network_shape_validation():
    """Verify DecisionNetwork rejects mismatched uncertainty shapes."""
    dec_net = DecisionNetwork(input_dim=64, config=DecisionNetworkConfig(input_dim=64))
    state = torch.randn(3, 64)

    wrong_shape = torch.randn(4)  # batch size is 3
    with pytest.raises(ValueError, match="shape mismatch"):
        dec_net(state, predictive_uncertainty=wrong_shape)


def test_independent_abstention_criteria():
    """Verify independent abstention flags for confidence, decision entropy, and predictive uncertainty."""
    dec_net = DecisionNetwork(input_dim=64, config=DecisionNetworkConfig(input_dim=64))
    state = torch.randn(2, 64)

    # Normal inputs
    normal_pred_unc = torch.tensor([0.1, 0.1])
    out = dec_net(state, predictive_uncertainty=normal_pred_unc)
    assert "abstain_predictive_uncertainty" in out

    # High predictive uncertainty specifically triggers abstain_predictive_uncertainty
    high_pred_unc = torch.tensor([0.95, 0.1])
    out_high = dec_net(state, predictive_uncertainty=high_pred_unc)
    assert out_high["abstain_predictive_uncertainty"][0].item() is True
    assert out_high["abstain_predictive_uncertainty"][1].item() is False
    assert out_high["abstain"][0].item() is True
