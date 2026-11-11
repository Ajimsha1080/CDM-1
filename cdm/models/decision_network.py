"""Component E: Decision Network for CDM-1
Houses independent output decision heads: Binary (YES/NO), Continuous Scoring,
Priority Classification, Risk Categorization, Calibrated Confidence, and Abstention Logic.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from cdm.config.model_config import DecisionNetworkConfig


class DecisionNetwork(nn.Module):
    """Primary decision module hosting independent operational decision heads

    with explicit calibration scaling and abstention gating.
    """

    def __init__(self, input_dim: int = 256, config: Optional[DecisionNetworkConfig] = None):
        super().__init__()
        self.config = config or DecisionNetworkConfig()

        # Shared decision representation layer
        self.decision_trunk = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Head 1: Binary Decision (YES / NO)
        self.binary_head = nn.Linear(input_dim, 2)

        # Head 3: Scoring Decision (e.g. 0 to 100 continuous score)
        self.score_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # maps to (0, 1), then scaled to score_range
        )

        # Head 4: Priority Decision (LOW, MEDIUM, HIGH, CRITICAL)
        self.priority_head = nn.Linear(input_dim, self.config.num_priority_classes)

        # Head 5: Risk Decision (Categorical risk levels)
        self.risk_head = nn.Linear(input_dim, self.config.num_risk_classes)

        # Head 6: Temperature parameter for Platt / Temperature scaling calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(
        self,
        cognitive_state: torch.Tensor,
        predictive_uncertainty: Optional[torch.Tensor] = None,
        candidate_uncertainty: Optional[torch.Tensor] = None,
        policy_rejected: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Args:
            cognitive_state: [batch_size, input_dim]
            predictive_uncertainty: [batch_size] optional predictive outcome entropy or epistemic variance
            candidate_uncertainty: [batch_size] optional candidate choice entropy
            policy_rejected: [batch_size] boolean tensor indicating hard policy violation

        Returns:
            Dictionary with logits, calibrated probabilities, continuous score,
            priority, risk, decoupled uncertainties, and individual/combined abstention flags.
        """
        batch_size = cognitive_state.shape[0]
        device = cognitive_state.device

        rep = self.decision_trunk(cognitive_state)

        # Binary decision
        raw_binary_logits = self.binary_head(rep)  # [batch_size, 2]
        # Calibrated with learned temperature
        temp = torch.clamp(self.temperature, min=0.05, max=10.0)
        calibrated_binary_logits = raw_binary_logits / temp
        binary_probs = F.softmax(calibrated_binary_logits, dim=-1)  # [batch_size, 2]

        # Scoring decision: scaled to configured [min_score, max_score]
        min_s, max_s = self.config.score_range
        raw_score_ratio = self.score_head(rep).squeeze(-1)  # [batch_size]
        decision_score = min_s + (max_s - min_s) * raw_score_ratio

        # Priority decision
        priority_logits = self.priority_head(rep)  # [batch_size, num_priority_classes]
        priority_probs = F.softmax(priority_logits / temp, dim=-1)

        # Risk category decision
        risk_logits = self.risk_head(rep)  # [batch_size, num_risk_classes]
        risk_probs = F.softmax(risk_logits / temp, dim=-1)

        # Head 6: Decision Confidence (maximum probability across decision distribution)
        binary_confidence, binary_predicted_class = torch.max(binary_probs, dim=-1)
        priority_confidence, priority_predicted_class = torch.max(priority_probs, dim=-1)
        risk_confidence, risk_predicted_class = torch.max(risk_probs, dim=-1)

        # 1. Decision Uncertainty: Normalized Shannon entropy of calibrated binary decision
        # H(p) = -sum(p * log(p)) / ln(2), strictly bounded in [0, 1]
        binary_entropy = -torch.sum(
            binary_probs * torch.log(torch.clamp(binary_probs, min=1e-9)), dim=-1
        ) / 0.6931471805599453

        # 2. Decision Confidence Abstention
        low_confidence_mask = binary_confidence < self.config.abstention_threshold

        # 3. Decision Entropy Abstention (independent threshold)
        dec_thresh = getattr(self.config, "decision_uncertainty_threshold", self.config.uncertainty_threshold)
        high_decision_entropy_mask = binary_entropy > dec_thresh

        # 4. Predictive Uncertainty Abstention (independently calibrated, not combined via arbitrary summation)
        pred_thresh = getattr(self.config, "predictive_uncertainty_threshold", 0.40)
        if predictive_uncertainty is not None:
            high_pred_uncertainty_mask = predictive_uncertainty.to(device) > pred_thresh
        else:
            high_pred_uncertainty_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 5. Candidate Uncertainty Abstention (independently evaluated)
        cand_thresh = getattr(self.config, "candidate_uncertainty_threshold", 0.60)
        if candidate_uncertainty is not None:
            high_cand_uncertainty_mask = candidate_uncertainty.to(device) > cand_thresh
        else:
            high_cand_uncertainty_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 6. Policy Rejection Abstention
        if policy_rejected is not None:
            policy_rejected_mask = policy_rejected.to(device).bool()
        else:
            policy_rejected_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Overall Abstention: logical disjunction of explicit, independent safety criteria
        should_abstain = (
            low_confidence_mask
            | high_decision_entropy_mask
            | high_pred_uncertainty_mask
            | high_cand_uncertainty_mask
            | policy_rejected_mask
        )

        return {
            "binary_logits": raw_binary_logits,
            "calibrated_binary_logits": calibrated_binary_logits,
            "binary_probabilities": binary_probs,
            "binary_prediction": binary_predicted_class,
            "binary_confidence": binary_confidence,
            "decision_score": decision_score,
            "priority_logits": priority_logits,
            "priority_probabilities": priority_probs,
            "priority_prediction": priority_predicted_class,
            "risk_logits": risk_logits,
            "risk_probabilities": risk_probs,
            "risk_prediction": risk_predicted_class,
            "risk_confidence": risk_confidence,
            "decision_entropy": binary_entropy,
            "decision_uncertainty": binary_entropy,
            "predictive_uncertainty": predictive_uncertainty,
            "candidate_uncertainty": candidate_uncertainty,
            "policy_rejected": policy_rejected_mask,
            "abstain_low_confidence": low_confidence_mask,
            "abstain_high_decision_entropy": high_decision_entropy_mask,
            "abstain_predictive_uncertainty": high_pred_uncertainty_mask,
            "abstain_candidate_uncertainty": high_cand_uncertainty_mask,
            "abstain_policy_rejected": policy_rejected_mask,
            "abstain": should_abstain,
            "temperature": temp,
        }
