"""Component E: Decision Network for CDM-1
Houses independent output decision heads: Binary (YES/NO), Continuous Scoring,
Priority Classification, Risk Categorization, Calibrated Confidence, and Abstention Logic.
"""

from typing import Dict, Optional, Tuple
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

    def forward(self, cognitive_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:

            cognitive_state: [batch_size, input_dim]

        Returns:
            Dictionary with logits, calibrated probabilities, continuous score,
            priority, risk, entropy, and abstention flags.
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

        # Predictive uncertainty measure: Shannon entropy of binary decision
        binary_entropy = -torch.sum(
            binary_probs * torch.log(torch.clamp(binary_probs, min=1e-9)), dim=-1
        ) / 0.693147  # normalized by ln(2) so max entropy is 1.0

        # Head 7: Abstention Mechanism
        # Model abstains if confidence < abstention_threshold OR entropy > uncertainty_threshold
        low_confidence_mask = binary_confidence < self.config.abstention_threshold
        high_uncertainty_mask = binary_entropy > self.config.uncertainty_threshold
        should_abstain = low_confidence_mask | high_uncertainty_mask

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
            "abstain": should_abstain,
            "temperature": temp,
        }
