"""Component D: Predictive Network for CDM-1
Estimates potential operational outcomes, risk scores, and uncertainty conditioned
on cognitive representation and optional candidate action embeddings.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from cdm.config.model_config import PredictiveNetworkConfig


class PredictiveNetwork(nn.Module):
    """Predicts multiple forward-looking outcomes conditioned on context state

    and optional candidate action vectors.
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 256,
        config: Optional[PredictiveNetworkConfig] = None,
    ):
        super().__init__()
        self.config = config or PredictiveNetworkConfig()
        hidden = self.config.hidden_dim

        # Action conditioning projection
        self.action_projector = nn.Sequential(
            nn.Linear(action_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )

        # Shared predictive backbone
        self.shared_backbone = nn.Sequential(
            nn.Linear(input_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Head 1: Binary Outcome Prediction (e.g. success vs failure)
        self.binary_head = nn.Linear(hidden, 1)

        # Head 2: Multi-class Outcome Prediction (e.g. resolution categories)
        self.multiclass_head = nn.Linear(hidden, self.config.num_outcome_classes)

        # Head 3: Numerical Outcome Prediction (e.g. resolution time hours)
        self.numeric_head = nn.Linear(hidden, 1)

        # Head 4: Risk Estimation Head (continuous score bounded in [0, 1])
        self.risk_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        cognitive_state: torch.Tensor,
        action_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

            cognitive_state: [batch_size, input_dim]
            action_embedding: [batch_size, action_dim] or None

        Returns:
            Dictionary containing:
                binary_outcome_logits: [batch_size, 1]
                multiclass_outcome_logits: [batch_size, num_classes]
                numeric_outcome: [batch_size, 1]
                estimated_risk: [batch_size, 1]
                outcome_entropy: [batch_size]
        """
        batch_size = cognitive_state.shape[0]
        device = cognitive_state.device

        if action_embedding is not None:
            proj_action = self.action_projector(action_embedding)
        else:
            proj_action = torch.zeros_like(cognitive_state)

        # Conditioned input: [batch_size, input_dim * 2]
        conditioned_input = torch.cat([cognitive_state, proj_action], dim=-1)
        features = self.shared_backbone(conditioned_input)

        bin_logits = self.binary_head(features)
        multi_logits = self.multiclass_head(features)
        num_pred = self.numeric_head(features)
        risk_pred = self.risk_head(features)

        # Calculate predictive entropy (uncertainty) on multiclass outcome
        probs = F.softmax(multi_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1e-9)), dim=-1)
        max_entropy = torch.log(torch.tensor(float(self.config.num_outcome_classes), device=device))
        normalized_entropy = entropy / torch.clamp(max_entropy, min=1e-5)

        return {
            "binary_outcome_logits": bin_logits,
            "multiclass_outcome_logits": multi_logits,
            "numeric_outcome": num_pred,
            "estimated_risk": risk_pred,
            "outcome_entropy": normalized_entropy,
        }

    def estimate_uncertainty_mc_dropout(
        self,
        cognitive_state: torch.Tensor,
        action_embedding: Optional[torch.Tensor] = None,
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Performs Monte Carlo Dropout sampling to compute epistemic uncertainty (variance)."""
        self.train()  # Enable dropout during inference
        preds = []
        for _ in range(num_samples):
            with torch.no_grad():
                out = self.forward(cognitive_state, action_embedding)
                preds.append(torch.sigmoid(out["binary_outcome_logits"]))
        stacked = torch.stack(preds, dim=0)  # [num_samples, batch_size, 1]
        variance = torch.var(stacked, dim=0).squeeze(-1)  # [batch_size]
        self.eval()
        return {"epistemic_variance": variance}
