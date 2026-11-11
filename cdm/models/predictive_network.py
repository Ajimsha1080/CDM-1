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
        self.input_dim = input_dim
        self.action_dim = action_dim
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
        candidate_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

            cognitive_state: [batch_size, input_dim]
            action_embedding: [batch_size, action_dim] or None
            candidate_embeddings: [batch_size, num_cands, candidate_dim] or None

        Returns:
            Dictionary containing:
                binary_outcome_logits: [batch_size, 1]
                multiclass_outcome_logits: [batch_size, num_classes]
                numeric_outcome: [batch_size, 1]
                estimated_risk: [batch_size, 1]
                outcome_entropy: [batch_size]
                candidate_representations: [batch_size, num_cands, input_dim * 2] (if candidate_embeddings provided)
                candidate_binary_outcome_logits: [batch_size, num_cands, 1] (if candidate_embeddings provided)
                candidate_multiclass_outcome_logits: [batch_size, num_cands, num_classes] (if candidate_embeddings provided)
                candidate_numeric_outcome: [batch_size, num_cands, 1] (if candidate_embeddings provided)
                candidate_estimated_risk: [batch_size, num_cands, 1] (if candidate_embeddings provided)
                candidate_outcome_entropy: [batch_size, num_cands] (if candidate_embeddings provided)
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

        result: Dict[str, torch.Tensor] = {
            "binary_outcome_logits": bin_logits,
            "multiclass_outcome_logits": multi_logits,
            "numeric_outcome": num_pred,
            "estimated_risk": risk_pred,
            "outcome_entropy": normalized_entropy,
        }

        # Priority 1: Action-Conditioned Forward Predictive Modeling for Candidate Actions
        if candidate_embeddings is not None and candidate_embeddings.shape[1] > 0:
            num_cands = candidate_embeddings.shape[1]
            # Project candidate embeddings into input_dim space
            cand_proj = self.action_projector(candidate_embeddings)  # [batch_size, num_cands, input_dim]
            ctx_expanded = cognitive_state.unsqueeze(1).expand(-1, num_cands, -1)  # [batch_size, num_cands, input_dim]

            # Candidate predictive representations: [batch_size, num_cands, input_dim * 2]
            cand_rep = torch.cat([ctx_expanded, cand_proj], dim=-1)

            # Pass each candidate's representation through shared predictive backbone
            flat_cand_rep = cand_rep.reshape(-1, self.input_dim * 2)
            flat_cand_features = self.shared_backbone(flat_cand_rep)  # [batch_size * num_cands, hidden]

            # Independent candidate outcome predictions
            cand_bin_logits = self.binary_head(flat_cand_features).reshape(batch_size, num_cands, 1)
            cand_multi_logits = self.multiclass_head(flat_cand_features).reshape(
                batch_size, num_cands, self.config.num_outcome_classes
            )
            cand_numeric = self.numeric_head(flat_cand_features).reshape(batch_size, num_cands, 1)
            cand_risk = self.risk_head(flat_cand_features).reshape(batch_size, num_cands, 1)

            cand_probs = F.softmax(cand_multi_logits, dim=-1)
            cand_entropy = -torch.sum(cand_probs * torch.log(torch.clamp(cand_probs, min=1e-9)), dim=-1)
            cand_norm_entropy = cand_entropy / torch.clamp(max_entropy, min=1e-5)

            result["candidate_representations"] = cand_rep
            result["candidate_binary_outcome_logits"] = cand_bin_logits
            result["candidate_multiclass_outcome_logits"] = cand_multi_logits
            result["candidate_numeric_outcome"] = cand_numeric
            result["candidate_estimated_risk"] = cand_risk
            result["candidate_outcome_entropy"] = cand_norm_entropy

        return result

    def estimate_uncertainty_mc_dropout(
        self,
        cognitive_state: torch.Tensor,
        action_embedding: Optional[torch.Tensor] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
        num_samples: int = 10,
    ) -> Dict[str, torch.Tensor]:
        """Performs Monte Carlo Dropout sampling to compute epistemic uncertainty (variance).

        Requires num_samples >= 2. Uses unbiased=False for sample variance computation.
        Safely restores the original model training/evaluation mode in a finally block.
        """
        if num_samples < 2:
            raise ValueError(
                f"num_samples must be at least 2 for Monte Carlo variance estimation, got {num_samples}."
            )

        was_training = self.training
        try:
            self.train(True)  # Enable dropout during sampling
            binary_preds = []
            risk_preds = []
            multiclass_preds = []
            for _ in range(num_samples):
                with torch.no_grad():
                    out = self.forward(
                        cognitive_state=cognitive_state,
                        action_embedding=action_embedding,
                        candidate_embeddings=candidate_embeddings,
                    )
                    binary_preds.append(torch.sigmoid(out["binary_outcome_logits"]))
                    risk_preds.append(out["estimated_risk"])
                    multiclass_preds.append(F.softmax(out["multiclass_outcome_logits"], dim=-1))

            stacked_binary = torch.stack(binary_preds, dim=0)  # [num_samples, batch_size, 1]
            stacked_risk = torch.stack(risk_preds, dim=0)  # [num_samples, batch_size, 1]
            stacked_multi = torch.stack(multiclass_preds, dim=0)  # [num_samples, batch_size, num_classes]

            # Monte Carlo variance with unbiased=False as specified
            var_binary = torch.var(stacked_binary, dim=0, unbiased=False).squeeze(-1)  # [batch_size]
            var_risk = torch.var(stacked_risk, dim=0, unbiased=False).squeeze(-1)  # [batch_size]
            var_multi = torch.var(stacked_multi, dim=0, unbiased=False).mean(dim=-1)  # [batch_size]

            mean_binary = torch.mean(stacked_binary, dim=0).squeeze(-1)
            mean_risk = torch.mean(stacked_risk, dim=0).squeeze(-1)

            return {
                "epistemic_variance": var_binary,
                "risk_variance": var_risk,
                "multiclass_variance": var_multi,
                "mean_binary_prob": mean_binary,
                "mean_risk": mean_risk,
                "num_samples": num_samples,
            }
        finally:
            self.train(was_training)
