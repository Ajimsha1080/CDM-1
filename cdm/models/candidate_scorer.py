"""Component: Candidate Scorer & Selection Mechanism for CDM-1.
Evaluates alternative candidate actions against context, predicted outcomes,
risk, and hard external policy constraints.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateScorer(nn.Module):
    """Scores candidate actions conditioned on cognitive context, predicted outcome,

    and external hard policy constraints.
    """

    def __init__(
        self,
        context_dim: int = 256,
        candidate_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.candidate_dim = candidate_dim

        # Candidate projection layer
        self.candidate_proj = nn.Sequential(
            nn.Linear(candidate_dim, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
        )

        # Bilinear interaction matrix between context and candidate
        self.bilinear = nn.Bilinear(context_dim, context_dim, hidden_dim)

        # Deep scoring MLP
        # Input: [context_dim, candidate_dim, bilinear_feat, outcome_feat(1), risk_feat(1)]
        in_features = context_dim + candidate_dim + hidden_dim + 2
        self.scorer_mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cognitive_context: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        candidate_outcomes: Optional[torch.Tensor] = None,
        candidate_risks: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

            cognitive_context: [batch_size, context_dim]
            candidate_embeddings: [batch_size, num_candidates, candidate_dim]
            candidate_outcomes: [batch_size, num_candidates, 1] or None
            candidate_risks: [batch_size, num_candidates, 1] or None
            constraint_mask: [batch_size, num_candidates] (1=allowed, 0=prohibited)

        Returns:
            raw_scores: [batch_size, num_candidates]
            constrained_scores: [batch_size, num_candidates]
            choice_probabilities: [batch_size, num_candidates]
            selected_candidate_idx: [batch_size]
        """
        batch_size, num_candidates, c_dim = candidate_embeddings.shape
        device = cognitive_context.device

        # Expand context across candidates: [batch_size, num_candidates, context_dim]
        ctx_expanded = cognitive_context.unsqueeze(1).expand(-1, num_candidates, -1)

        # Project candidates: [batch_size, num_candidates, context_dim]
        c_proj = self.candidate_proj(candidate_embeddings)

        # Flatten batch and candidate dimensions for bilinear & mlp
        ctx_flat = ctx_expanded.reshape(-1, self.context_dim)
        c_flat = c_proj.reshape(-1, self.context_dim)
        bilinear_feat = self.bilinear(ctx_flat, c_flat)  # [batch * num_cand, hidden_dim]

        # Handle auxiliary outcome and risk features
        if candidate_outcomes is None:
            cand_outcomes_flat = torch.zeros(batch_size * num_candidates, 1, device=device)
        else:
            cand_outcomes_flat = candidate_outcomes.reshape(-1, 1)

        if candidate_risks is None:
            cand_risks_flat = torch.zeros(batch_size * num_candidates, 1, device=device)
        else:
            cand_risks_flat = candidate_risks.reshape(-1, 1)

        cand_orig_flat = candidate_embeddings.reshape(-1, self.candidate_dim)

        combined = torch.cat(
            [ctx_flat, cand_orig_flat, bilinear_feat, cand_outcomes_flat, cand_risks_flat],
            dim=-1,
        )

        raw_scores_flat = self.scorer_mlp(combined)  # [batch * num_cand, 1]
        raw_scores = raw_scores_flat.reshape(batch_size, num_candidates)  # [batch, num_candidates]

        # Enforce external policy constraints:
        # If a candidate violates a hard policy constraint, mask its logit with -1e9
        if constraint_mask is not None:
            mask_val = -1e9
            constrained_scores = raw_scores.masked_fill(constraint_mask == 0, mask_val)
        else:
            constrained_scores = raw_scores

        choice_probs = F.softmax(constrained_scores, dim=-1)
        selected_idx = torch.argmax(constrained_scores, dim=-1)

        return {
            "raw_scores": raw_scores,
            "constrained_scores": constrained_scores,
            "choice_probabilities": choice_probs,
            "selected_candidate_idx": selected_idx,
        }

    @staticmethod
    def compute_ranking_loss(
        scores: torch.Tensor,
        target_idx: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss ensuring target candidate score exceeds competitors."""
        batch_size, num_cand = scores.shape
        device = scores.device
        loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            t_idx = target_idx[b].item()
            target_score = scores[b, t_idx]
            for c in range(num_cand):
                if c != t_idx:
                    other_score = scores[b, c]
                    diff = other_score - target_score + margin
                    loss = loss + torch.clamp(diff, min=0.0)

        return loss / max(batch_size * (num_cand - 1), 1)
