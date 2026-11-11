"""Component: Candidate Scorer & Selection Mechanism for CDM-1.
Evaluates alternative candidate actions against context, predicted outcomes,
risk, and hard external policy constraints with vectorized ranking losses.
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
        existence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Args:

            cognitive_context: [batch_size, context_dim]
            candidate_embeddings: [batch_size, num_candidates, candidate_dim]
            candidate_outcomes: [batch_size, num_candidates, 1] or None
            candidate_risks: [batch_size, num_candidates, 1] or None
            constraint_mask: [batch_size, num_candidates] (1=allowed, 0=prohibited)
            existence_mask: [batch_size, num_candidates] (1=valid candidate, 0=padding)

        Returns:
            raw_scores: [batch_size, num_candidates]
            constrained_scores: [batch_size, num_candidates]
            choice_probabilities: [batch_size, num_candidates]
            selected_candidate_idx: [batch_size]
        """
        # 1. Validation of Candidate Tensor Dimensions
        if candidate_embeddings.ndim != 3:
            raise ValueError(
                f"candidate_embeddings must be 3D tensor [batch_size, num_candidates, candidate_dim], "
                f"got shape {candidate_embeddings.shape}."
            )
        batch_size, num_candidates, c_dim = candidate_embeddings.shape
        device = cognitive_context.device

        # Reject zero-candidate inputs with an explicit, documented exception
        if num_candidates == 0:
            raise ValueError(
                "CandidateScorer requires at least one candidate embedding; num_candidates cannot be zero."
            )

        if cognitive_context.ndim != 2 or cognitive_context.shape[0] != batch_size:
            raise ValueError(
                f"cognitive_context batch dimension {cognitive_context.shape[0]} does not match "
                f"candidate batch dimension {batch_size}."
            )
        if c_dim != self.candidate_dim:
            raise ValueError(
                f"candidate_dim mismatch: expected {self.candidate_dim}, got {c_dim}."
            )
        if cognitive_context.shape[1] != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: expected {self.context_dim}, got {cognitive_context.shape[1]}."
            )

        # Validate masks and outcome dimensions
        if constraint_mask is not None and constraint_mask.shape != (batch_size, num_candidates):
            raise ValueError(
                f"constraint_mask shape {constraint_mask.shape} does not match (batch_size, num_candidates) "
                f"({batch_size}, {num_candidates})."
            )
        if existence_mask is not None and existence_mask.shape != (batch_size, num_candidates):
            raise ValueError(
                f"existence_mask shape {existence_mask.shape} does not match (batch_size, num_candidates) "
                f"({batch_size}, {num_candidates})."
            )
        if candidate_outcomes is not None and candidate_outcomes.shape[:2] != (batch_size, num_candidates):
            raise ValueError(
                f"candidate_outcomes leading shape {candidate_outcomes.shape[:2]} does not match "
                f"({batch_size}, {num_candidates})."
            )
        if candidate_risks is not None and candidate_risks.shape[:2] != (batch_size, num_candidates):
            raise ValueError(
                f"candidate_risks leading shape {candidate_risks.shape[:2]} does not match "
                f"({batch_size}, {num_candidates})."
            )

        # Expand context across candidates: [batch_size, num_candidates, context_dim]
        ctx_expanded = cognitive_context.unsqueeze(1).expand(-1, num_candidates, -1)

        # Project candidates: [batch_size, num_candidates, context_dim]
        c_proj = self.candidate_proj(candidate_embeddings)

        # Flatten batch and candidate dimensions for bilinear & mlp
        ctx_flat = ctx_expanded.reshape(-1, self.context_dim)
        c_flat = c_proj.reshape(-1, self.context_dim)
        bilinear_feat = self.bilinear(ctx_flat, c_flat)

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

        raw_scores_flat = self.scorer_mlp(combined)
        raw_scores = raw_scores_flat.reshape(batch_size, num_candidates)

        # Effective mask combines allowed constraints and padding existence
        effective_mask = torch.ones_like(raw_scores)
        if constraint_mask is not None:
            effective_mask = effective_mask * constraint_mask
        if existence_mask is not None:
            effective_mask = effective_mask * existence_mask

        # Priority 2: Candidate Constraint Safety - Detect All-Prohibited Edge Cases
        valid_counts = effective_mask.sum(dim=-1)  # [batch_size]
        all_prohibited = (valid_counts == 0)      # [batch_size] boolean tensor

        mask_val = -1e9
        constrained_scores = raw_scores.masked_fill(effective_mask == 0, mask_val)

        # Compute softmax over candidates
        choice_probs = F.softmax(constrained_scores, dim=-1)

        # Zero out prohibited and padded candidates explicitly
        choice_probs = choice_probs * effective_mask

        # Re-normalize over valid candidates; for all_prohibited, set all probabilities to 0.0
        sum_p = choice_probs.sum(dim=-1, keepdim=True)
        # Avoid division by zero: where sum_p > 0 normalize, else keep 0.0
        choice_probs = torch.where(sum_p > 0, choice_probs / torch.clamp(sum_p, min=1e-9), torch.zeros_like(choice_probs))

        # Argmax selection: if all candidates are prohibited, select -1 (abstain sentinel)
        raw_selected_idx = torch.argmax(constrained_scores, dim=-1)
        # Sentinel -1 indicates abstention / no candidate permitted
        selected_idx = torch.where(all_prohibited, torch.tensor(-1, device=device, dtype=torch.long), raw_selected_idx)

        return {
            "raw_scores": raw_scores,
            "constrained_scores": constrained_scores,
            "choice_probabilities": choice_probs,
            "selected_candidate_idx": selected_idx,
            "abstained": all_prohibited,
            "all_prohibited": all_prohibited,
        }

    @staticmethod
    def compute_ranking_loss(
        scores: torch.Tensor,
        target_idx: torch.Tensor,
        choice_mask: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss ensuring target candidate score exceeds competitors,
        vectorized and masked to skip padded candidates and examples without valid choice labels.

        Rejects out-of-range candidate targets, padding targets, and prohibited targets
        for all active choice examples instead of silently clamping them.
        """
        batch_size, num_cand = scores.shape
        device = scores.device

        if choice_mask is None:
            choice_mask = torch.ones(batch_size, device=device)
        if existence_mask is None:
            existence_mask = torch.ones(batch_size, num_cand, device=device)

        active = (choice_mask > 0)
        if active.any():
            active_targets = target_idx[active]
            # 1. Reject out-of-range target indices
            out_of_bounds = (active_targets < 0) | (active_targets >= num_cand)
            if out_of_bounds.any():
                invalid_vals = active_targets[out_of_bounds].tolist()
                raise IndexError(
                    f"Out-of-range candidate target indices detected: {invalid_vals} "
                    f"for candidate pool of size {num_cand}."
                )

            # 2. Reject targets that correspond to padding entries
            active_batch_indices = torch.nonzero(active, as_tuple=True)[0]
            for b_idx in active_batch_indices:
                t_idx = target_idx[b_idx].item()
                if existence_mask[b_idx, t_idx] <= 0:
                    raise ValueError(
                        f"Selected candidate target index {t_idx} at batch index {b_idx} "
                        f"corresponds to a padding entry (existence_mask == 0)."
                    )
                if constraint_mask is not None and constraint_mask[b_idx, t_idx] <= 0:
                    raise ValueError(
                        f"Selected candidate target index {t_idx} at batch index {b_idx} "
                        f"corresponds to a prohibited candidate (constraint_mask == 0)."
                    )

        # For safe indexing across whole tensor (including inactive items where choice_mask == 0)
        safe_targets = torch.clamp(target_idx, 0, num_cand - 1)
        target_scores = scores.gather(1, safe_targets.unsqueeze(1))  # [batch_size, 1]

        # Margin diff: other_score - target_score + margin
        margin_matrix = scores - target_scores + margin  # [batch_size, num_cand]
        hinge_loss = torch.clamp(margin_matrix, min=0.0)

        # Zero out target candidate itself
        target_one_hot = F.one_hot(safe_targets, num_classes=num_cand).float()
        valid_competitor_mask = existence_mask * (1.0 - target_one_hot)

        masked_hinge = hinge_loss * valid_competitor_mask
        per_sample_loss = masked_hinge.sum(dim=-1) / torch.clamp(
            valid_competitor_mask.sum(dim=-1), min=1.0
        )

        # Apply choice_mask (only average over examples with valid choice labels)
        total_valid_examples = torch.clamp(choice_mask.sum(), min=1.0)
        return (per_sample_loss * choice_mask).sum() / total_valid_examples
