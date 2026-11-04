"""Component C: Cognitive Representation Fusion for CDM-1
Combines linguistic representations and structured tabular features into a unified
cognitive state using gated fusion, cross-attention, or projected concatenation.
"""

from typing import Optional
import torch
import torch.nn as nn
from cdm.config.model_config import FusionConfig


class GatedFusion(nn.Module):
    """Gated adaptive fusion: computes a dynamic weighting gate between text and structured modalities."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_rep: torch.Tensor, struct_rep: torch.Tensor) -> torch.Tensor:
        # text_rep: [batch_size, dim], struct_rep: [batch_size, dim]
        concat = torch.cat([text_rep, struct_rep], dim=-1)
        gate = self.gate_layer(concat)  # [batch_size, dim] in (0, 1)
        fused = gate * text_rep + (1.0 - gate) * struct_rep
        return self.fusion_mlp(fused)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between linguistic and structured feature tokens."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn_text_to_struct = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_struct_to_text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_rep: torch.Tensor, struct_rep: torch.Tensor) -> torch.Tensor:
        # Reshape to [batch_size, 1, dim] for MHA
        t = text_rep.unsqueeze(1)
        s = struct_rep.unsqueeze(1)

        t_attended, _ = self.attn_text_to_struct(query=t, key=s, value=s)
        s_attended, _ = self.attn_struct_to_text(query=s, key=t, value=t)

        t_res = (t + t_attended).squeeze(1)
        s_res = (s + s_attended).squeeze(1)

        g = self.gate(torch.cat([t_res, s_res], dim=-1))
        fused = g * t_res + (1.0 - g) * s_res
        return self.norm(self.dropout(fused))


class CognitiveFusion(nn.Module):
    """Unified Cognitive Representation Fusion module supporting gated, cross-attention,

    and baseline concatenated fusion mechanisms.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        super().__init__()
        self.config = config or FusionConfig()
        dim = self.config.fusion_dim

        if self.config.fusion_type == "cross_attention":
            self.fusion_module = CrossAttentionFusion(
                dim=dim,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout,
            )
        elif self.config.fusion_type == "concat":
            self.fusion_module = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
        else:
            # Default: Gated fusion
            self.fusion_module = GatedFusion(dim=dim, dropout=self.config.dropout)

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, text_embedding: torch.Tensor, structured_embedding: torch.Tensor) -> torch.Tensor:
        """Args:

            text_embedding: [batch_size, fusion_dim]
            structured_embedding: [batch_size, fusion_dim]

        Returns:
            cognitive_representation: [batch_size, fusion_dim]
        """
        if self.config.fusion_type == "concat":
            fused = self.fusion_module(torch.cat([text_embedding, structured_embedding], dim=-1))
        else:
            fused = self.fusion_module(text_embedding, structured_embedding)
        return self.final_norm(fused)
