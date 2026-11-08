"""Component A: Language Encoder for CDM-1
Extracts contextual semantic representations from text using pretrained Transformer encoders
(ModernBERT-base baseline) with attention-mask-aware pooling, feature projection,
and freezing/fine-tuning controls.
"""

from typing import Optional, Dict, Any
import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from cdm.config.model_config import LanguageEncoderConfig


class AttentionPooling(nn.Module):
    """Attention-based pooling over transformer sequence tokens."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        scores = self.attention(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        mask_val = -1e9
        scores = scores.masked_fill(attention_mask == 0, mask_val)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = torch.sum(hidden_states * weights, dim=1)  # [batch_size, hidden_size]
        return pooled


class LanguageEncoder(nn.Module):
    """Transformer-based language encoder with projection and pooling strategies."""

    def __init__(self, config: Optional[LanguageEncoderConfig] = None):
        super().__init__()
        self.config = config or LanguageEncoderConfig()
        
        # Load transformer encoder safely across model families
        model_kwargs: Dict[str, Any] = {}
        if "modernbert" not in self.config.model_name_or_path.lower() and "bert" in self.config.model_name_or_path.lower():
            model_kwargs["add_pooling_layer"] = False

        if not os.path.isdir(self.config.model_name_or_path):
            if getattr(self.config, "revision", None):
                model_kwargs["revision"] = self.config.revision
        if getattr(self.config, "local_files_only", False):
            model_kwargs["local_files_only"] = True

        try:
            self.transformer = AutoModel.from_pretrained(
                self.config.model_name_or_path,
                **model_kwargs,
            )
        except Exception as e:
            if getattr(self.config, "local_files_only", False):
                raise RuntimeError(
                    f"ModernBERT backbone '{self.config.model_name_or_path}' is missing from local files/cache "
                    f"with local_files_only=True. Run 'python scripts/cache_backbone.py' to pre-cache the pinned "
                    f"backbone (revision: {getattr(self.config, 'revision', 'pinned')}) before starting offline: {e}"
                ) from e
            raise
        
        actual_hidden = getattr(self.transformer.config, "hidden_size", self.config.encoder_hidden_size)
        self.config.encoder_hidden_size = actual_hidden
        
        # Pooling strategy
        if self.config.pooling_strategy == "attention_pool":
            self.pooler = AttentionPooling(actual_hidden)
        else:
            self.pooler = None

        # Projection head to fusion dimension
        self.projection = nn.Sequential(
            nn.Linear(actual_hidden, self.config.projection_dim),
            nn.LayerNorm(self.config.projection_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        if self.config.freeze_encoder:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze transformer backbone weights to preserve memory and baseline representations."""
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze transformer backbone weights for end-to-end fine-tuning."""
        for param in self.transformer.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, num_layers: int = 2) -> None:
        """Gradually unfreeze only the top N layers of the transformer."""
        self.freeze_backbone()
        # ModernBERT layers are in self.transformer.layers
        layers = getattr(self.transformer, "layers", None)
        if layers is None:
            # Fallback for standard BertModel / others
            encoder = getattr(self.transformer, "encoder", None)
            layers = getattr(encoder, "layer", None)
            
        if layers is not None:
            for layer in list(layers)[-num_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def pool_tokens(self, sequence_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply attention-mask-aware pooling over sequence tokens."""
        if self.config.pooling_strategy == "cls":
            return sequence_output[:, 0, :]
        elif self.config.pooling_strategy == "attention_pool" and self.pooler is not None:
            return self.pooler(sequence_output, attention_mask)
        else:
            # Default: Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returns projected contextual text embedding [batch_size, projection_dim]."""
        kwargs: Dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "token_type_ids" in self.transformer.forward.__code__.co_varnames:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**kwargs)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        pooled = self.pool_tokens(sequence_output, attention_mask)
        projected = self.projection(pooled)
        return projected
