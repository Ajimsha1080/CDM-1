"""Component B: Structured Feature Encoder for CDM-1
Encodes numerical, categorical, and missing-value indicators into a dense representation
using learnable entity embeddings and an MLP-based representation network.
"""

from typing import Dict, Optional, List
import torch
import torch.nn as nn
from cdm.config.model_config import StructuredEncoderConfig


class StructuredFeatureEncoder(nn.Module):
    """Encodes tabular structured variables: continuous features, missingness masks,

    and categorical variables via learned embedding tables.
    """

    def __init__(self, config: Optional[StructuredEncoderConfig] = None):
        super().__init__()
        self.config = config or StructuredEncoderConfig()

        # Learnable categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for cat_name, cardinality in self.config.categorical_cardinalities.items():
            self.cat_embeddings[cat_name] = nn.Embedding(
                num_embeddings=cardinality + 1,  # extra index for safety
                embedding_dim=self.config.categorical_embedding_dim,
                padding_idx=0,
            )
            total_cat_dim += self.config.categorical_embedding_dim

        # Input dimension:
        # continuous values + missing value indicator flags (1 per continuous feature) + categorical embeddings
        input_dim = (self.config.num_numeric_features * 2) + total_cat_dim

        # Multi-Layer Perceptron
        layers: List[nn.Module] = []
        curr_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.dropout))
            curr_dim = hidden_dim

        # Final projection layer to projection_dim
        layers.append(nn.Linear(curr_dim, self.config.projection_dim))
        layers.append(nn.LayerNorm(self.config.projection_dim))
        layers.append(nn.GELU())

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        numeric_values: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Args:

            numeric_values: [batch_size, num_numeric_features]
            missing_indicators: [batch_size, num_numeric_features] (0.0 if present, 1.0 if missing)
            categorical_indices: dict mapping feature_name -> [batch_size] int tensor

        Returns:
            projected: [batch_size, projection_dim]
        """
        batch_size = numeric_values.shape[0]
        device = numeric_values.device

        # If no missing indicators provided, assume all present (zeros)
        if missing_indicators is None:
            missing_indicators = torch.zeros_like(numeric_values)

        parts = [numeric_values, missing_indicators]

        # Embed categorical variables
        if categorical_indices is not None:
            for cat_name in self.config.categorical_cardinalities.keys():
                if cat_name in categorical_indices:
                    idx = categorical_indices[cat_name].to(device)
                    emb = self.cat_embeddings[cat_name](idx)
                    parts.append(emb)
                else:
                    # Missing categorical: index 0 (unknown)
                    zero_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
                    emb = self.cat_embeddings[cat_name](zero_idx)
                    parts.append(emb)
        else:
            for cat_name in self.config.categorical_cardinalities.keys():
                zero_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
                emb = self.cat_embeddings[cat_name](zero_idx)
                parts.append(emb)

        combined_input = torch.cat(parts, dim=-1)
        output = self.network(combined_input)
        return output
