"""CDM-1 Configuration Schema
Defines model hyperparameters, component topologies, and training settings.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class LanguageEncoderConfig(BaseModel):
    model_name_or_path: str = Field(
        default="answerdotai/ModernBERT-base",
        description="Pretrained transformer encoder model identifier or local path."
    )
    revision: Optional[str] = Field(
        default="8949b909ec900327062f0ebf497f51aef5e6f0c8",
        description="Pinned git commit revision of the model."
    )
    local_files_only: bool = Field(
        default=False,
        description="Whether to only use local files and avoid downloading from remote hub."
    )
    freeze_encoder: bool = Field(
        default=True,
        description="Whether to freeze transformer weights during initial training stages."
    )
    pooling_strategy: str = Field(
        default="mean",
        description="Pooling method across sequence length: 'mean', 'cls', or 'attention_pool'."
    )
    max_seq_length: int = Field(
        default=256,
        description="Maximum sequence length for tokenization."
    )
    encoder_hidden_size: int = Field(
        default=768,
        description="Hidden representation dimension output by language encoder."
    )
    projection_dim: int = Field(
        default=256,
        description="Dimension to project pooled text representation to."
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout applied after text projection."
    )


class StructuredEncoderConfig(BaseModel):
    num_numeric_features: int = Field(
        default=6,
        description="Count of continuous numeric input features."
    )
    categorical_cardinalities: Dict[str, int] = Field(
        default_factory=lambda: {
            "domain": 8,
            "channel": 6,
            "customer_tier": 5,
            "urgency_level": 4,
        },
        description="Mapping from categorical feature name to vocabulary size."
    )
    categorical_embedding_dim: int = Field(
        default=16,
        description="Embedding dimension for categorical features."
    )
    hidden_dims: List[int] = Field(
        default_factory=lambda: [128, 256],
        description="Hidden layer dimensions for structured MLP."
    )
    projection_dim: int = Field(
        default=256,
        description="Final output dimension for structured feature representation."
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout rate for structured feature MLP."
    )


class FusionConfig(BaseModel):
    fusion_type: str = Field(
        default="gated",
        description="Fusion mechanism: 'gated', 'cross_attention', or 'concat'."
    )
    fusion_dim: int = Field(
        default=256,
        description="Dimensionality of the shared cognitive representation."
    )
    num_attention_heads: int = Field(
        default=4,
        description="Number of attention heads if cross_attention fusion is active."
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout rate within fusion block."
    )


class PredictiveNetworkConfig(BaseModel):
    enabled: bool = True
    hidden_dim: int = Field(default=128)
    num_outcome_classes: int = Field(
        default=4,
        description="Number of discrete outcome categories."
    )
    dropout: float = Field(default=0.1)


class DecisionNetworkConfig(BaseModel):
    num_priority_classes: int = Field(
        default=4,
        description="Priority levels: e.g. 0: Low, 1: Medium, 2: High, 3: Critical."
    )
    num_risk_classes: int = Field(
        default=4,
        description="Risk tiers: 0: Low, 1: Moderate, 2: High, 3: Severe."
    )
    score_range: tuple = Field(
        default=(0.0, 100.0),
        description="Continuous decision score range min/max."
    )
    abstention_threshold: float = Field(
        default=0.55,
        description="Calibrated probability threshold below which the model abstains."
    )
    uncertainty_threshold: float = Field(
        default=0.45,
        description="Maximum normalized decision entropy allowed before abstaining."
    )
    decision_uncertainty_threshold: float = Field(
        default=0.45,
        description="Maximum normalized decision entropy allowed before abstaining."
    )
    predictive_uncertainty_threshold: float = Field(
        default=0.40,
        description="Maximum predictive outcome uncertainty (entropy/variance) allowed before abstaining."
    )
    candidate_uncertainty_threshold: float = Field(
        default=0.60,
        description="Maximum candidate choice entropy allowed before abstaining."
    )
    dropout: float = Field(default=0.1)


class LossWeightsConfig(BaseModel):
    lambda_binary: float = 1.0
    lambda_choice: float = 1.0
    lambda_ranking: float = 0.5
    lambda_score: float = 0.5
    lambda_priority: float = 0.8
    lambda_risk: float = 0.8
    lambda_outcome_binary: float = 0.6
    lambda_outcome_multiclass: float = 0.6
    lambda_outcome_numeric: float = 0.4


class CDMModelConfig(BaseModel):
    model_name: str = "CDM-1"
    version: str = "1.0.0"
    language_encoder: LanguageEncoderConfig = Field(default_factory=LanguageEncoderConfig)
    structured_encoder: StructuredEncoderConfig = Field(default_factory=StructuredEncoderConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    predictive: PredictiveNetworkConfig = Field(default_factory=PredictiveNetworkConfig)
    decision: DecisionNetworkConfig = Field(default_factory=DecisionNetworkConfig)
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "CDMModelConfig":
        return cls.model_validate(data)
