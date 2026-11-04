"""Structured Feature Schema Specification for CDM-1.
Defines versioned feature layouts, numerical normalization metadata,
and categorical encodings with missing-value indicator handling.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NumericFeatureDef(BaseModel):
    name: str
    description: str = ""
    mean: float = 0.0
    std: float = 1.0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    default_value: float = 0.0


class CategoricalFeatureDef(BaseModel):
    name: str
    description: str = ""
    vocabulary: List[str]
    unknown_token: str = "<unk>"
    default_token: str = "<unk>"


class FeatureSchema(BaseModel):
    schema_version: str = "1.0.0"
    numeric_features: List[NumericFeatureDef] = Field(default_factory=list)
    categorical_features: List[CategoricalFeatureDef] = Field(default_factory=list)

    @classmethod
    def get_default_schema(cls) -> "FeatureSchema":
        """Standard baseline schema for customer support & communication decision-making."""
        return cls(
            schema_version="1.0.0",
            numeric_features=[
                NumericFeatureDef(
                    name="sentiment_score",
                    description="Incoming customer sentiment score from -1.0 to 1.0",
                    mean=0.0,
                    std=0.5,
                    min_val=-1.0,
                    max_val=1.0,
                    default_value=0.0,
                ),
                NumericFeatureDef(
                    name="interaction_count",
                    description="Count of previous touches or tickets in thread",
                    mean=2.5,
                    std=2.0,
                    min_val=0.0,
                    max_val=50.0,
                    default_value=1.0,
                ),
                NumericFeatureDef(
                    name="account_tenure_months",
                    description="Duration of account relationship in months",
                    mean=24.0,
                    std=18.0,
                    min_val=0.0,
                    max_val=240.0,
                    default_value=12.0,
                ),
                NumericFeatureDef(
                    name="monetary_value_usd",
                    description="Transaction or monthly recurring value in USD",
                    mean=150.0,
                    std=200.0,
                    min_val=0.0,
                    max_val=100000.0,
                    default_value=50.0,
                ),
                NumericFeatureDef(
                    name="sla_remaining_hours",
                    description="Hours left before response SLA breach",
                    mean=8.0,
                    std=6.0,
                    min_val=-24.0,
                    max_val=72.0,
                    default_value=12.0,
                ),
                NumericFeatureDef(
                    name="historical_escalation_rate",
                    description="Prior escalation frequency for this entity (0.0 to 1.0)",
                    mean=0.15,
                    std=0.2,
                    min_val=0.0,
                    max_val=1.0,
                    default_value=0.0,
                ),
            ],
            categorical_features=[
                CategoricalFeatureDef(
                    name="domain",
                    description="Operational domain of the decision problem",
                    vocabulary=[
                        "<unk>",
                        "communication",
                        "customer_support",
                        "business_operations",
                        "retail",
                        "security",
                        "billing",
                        "technical",
                    ],
                ),
                CategoricalFeatureDef(
                    name="channel",
                    description="Ingestion channel of interaction",
                    vocabulary=["<unk>", "email", "chat", "phone", "social", "api"],
                ),
                CategoricalFeatureDef(
                    name="customer_tier",
                    description="Customer loyalty or contract tier",
                    vocabulary=["<unk>", "free", "starter", "professional", "enterprise"],
                ),
                CategoricalFeatureDef(
                    name="urgency_level",
                    description="Initial detected urgency label",
                    vocabulary=["<unk>", "low", "medium", "high", "critical"],
                ),
            ],
        )

    def get_categorical_cardinalities(self) -> Dict[str, int]:
        return {cat.name: len(cat.vocabulary) for cat in self.categorical_features}

    def transform_numeric(self, values: Dict[str, Optional[float]]) -> List[float]:
        """Normalize numeric features with standard scaling and missing-value flags."""
        result = []
        for feat in self.numeric_features:
            val = values.get(feat.name)
            if val is None:
                # Use mean normalized to 0.0
                norm_val = 0.0
            else:
                std = feat.std if feat.std > 1e-6 else 1.0
                norm_val = (val - feat.mean) / std
            result.append(norm_val)
        return result

    def get_missing_indicators(self, values: Dict[str, Any]) -> List[float]:
        """Returns 1.0 if missing, 0.0 if present for each numeric feature."""
        return [1.0 if values.get(feat.name) is None else 0.0 for feat in self.numeric_features]

    def transform_categorical(self, values: Dict[str, Optional[str]]) -> Dict[str, int]:
        """Map categorical values to vocabulary indices."""
        indices = {}
        for feat in self.categorical_features:
            val = values.get(feat.name)
            if val is None or val not in feat.vocabulary:
                indices[feat.name] = feat.vocabulary.index(feat.unknown_token)
            else:
                indices[feat.name] = feat.vocabulary.index(val)
        return indices
