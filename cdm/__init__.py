"""CDM-1 — Cognitive Decision Model
Independent Trainable AI Decision Architecture
"""

from cdm.models.cdm_model import CDMModel
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult

__version__ = "0.1.0"

__all__ = [
    "CDMModel",
    "CognitiveDecisionModel",
    "DecisionResult",
]
