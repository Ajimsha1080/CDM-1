"""CDM-1 — Cognitive Decision Model & Integrated Cognitive AI Engine
Independent Trainable AI Decision Architecture
"""

from cdm.models.cdm_model import CDMModel
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult
from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import (
    SharedCognitiveState,
    PerceptionResult,
    ContextFrame,
    WorkingMemoryState,
    WorldState,
    ReasoningTrace,
    Plan,
    CognitiveEngineResult,
)

__version__ = "0.1.0"

__all__ = [
    "CDMModel",
    "CognitiveDecisionModel",
    "DecisionResult",
    "CognitiveEngine",
    "SharedCognitiveState",
    "PerceptionResult",
    "ContextFrame",
    "WorkingMemoryState",
    "WorldState",
    "ReasoningTrace",
    "Plan",
    "CognitiveEngineResult",
]
