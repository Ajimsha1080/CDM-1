"""CDM-1 — Cognitive Decision Model & Integrated Cognitive AI Engine
Independent Trainable AI Decision Architecture
"""

import importlib.metadata
from typing import TYPE_CHECKING

try:
    __version__ = importlib.metadata.version("cdm-1")
except Exception:
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

if TYPE_CHECKING:
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


def __getattr__(name: str):
    if name == "CDMModel":
        from cdm.models.cdm_model import CDMModel
        return CDMModel
    elif name in ("CognitiveDecisionModel", "DecisionResult"):
        from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult
        if name == "CognitiveDecisionModel":
            return CognitiveDecisionModel
        return DecisionResult
    elif name == "CognitiveEngine":
        from cdm.cognitive.engine import CognitiveEngine
        return CognitiveEngine
    elif name in (
        "SharedCognitiveState",
        "PerceptionResult",
        "ContextFrame",
        "WorkingMemoryState",
        "WorldState",
        "ReasoningTrace",
        "Plan",
        "CognitiveEngineResult",
    ):
        import cdm.cognitive.state as state_mod
        return getattr(state_mod, name)
    raise AttributeError(f"module 'cdm' has no attribute '{name}'")
