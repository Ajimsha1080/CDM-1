"""CDM-1 Cognitive AI Engine Package."""

from cdm.cognitive.state import (
    SharedCognitiveState,
    PerceptionResult,
    EntityMention,
    ContextFrame,
    WorkingMemoryState,
    GoalNode,
    GoalStatus,
    WorldEntity,
    WorldRelation,
    WorldState,
    ReasoningStep,
    ReasoningTrace,
    PlanStep,
    Plan,
    CognitiveEngineResult,
)
def __getattr__(name):
    if name == "CognitiveEngine":
        from cdm.cognitive.engine import CognitiveEngine
        return CognitiveEngine
    raise AttributeError(f"module {__name__} has no attribute {name}")

__all__ = [
    "CognitiveEngine",
    "SharedCognitiveState",
    "PerceptionResult",
    "EntityMention",
    "ContextFrame",
    "WorkingMemoryState",
    "GoalNode",
    "GoalStatus",
    "WorldEntity",
    "WorldRelation",
    "WorldState",
    "ReasoningStep",
    "ReasoningTrace",
    "PlanStep",
    "Plan",
    "CognitiveEngineResult",
]
