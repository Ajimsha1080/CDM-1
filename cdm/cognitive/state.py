"""Shared Cognitive State & Typed Contracts for CDM-1 Cognitive AI Engine.
Provides strict Pydantic v2 data models for inter-module communication across
Perception, Context, Memory, World Model, Reasoning, Planning, CDM-1 Decisions, and Metacognition.
"""

from typing import Dict, List, Optional, Any, Union
import time
import uuid
from pydantic import BaseModel, Field


class EntityMention(BaseModel):
    """Extracted entity mention from perception."""
    entity_id: str
    entity_type: str  # e.g., 'customer', 'system', 'transaction', 'incident', 'policy'
    name: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = Field(default_factory=dict)


class PerceptionResult(BaseModel):
    """Structured perception representation of incoming raw stimuli."""
    perception_id: str = Field(default_factory=lambda: f"perc_{uuid.uuid4().hex[:8]}")
    raw_text: str
    validated_tabular: Dict[str, Any] = Field(default_factory=dict)
    extracted_entities: List[EntityMention] = Field(default_factory=list)
    signals: Dict[str, float] = Field(default_factory=dict)
    modality_confidence: float = 1.0
    timestamp: float = Field(default_factory=time.time)


class ContextFrame(BaseModel):
    """Contextual situation tracking over conversations, time, and environment."""
    frame_id: str = Field(default_factory=lambda: f"ctx_{uuid.uuid4().hex[:8]}")
    session_id: str = "default_session"
    tenant_id: str = "default_tenant"
    turn_index: int = 0
    dialogue_history: List[Dict[str, str]] = Field(default_factory=list)
    temporal_delta_seconds: float = 0.0
    timestamp: float = Field(default_factory=time.time)
    detected_contradictions: List[str] = Field(default_factory=list)
    relevance_scores: Dict[str, float] = Field(default_factory=dict)


class GoalNode(BaseModel):
    """Structured goal in the goal hierarchy."""
    goal_id: str
    description: str
    priority: int = 1  # 1 (Highest) to 5 (Lowest)
    status: str = "active"  # 'active', 'completed', 'blocked', 'abandoned'
    parent_goal_id: Optional[str] = None
    subgoal_ids: List[str] = Field(default_factory=list)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)


class WorkingMemoryState(BaseModel):
    """Active task state, transient scratchpad, and active goals in working memory."""
    active_goals: List[GoalNode] = Field(default_factory=list)
    scratchpad: Dict[str, Any] = Field(default_factory=dict)
    active_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    recent_memory_keys: List[str] = Field(default_factory=list)
    attention_focus: Optional[str] = None


class WorldEntity(BaseModel):
    """Entity representation in the World Model."""
    entity_id: str
    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    inferred_properties: Dict[str, Any] = Field(default_factory=dict)
    uncertainty: float = 0.0  # 0.0 (Certain) to 1.0 (Completely Unknown)
    last_updated: float = Field(default_factory=time.time)


class WorldRelation(BaseModel):
    """Directed relationship between entities in the World Model."""
    source_id: str
    relation_type: str  # e.g., 'owns', 'impacts', 'depends_on', 'violates'
    target_id: str
    confidence: float = 1.0


class WorldState(BaseModel):
    """Explicit internal representation of the operational environment.
    Strictly distinguishes observed facts, inferred state, simulated state, and predicted future state.
    """
    state_id: str = Field(default_factory=lambda: f"ws_{uuid.uuid4().hex[:8]}")
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"
    entities: Dict[str, WorldEntity] = Field(default_factory=dict)
    relations: List[WorldRelation] = Field(default_factory=list)
    environment_metrics: Dict[str, float] = Field(default_factory=dict)
    active_constraints: List[str] = Field(default_factory=list)
    observed_facts: Dict[str, Any] = Field(default_factory=dict)
    inferred_state: Dict[str, Any] = Field(default_factory=dict)
    simulated_state: Dict[str, Any] = Field(default_factory=dict)
    predicted_future_state: Dict[str, Any] = Field(default_factory=dict)
    overall_state_uncertainty: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class ReasoningStep(BaseModel):
    """Single deductive step in a reasoning trace."""
    step_index: int
    rule_or_premise: str
    inputs: List[str]
    conclusion: str
    confidence: float = 1.0
    is_hard_constraint: bool = False


class ReasoningTrace(BaseModel):
    """Traceable deduction graph and constraint satisfaction evidence."""
    trace_id: str = Field(default_factory=lambda: f"reas_{uuid.uuid4().hex[:8]}")
    steps: List[ReasoningStep] = Field(default_factory=list)
    satisfied_constraints: List[str] = Field(default_factory=list)
    violated_constraints: List[str] = Field(default_factory=list)
    contradictions_found: List[str] = Field(default_factory=list)
    reasoning_confidence: float = 1.0


class PlanStep(BaseModel):
    """Single action step in a multi-step plan."""
    step_number: int
    action_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expected_outcome: Dict[str, Any] = Field(default_factory=dict)
    preconditions: List[str] = Field(default_factory=list)
    postconditions: List[str] = Field(default_factory=list)
    risk_estimate: float = 0.0
    is_reversible: bool = True


class Plan(BaseModel):
    """Multi-step plan evaluated against World Model transitions."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    target_goal_id: str
    steps: List[PlanStep] = Field(default_factory=list)
    cumulative_risk: float = 0.0
    plan_efficiency_score: float = 1.0
    constraint_compliance: bool = True
    alternative_plans: List[Dict[str, Any]] = Field(default_factory=list)


class SharedCognitiveState(BaseModel):
    """Unified shared cognitive context binding all cognitive subsystems."""
    state_id: str = Field(default_factory=lambda: f"cog_{uuid.uuid4().hex[:8]}")
    perception: PerceptionResult
    context: ContextFrame
    working_memory: WorkingMemoryState
    world_state: WorldState
    reasoning: ReasoningTrace
    plan: Optional[Plan] = None
    episodic_context: List[Dict[str, Any]] = Field(default_factory=list)
    semantic_rules: List[Dict[str, Any]] = Field(default_factory=list)
    procedural_workflows: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class CognitiveEngineResult(BaseModel):
    """Unified output from the integrated Cognitive AI Engine."""
    execution_id: str = Field(default_factory=lambda: f"exec_{uuid.uuid4().hex[:8]}")
    decision: Any
    confidence: float
    calibrated_confidence: float
    priority: str
    risk: float
    operational_score: Optional[float] = None
    predicted_outcomes: Dict[str, Any] = Field(default_factory=dict)
    selected_plan: Optional[Plan] = None
    reasoning_evidence: ReasoningTrace
    world_state_summary: Dict[str, Any] = Field(default_factory=dict)
    abstain: bool = False
    abstention_reason: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    execution_recommendation: str
    latency_ms: float = 0.0
