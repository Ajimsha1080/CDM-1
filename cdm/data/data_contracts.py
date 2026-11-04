"""CDM-1 Standardized Decision Dataset Contract.
Defines schema and contracts for multi-domain decision examples, candidate actions,
constraints, outcomes, and provenance.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class CandidateAction(BaseModel):
    action_id: str
    name: str
    description: str
    expected_impact: Optional[str] = None
    estimated_cost: Optional[float] = 0.0


class PolicyConstraint(BaseModel):
    constraint_id: str
    description: str
    prohibited_actions: List[str] = Field(default_factory=list)
    required_conditions: Dict[str, Any] = Field(default_factory=dict)
    is_hard_constraint: bool = True


class OutcomeLabel(BaseModel):
    observed_outcome_binary: Optional[int] = None       # e.g., 1=success, 0=failure
    observed_outcome_multiclass: Optional[int] = None   # categorical outcome index
    observed_resolution_time_hours: Optional[float] = None
    observed_risk_realized: Optional[float] = None      # 0.0 to 1.0


class TargetDecision(BaseModel):
    binary_decision: Optional[int] = None        # 0=NO, 1=YES
    selected_candidate_id: Optional[str] = None  # Choice decision
    decision_score: Optional[float] = None       # Continuous score (e.g. 0-100)
    priority_level: Optional[int] = None         # 0=Low, 1=Medium, 2=High, 3=Critical
    risk_level: Optional[int] = None             # 0=Low, 1=Moderate, 2=High, 3=Severe


class LabelProvenance(BaseModel):
    source_type: str = "curated_expert"          # curated_expert, historical_telemetry, policy_rule
    verifier_id: Optional[str] = None
    verification_method: str = "cross_validated"
    synthetic: bool = False
    notes: Optional[str] = None


class DecisionExample(BaseModel):
    example_id: str
    domain: str                                  # communication, customer_support, business_ops, retail
    input_text: str
    structured_features: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[str] = None
    objective: str = "Make optimal, risk-bounded operational decision"
    candidate_actions: List[CandidateAction] = Field(default_factory=list)
    constraints: List[PolicyConstraint] = Field(default_factory=list)
    target_decision: TargetDecision
    evidence: List[str] = Field(default_factory=list)
    label_provenance: LabelProvenance = Field(default_factory=LabelProvenance)
    outcome_label: Optional[OutcomeLabel] = None
    temporal_timestamp: Optional[str] = None
