"""CDM-1 Causal Analysis and Counterfactual Reasoning Subsystem.
Explicitly models directed cause-and-effect relationships, distinguishes correlation
from temporal association and causal mechanisms, supports causal hypotheses,
and evaluates justified counterfactual queries without fabricating ungrounded links.
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
import time
import uuid
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger("cdm.causal")


class AssociationType(str, Enum):
    """Categorizes the nature of an observed association."""
    CORRELATION = "correlation"                  # Statistical co-occurrence without directional mechanism
    TEMPORAL_SEQUENCE = "temporal_sequence"      # A preceded B in time, but confounding/coincidence unruled
    CAUSAL_MECHANISM = "causal_mechanism"        # Direct intervention do(X) -> Y via verified mechanism/invariant


class CausalVariable(BaseModel):
    """A variable or event in the causal graph."""
    name: str
    value: Any = None
    variable_type: str = "state_metric"  # 'action', 'state_metric', 'external_event', 'outcome'
    observed_at: float = Field(default_factory=time.time)


class CausalEdge(BaseModel):
    """Directed edge representing an association or causal relationship between variables."""
    source: str
    target: str
    association_type: AssociationType
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_sources: List[str] = Field(default_factory=list)
    mechanism_description: str = ""
    is_intervention_validated: bool = False


class CausalHypothesis(BaseModel):
    """Testable causal claim: 'Action X causes Outcome Y under Condition C'."""
    hypothesis_id: str = Field(default_factory=lambda: f"ch_{uuid.uuid4().hex[:8]}")
    cause: str
    effect: str
    association_type: AssociationType = AssociationType.CORRELATION
    confidence: float = 0.5
    supporting_evidence: List[str] = Field(default_factory=list)
    counter_evidence: List[str] = Field(default_factory=list)
    status: str = "formulated"  # 'formulated', 'supported', 'refuted', 'intervention_verified'


class CounterfactualQuery(BaseModel):
    """Query asking what would have happened if a different action had been taken."""
    action_taken: str
    observed_outcome: Dict[str, Any]
    hypothetical_action: str
    context_state: Dict[str, Any] = Field(default_factory=dict)


class CounterfactualResult(BaseModel):
    """Ground-truth or model-derived counterfactual prediction."""
    hypothetical_action: str
    predicted_outcome: Dict[str, Any]
    confidence: float
    justified: bool
    causal_pathway: List[str] = Field(default_factory=list)
    limitation_note: Optional[str] = None


class CausalEngine:
    """Explicit causal reasoning and counterfactual analysis engine.
    Maintains a directed causal graph, verifies intervention mechanisms,
    distinguishes correlation from causation, and evaluates candidate consequences.
    """

    def __init__(self):
        self.edges: Dict[Tuple[str, str], CausalEdge] = {}
        self.variables: Dict[str, CausalVariable] = {}
        self.temporal_history: List[Tuple[str, Any, float]] = []
        self._init_domain_mechanisms()

    def _init_domain_mechanisms(self) -> None:
        """Seeds verified physical and domain causal mechanisms from specification invariants."""
        # Verified actions that directly cause state transitions
        verified_mechanisms = [
            # Action -> Direct Outcome (Intervention-validated causal mechanisms)
            ("issue_refund", "refund_issued", AssociationType.CAUSAL_MECHANISM, 1.0, "Financial ledger debit and dispute settlement transaction", True),
            ("issue_refund", "monetary_risk", AssociationType.CAUSAL_MECHANISM, 0.95, "Settling refund directly resolves pending monetary dispute", True),
            ("escalate_to_tier2", "escalation_active", AssociationType.CAUSAL_MECHANISM, 1.0, "Ticket dispatch to Tier 2 queue creates escalation state", True),
            ("escalate_to_tier2", "sla_remaining_hours", AssociationType.CAUSAL_MECHANISM, 0.90, "Priority elevation halves remaining SLA window", True),
            ("auto_resolve", "resolved", AssociationType.CAUSAL_MECHANISM, 0.95, "Autonomous ticket closing marks state resolved", True),
            ("isolate_cluster", "system_exposure", AssociationType.CAUSAL_MECHANISM, 0.95, "Network partition immediately restricts external exposure", True),
            ("write_file", "file_exists", AssociationType.CAUSAL_MECHANISM, 1.0, "Filesystem write creates or modifies file on disk", True),
            ("edit_file", "file_contains", AssociationType.CAUSAL_MECHANISM, 1.0, "String replacement modifies file contents on disk", True),

            # Non-causal correlations (Statistical co-occurrences without intervention proof)
            ("high_monetary_value", "distress_level", AssociationType.CORRELATION, 0.65, "Customers disputing large amounts frequently display distress words", False),
            ("high_turn_count", "customer_frustration", AssociationType.CORRELATION, 0.70, "Multi-turn dialogues correlate with complex or contested issues", False),
            ("error_code_500", "user_complaint", AssociationType.TEMPORAL_SEQUENCE, 0.85, "Server internal errors temporally precede user incident tickets", False),
        ]

        for src, tgt, atype, conf, desc, validated in verified_mechanisms:
            self.edges[(src, tgt)] = CausalEdge(
                source=src,
                target=tgt,
                association_type=atype,
                confidence=conf,
                evidence_sources=["system_domain_specification"],
                mechanism_description=desc,
                is_intervention_validated=validated,
            )

    def register_variable(self, name: str, value: Any, var_type: str = "state_metric") -> None:
        now = time.time()
        self.variables[name] = CausalVariable(name=name, value=value, variable_type=var_type, observed_at=now)
        self.temporal_history.append((name, value, now))

    def add_causal_relation(
        self,
        source: str,
        target: str,
        association_type: AssociationType,
        confidence: float,
        mechanism: str = "",
        is_intervention_validated: bool = False,
        evidence: Optional[List[str]] = None,
    ) -> CausalEdge:
        edge = CausalEdge(
            source=source,
            target=target,
            association_type=association_type,
            confidence=min(max(confidence, 0.0), 1.0),
            mechanism_description=mechanism,
            is_intervention_validated=is_intervention_validated,
            evidence_sources=evidence or ["manual_registration"],
        )
        self.edges[(source, target)] = edge
        return edge

    def get_relationship(self, source: str, target: str) -> Optional[CausalEdge]:
        """Looks up the known direct relationship between two variables."""
        return self.edges.get((source, target))

    def evaluate_causality(self, cause: str, effect: str) -> CausalHypothesis:
        """Evaluates whether cause -> effect is a genuine causal mechanism, temporal sequence, or mere correlation."""
        edge = self.get_relationship(cause, effect)
        if edge:
            status = "intervention_verified" if edge.is_intervention_validated else "supported"
            return CausalHypothesis(
                cause=cause,
                effect=effect,
                association_type=edge.association_type,
                confidence=edge.confidence,
                supporting_evidence=[edge.mechanism_description] + edge.evidence_sources,
                counter_evidence=[],
                status=status,
            )

        # Check temporal sequence in recorded event log
        cause_times = [t for n, v, t in self.temporal_history if n == cause]
        effect_times = [t for n, v, t in self.temporal_history if n == effect]

        if cause_times and effect_times and min(cause_times) < min(effect_times):
            # A preceded B, but no verified causal mechanism exists
            return CausalHypothesis(
                cause=cause,
                effect=effect,
                association_type=AssociationType.TEMPORAL_SEQUENCE,
                confidence=0.40,
                supporting_evidence=["Observed temporal precedence in session history"],
                counter_evidence=["No validated causal mechanism or intervention trial connecting variables"],
                status="formulated",
            )

        return CausalHypothesis(
            cause=cause,
            effect=effect,
            association_type=AssociationType.CORRELATION,
            confidence=0.10,
            supporting_evidence=[],
            counter_evidence=["No empirical correlation, temporal precedence, or causal pathway found"],
            status="refuted",
        )

    def analyze_candidate_consequences(
        self,
        candidate_actions: List[str],
        current_metrics: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyzes verified causal consequences for candidate actions versus ungrounded correlations."""
        consequences = {}
        for action in candidate_actions:
            action_clean = action.lower()
            direct_effects = []
            correlations = []

            for (src, tgt), edge in self.edges.items():
                if src.lower() in action_clean or action_clean in src.lower():
                    if edge.association_type == AssociationType.CAUSAL_MECHANISM:
                        direct_effects.append({
                            "target_variable": tgt,
                            "mechanism": edge.mechanism_description,
                            "confidence": edge.confidence,
                            "is_intervention_validated": edge.is_intervention_validated,
                        })
                    else:
                        correlations.append({
                            "target_variable": tgt,
                            "type": edge.association_type.value,
                            "confidence": edge.confidence,
                        })

            consequences[action] = {
                "has_verified_causal_mechanism": len(direct_effects) > 0,
                "direct_causal_effects": direct_effects,
                "associated_correlations": correlations,
            }
        return consequences

    def evaluate_counterfactual(
        self,
        query: CounterfactualQuery,
        world_model: Optional[Any] = None,
    ) -> CounterfactualResult:
        """Evaluates a counterfactual query: 'What would have occurred under hypothetical_action?'
        Only permits counterfactual claims when supported by verified causal mechanisms or a transition model.
        Explicitly flags ungrounded or speculative counterfactuals as unjustified.
        """
        hyp_act = query.hypothetical_action
        hyp_clean = hyp_act.lower()

        # Check if verified causal mechanism exists for hypothetical action
        relevant_mechanisms = [
            edge for (src, tgt), edge in self.edges.items()
            if (src.lower() in hyp_clean or hyp_clean in src.lower()) and edge.association_type == AssociationType.CAUSAL_MECHANISM
        ]

        if not relevant_mechanisms and world_model is None:
            return CounterfactualResult(
                hypothetical_action=hyp_act,
                predicted_outcome={"status": "unknown"},
                confidence=0.0,
                justified=False,
                causal_pathway=[],
                limitation_note="Counterfactual analysis unjustified: No verified causal mechanism or world transition model available for hypothetical action.",
            )

        # If world_model available, run counterfactual transition simulation
        predicted = {}
        pathway = []
        conf = 0.85

        if world_model and hasattr(world_model, "simulate_transition"):
            sim_state = world_model.simulate_transition(action=hyp_act)
            predicted = sim_state.environment_metrics
            pathway = [f"WorldModel.simulate_transition({hyp_act}) -> metrics updated"]
        else:
            for mech in relevant_mechanisms:
                predicted[mech.target] = f"Altered via mechanism: {mech.mechanism_description}"
                pathway.append(f"{mech.source} -[{mech.association_type.value}]-> {mech.target}")
            conf = min([m.confidence for m in relevant_mechanisms]) if relevant_mechanisms else 0.5

        return CounterfactualResult(
            hypothetical_action=hyp_act,
            predicted_outcome=predicted,
            confidence=round(conf, 4),
            justified=True,
            causal_pathway=pathway,
            limitation_note=None,
        )
