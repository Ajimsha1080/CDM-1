"""World Model Subsystem for CDM-1 Cognitive AI Engine.
Maintains an explicit internal representation of the operational environment,
its entities, relations, active constraints, and transition dynamics S_{t+1} = T(S_t, A_t).
"""

from typing import Dict, List, Optional, Any, Tuple
import time
from cdm.cognitive.state import WorldState, WorldEntity, WorldRelation


class WorldModel:
    """Explicit world model representing the operational environment and simulating action transitions."""

    def __init__(self):
        self.state = WorldState()

    def update_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        inferred_properties: Optional[Dict[str, Any]] = None,
        uncertainty: float = 0.0,
    ) -> WorldEntity:
        """Adds or updates an entity in the world state."""
        existing = self.state.entities.get(entity_id)
        if existing:
            existing.properties.update(properties)
            if inferred_properties:
                existing.inferred_properties.update(inferred_properties)
            existing.uncertainty = uncertainty
            existing.last_updated = time.time()
            return existing
        else:
            entity = WorldEntity(
                entity_id=entity_id,
                entity_type=entity_type,
                properties=properties,
                inferred_properties=inferred_properties or {},
                uncertainty=uncertainty,
                last_updated=time.time(),
            )
            self.state.entities[entity_id] = entity
            self._update_overall_uncertainty()
            return entity

    def add_relation(self, source_id: str, relation_type: str, target_id: str, confidence: float = 1.0) -> None:
        """Adds or updates a directed relation between entities."""
        # Avoid duplicate relations
        for r in self.state.relations:
            if r.source_id == source_id and r.relation_type == relation_type and r.target_id == target_id:
                r.confidence = confidence
                return
        self.state.relations.append(
            WorldRelation(source_id=source_id, relation_type=relation_type, target_id=target_id, confidence=confidence)
        )

    def set_environment_metric(self, metric_name: str, value: float) -> None:
        """Updates a global environment metric (e.g., system latency, remaining SLA, error rate)."""
        self.state.environment_metrics[metric_name] = value

    def register_constraint(self, constraint: str) -> None:
        """Registers an active operational or policy constraint."""
        if constraint not in self.state.active_constraints:
            self.state.active_constraints.append(constraint)

    def simulate_transition(self, action: str, action_params: Optional[Dict[str, Any]] = None) -> WorldState:
        """Counterfactual forward transition model: S_{t+1} = T(S_t, Action).

        Simulates how taking a candidate action alters the environment state without mutating ground truth.
        """
        params = action_params or {}
        simulated = self.state.model_copy(deep=True)

        action_clean = action.lower()
        if "escalate" in action_clean:
            # Escalation reduces SLA remaining hours and increases operational urgency
            sla = simulated.environment_metrics.get("sla_remaining_hours", 12.0)
            simulated.environment_metrics["sla_remaining_hours"] = max(0.5, sla * 0.5)
            simulated.environment_metrics["escalation_active"] = 1.0
        elif "refund" in action_clean:
            # Refund adjusts monetary risk to settled
            simulated.environment_metrics["refund_issued"] = 1.0
            simulated.environment_metrics["monetary_risk"] = 0.0
        elif "block" in action_clean or "quarantine" in action_clean:
            # Security quarantine protects system integrity
            simulated.environment_metrics["system_exposure"] = 0.0
            simulated.environment_metrics["service_disruption_risk"] = 0.8
        elif "auto_resolve" in action_clean:
            # Resolution decreases pending queue
            simulated.environment_metrics["resolved"] = 1.0

        return simulated

    def _update_overall_uncertainty(self) -> None:
        """Computes average uncertainty across all world entities."""
        if not self.state.entities:
            self.state.overall_state_uncertainty = 0.0
            return
        total = sum(e.uncertainty for e in self.state.entities.values())
        self.state.overall_state_uncertainty = total / len(self.state.entities)

    def get_state(self) -> WorldState:
        """Returns the current world state."""
        return self.state.model_copy(deep=True)
