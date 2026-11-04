"""Feedback Collection and Decision Quality Auditing for CDM-1.
Gathers verified outcomes from external telemetry, tracks discrepancy between
predicted outcomes and verified facts, and flags systematic failure modes.
"""

from typing import Dict, List, Optional, Any
import time
from pydantic import BaseModel, Field


class FeedbackEvent(BaseModel):
    event_id: str
    decision_id: str
    predicted_decision: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    verified_actual_outcome: Dict[str, Any]
    decision_success: bool
    discrepancy_score: float
    timestamp: float = Field(default_factory=time.time)
    review_status: str = "pending"  # pending, verified, rejected_for_training


class FeedbackCollector:
    """Manages verified outcome collection, discrepancy detection, and approved dataset staging."""

    def __init__(self):
        self.events: List[FeedbackEvent] = []

    def record_feedback(
        self,
        decision_id: str,
        predicted_decision: Dict[str, Any],
        predicted_outcome: Dict[str, Any],
        actual_outcome: Dict[str, Any],
        is_success: bool,
    ) -> FeedbackEvent:
        # Measure error between predicted and actual
        pred_bin = predicted_decision.get("binary_prediction", 0)
        actual_bin = actual_outcome.get("binary_target", 0)
        discrepancy = 1.0 if pred_bin != actual_bin else 0.0

        event = FeedbackEvent(
            event_id=f"fb_{len(self.events) + 1:05d}",
            decision_id=decision_id,
            predicted_decision=predicted_decision,
            predicted_outcome=predicted_outcome,
            verified_actual_outcome=actual_outcome,
            decision_success=is_success,
            discrepancy_score=discrepancy,
        )
        self.events.append(event)
        return event

    def get_failure_cases(self) -> List[FeedbackEvent]:
        """Isolate instances where decisions produced negative or unexpected outcomes."""
        return [e for e in self.events if not e.decision_success or e.discrepancy_score > 0]

    def stage_approved_fine_tuning_batch(self, min_events: int = 10) -> Optional[List[FeedbackEvent]]:
        """Extracts validated events that have been verified by human review for scheduled fine-tuning."""
        approved = [e for e in self.events if e.review_status == "verified"]
        if len(approved) >= min_events:
            return approved
        return None
