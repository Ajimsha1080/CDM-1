"""Feedback Collection, Durable Storage, and Decision Quality Auditing for CDM-1.
Gathers verified outcomes from external telemetry, tracks discrepancy between
predicted outcomes and verified facts, stores records durably in SQLite with tenant isolation,
and stages human-reviewed feedback batches for controlled fine-tuning.
"""

from typing import Dict, List, Optional, Any
import time
import os
import json
import sqlite3
from pydantic import BaseModel, Field


class FeedbackEvent(BaseModel):
    event_id: str
    tenant_id: str = "default_tenant"
    decision_id: str
    predicted_decision: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    verified_actual_outcome: Dict[str, Any]
    decision_success: bool
    discrepancy_score: float
    timestamp: float = Field(default_factory=time.time)
    review_status: str = "pending"  # 'pending', 'verified', 'rejected_for_training'


class FeedbackCollector:
    """Manages verified outcome collection, discrepancy detection, and approved dataset staging with SQLite persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.events: List[FeedbackEvent] = []
        if self.db_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._init_db()

    def _init_db(self) -> None:
        if not self.db_path:
            return
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        event_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        decision_id TEXT NOT NULL,
                        predicted_decision_json TEXT NOT NULL,
                        predicted_outcome_json TEXT NOT NULL,
                        verified_actual_outcome_json TEXT NOT NULL,
                        decision_success INTEGER NOT NULL,
                        discrepancy_score REAL NOT NULL,
                        timestamp REAL NOT NULL,
                        review_status TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback_events (tenant_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_events (review_status)")
        finally:
            conn.close()

    def record_feedback(
        self,
        decision_id: str,
        predicted_decision: Dict[str, Any],
        predicted_outcome: Dict[str, Any],
        actual_outcome: Dict[str, Any],
        is_success: bool,
        tenant_id: str = "default_tenant",
        review_status: str = "pending",
    ) -> FeedbackEvent:
        # Measure discrepancy between predicted decision and actual outcome
        pred_bin = predicted_decision.get("binary_prediction", predicted_decision.get("decision", 0))
        actual_bin = actual_outcome.get("binary_target", actual_outcome.get("observed_outcome_binary", 0))
        discrepancy = 1.0 if pred_bin != actual_bin else 0.0

        event = FeedbackEvent(
            event_id=f"fb_{len(self.events) + 1:05d}",
            tenant_id=tenant_id,
            decision_id=decision_id,
            predicted_decision=predicted_decision,
            predicted_outcome=predicted_outcome,
            verified_actual_outcome=actual_outcome,
            decision_success=is_success,
            discrepancy_score=discrepancy,
            review_status=review_status,
        )
        self.events.append(event)

        # Persist to durable SQLite database if configured
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO feedback_events
                        (event_id, tenant_id, decision_id, predicted_decision_json, predicted_outcome_json,
                         verified_actual_outcome_json, decision_success, discrepancy_score, timestamp, review_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.tenant_id,
                            event.decision_id,
                            json.dumps(event.predicted_decision),
                            json.dumps(event.predicted_outcome),
                            json.dumps(event.verified_actual_outcome),
                            1 if event.decision_success else 0,
                            event.discrepancy_score,
                            event.timestamp,
                            event.review_status,
                        ),
                    )
            finally:
                conn.close()

        return event

    def get_failure_cases(self, tenant_id: Optional[str] = None) -> List[FeedbackEvent]:
        """Isolate instances where decisions produced negative or unexpected outcomes."""
        candidates = self.events
        if tenant_id:
            candidates = [e for e in candidates if e.tenant_id == tenant_id]
        return [e for e in candidates if not e.decision_success or e.discrepancy_score > 0]

    def update_review_status(self, event_id: str, new_status: str, tenant_id: Optional[str] = None) -> bool:
        """Controlled human-in-the-loop review status update ('verified' or 'rejected_for_training')."""
        updated = False
        for e in self.events:
            if e.event_id == event_id and (tenant_id is None or e.tenant_id == tenant_id):
                e.review_status = new_status
                updated = True

        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    if tenant_id:
                        cur = conn.execute(
                            "UPDATE feedback_events SET review_status = ? WHERE event_id = ? AND tenant_id = ?",
                            (new_status, event_id, tenant_id),
                        )
                    else:
                        cur = conn.execute(
                            "UPDATE feedback_events SET review_status = ? WHERE event_id = ?",
                            (new_status, event_id),
                        )
                    if cur.rowcount > 0:
                        updated = True
            finally:
                conn.close()

        return updated

    def stage_approved_fine_tuning_batch(
        self, min_events: int = 10, tenant_id: Optional[str] = None
    ) -> Optional[List[FeedbackEvent]]:
        """Extracts validated events that have been verified by human review for scheduled fine-tuning."""
        candidates = self.events
        if tenant_id:
            candidates = [e for e in candidates if e.tenant_id == tenant_id]
        approved = [e for e in candidates if e.review_status == "verified"]
        if len(approved) >= min_events:
            return approved
        return None
