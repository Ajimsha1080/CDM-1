"""Feedback Collection, Durable Storage, and Decision Quality Auditing for CDM-1.
Gathers verified outcomes from external telemetry, tracks discrepancy between
predicted outcomes and verified facts, stores records durably in SQLite with tenant isolation,
enforces collision-resistant UUIDs, prevents accidental overwrites, and stages human-reviewed
feedback batches for controlled fine-tuning.
"""

from typing import Dict, List, Optional, Any, Union
import time
import os
import json
import uuid
import sqlite3
from pydantic import BaseModel, Field


class FeedbackEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"fb_{uuid.uuid4().hex}")
    tenant_id: str = "default_tenant"
    decision_id: str
    predicted_decision: Dict[str, Any]
    predicted_outcome: Dict[str, Any]
    verified_actual_outcome: Dict[str, Any]
    decision_success: bool
    discrepancy_score: float
    timestamp: float = Field(default_factory=time.time)
    review_status: str = "pending"  # 'pending', 'verified', 'rejected_for_training'
    reviewer_id: Optional[str] = None
    verified_target_label: Optional[Union[str, int, Dict[str, Any]]] = None


class FeedbackCollector:
    """Manages verified outcome collection, discrepancy detection, and approved dataset staging with SQLite persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.events: List[FeedbackEvent] = []
        if self.db_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._init_db()
            self._load_persisted_events()

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
                        review_status TEXT NOT NULL,
                        reviewer_id TEXT
                    )
                """)
                # Handle existing tables that may lack reviewer_id column
                cols = [r[1] for r in conn.execute("PRAGMA table_info(feedback_events)").fetchall()]
                if "reviewer_id" not in cols:
                    conn.execute("ALTER TABLE feedback_events ADD COLUMN reviewer_id TEXT")

                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback_events (tenant_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_events (review_status)")
        finally:
            conn.close()

    def _load_persisted_events(self) -> None:
        """Retrieves persisted feedback records from SQLite after process restart."""
        if not self.db_path or not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                rows = conn.execute("SELECT * FROM feedback_events ORDER BY timestamp ASC").fetchall()
                self.events = []
                for r in rows:
                    event = FeedbackEvent(
                        event_id=r["event_id"],
                        tenant_id=r["tenant_id"],
                        decision_id=r["decision_id"],
                        predicted_decision=json.loads(r["predicted_decision_json"]),
                        predicted_outcome=json.loads(r["predicted_outcome_json"]),
                        verified_actual_outcome=json.loads(r["verified_actual_outcome_json"]),
                        decision_success=bool(r["decision_success"]),
                        discrepancy_score=float(r["discrepancy_score"]),
                        timestamp=float(r["timestamp"]),
                        review_status=r["review_status"],
                        reviewer_id=r["reviewer_id"] if "reviewer_id" in r.keys() else None,
                    )
                    self.events.append(event)
        finally:
            conn.close()

    def record_feedback(
        self,
        decision_id: Optional[str] = None,
        predicted_decision: Optional[Dict[str, Any]] = None,
        predicted_outcome: Optional[Dict[str, Any]] = None,
        actual_outcome: Optional[Dict[str, Any]] = None,
        is_success: bool = True,
        tenant_id: str = "default_tenant",
        review_status: str = "pending",
        event_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        event_obj: Optional[Union[FeedbackEvent, Dict[str, Any]]] = None,
    ) -> FeedbackEvent:
        """Records feedback supporting both individual parameters, raw dicts, and Pydantic FeedbackEvent models.
        Strictly enforces collision-resistant IDs and prevents silent overwriting of existing events.
        """
        # Handle dict or Pydantic event_obj
        if event_obj is not None:
            if isinstance(event_obj, FeedbackEvent):
                event = event_obj
            elif isinstance(event_obj, dict):
                event = FeedbackEvent(**event_obj)
            else:
                raise TypeError(f"Unsupported event_obj type: {type(event_obj)}")
        elif isinstance(decision_id, FeedbackEvent):
            event = decision_id
        elif isinstance(decision_id, dict):
            event = FeedbackEvent(**decision_id)
        else:
            if decision_id is None:
                raise ValueError("decision_id is required when creating a new feedback record.")
            pred_dec = predicted_decision or {}
            act_out = actual_outcome or {}
            pred_bin = pred_dec.get("binary_prediction", pred_dec.get("decision", 0))
            actual_bin = act_out.get("binary_target", act_out.get("observed_outcome_binary", 0))
            discrepancy = 1.0 if pred_bin != actual_bin else 0.0

            cid = event_id or f"fb_{uuid.uuid4().hex}"

            event = FeedbackEvent(
                event_id=cid,
                tenant_id=tenant_id,
                decision_id=decision_id,
                predicted_decision=pred_dec,
                predicted_outcome=predicted_outcome or {},
                verified_actual_outcome=act_out,
                decision_success=is_success,
                discrepancy_score=discrepancy,
                review_status=review_status,
                reviewer_id=reviewer_id,
            )

        # Duplicate check: prevent silent overwrite of existing records
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError(
                f"Duplicate feedback event ID '{event.event_id}' already exists in memory. "
                "Silent overwrites are forbidden."
            )

        # Persist to durable SQLite database if configured
        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    # Strict INSERT without OR REPLACE to prevent silent collision overwrites
                    try:
                        conn.execute(
                            """
                            INSERT INTO feedback_events
                            (event_id, tenant_id, decision_id, predicted_decision_json, predicted_outcome_json,
                             verified_actual_outcome_json, decision_success, discrepancy_score, timestamp, review_status, reviewer_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                                event.reviewer_id,
                            ),
                        )
                    except sqlite3.IntegrityError as err:
                        raise ValueError(
                            f"Duplicate feedback event ID '{event.event_id}' already exists in SQLite storage: {err}"
                        )
            finally:
                conn.close()

        self.events.append(event)
        return event

    def get_failure_cases(self, tenant_id: Optional[str] = None) -> List[FeedbackEvent]:
        """Isolate instances where decisions produced negative or unexpected outcomes."""
        candidates = self.events
        if tenant_id:
            candidates = [e for e in candidates if e.tenant_id == tenant_id]
        return [e for e in candidates if not e.decision_success or e.discrepancy_score > 0]

    def update_review_status(
        self,
        event_id: str,
        new_status: str,
        tenant_id: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> bool:
        """Controlled human-in-the-loop review status update ('verified' or 'rejected_for_training')."""
        updated = False
        rev_id = reviewer_id or "human_auditor"
        for e in self.events:
            if e.event_id == event_id and (tenant_id is None or e.tenant_id == tenant_id):
                e.review_status = new_status
                if new_status == "verified":
                    e.reviewer_id = rev_id
                updated = True

        if self.db_path:
            conn = sqlite3.connect(self.db_path)
            try:
                with conn:
                    if tenant_id:
                        cur = conn.execute(
                            "UPDATE feedback_events SET review_status = ?, reviewer_id = ? WHERE event_id = ? AND tenant_id = ?",
                            (new_status, rev_id if new_status == "verified" else None, event_id, tenant_id),
                        )
                    else:
                        cur = conn.execute(
                            "UPDATE feedback_events SET review_status = ?, reviewer_id = ? WHERE event_id = ?",
                            (new_status, rev_id if new_status == "verified" else None, event_id),
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
        approved = [e for e in candidates if e.review_status == "verified" and e.reviewer_id]
        if len(approved) >= min_events:
            return approved
        return None
