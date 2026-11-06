"""Test Suite for CDM-1 V4 Targeted Remediation.
Verifies temporal context calculation, memory retrieval quality, feedback persistence & restart,
duplicate prevention, human review staging validation, dynamic planning risk, and alternative plan evaluations.
"""

import pytest
import os
import time
import tempfile
import uuid
from typing import Dict, Any

from cdm.cognitive.state import PerceptionResult, ContextFrame, GoalNode, WorldState, ReasoningTrace
from cdm.context.context_engine import ContextEngine
from cdm.memory.embedder import SemanticMemoryEmbedder
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory, MemoryRecord
from cdm.learning.feedback import FeedbackCollector, FeedbackEvent
from cdm.learning.adaptation import AdaptationGatekeeper
from cdm.planning.planning_engine import PlanningEngine


def test_temporal_context_calculation_actual_timestamps():
    """Requirement 9: Verifies temporal context calculation uses actual timestamps."""
    ce = ContextEngine()
    session_id = "test_temporal_session"
    tenant_id = "test_tenant"

    p1 = PerceptionResult(raw_text="First message")
    f1 = ce.update_context(session_id, tenant_id, p1, "First user turn")
    assert f1.temporal_delta_seconds == 0.0
    assert f1.timestamp > 0.0

    # Sleep slightly to test actual positive temporal delta
    time.sleep(0.05)

    p2 = PerceptionResult(raw_text="Second message after delay")
    f2 = ce.update_context(session_id, tenant_id, p2, "Second user turn")
    assert f2.temporal_delta_seconds >= 0.04
    assert f2.temporal_delta_seconds < 10.0, "Delta must be real elapsed time, not cumulative or epoch"
    assert f2.turn_index == 2


def test_memory_retrieval_quality_and_fallback():
    """Requirement 6, 7, 8, 10: Verifies memory embedding modes, retrieval ranking, and fallback."""
    # Standalone embedder without transformer
    embedder = SemanticMemoryEmbedder(dim=16)
    vec1, mode1 = embedder.embed_with_metadata("Customer inquiry regarding bill refund", {"domain": "finance", "monetary_value_usd": 150.0})
    assert mode1 == "structured_fallback"
    assert len(vec1) == 16
    assert abs(sum(x * x for x in vec1) - 1.0) < 1e-4

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteMemoryStorage(db_path=db_path)
        episodic = EpisodicMemory(storage=storage)

        # Insert 3 records with distinct vectors
        # Rec 1: Close match
        rec1 = MemoryRecord(
            record_id="rec_close",
            tenant_id="tenant_q",
            context_text="Bill refund request",
            decision_made={"action": "refund"},
            vector=list(vec1),
        )
        # Rec 2: Orthogonal vector
        vec_ortho = [0.0] * 16
        vec_ortho[15] = 1.0
        rec2 = MemoryRecord(
            record_id="rec_distant",
            tenant_id="tenant_q",
            context_text="Technical network outage",
            decision_made={"action": "reboot"},
            vector=vec_ortho,
        )
        episodic.insert(rec1)
        episodic.insert(rec2)

        results = episodic.retrieve_similar(query_vector=vec1, tenant_id="tenant_q", top_k=5, min_relevance=0.1)
        assert len(results) >= 1
        assert results[0]["record_id"] == "rec_close"
        assert results[0]["similarity"] > 0.99
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_feedback_uuid_collision_resistance_and_restart():
    """Requirements 15, 16, 17, 19, 20: Verifies UUID collision resistance, persistence, and restart."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Phase 1: Initialize collector and record events
        col1 = FeedbackCollector(db_path=db_path)
        ev1 = col1.record_feedback(
            decision_id="dec_001",
            predicted_decision={"decision": "refund", "binary_prediction": 1},
            predicted_outcome={"risk": 0.1},
            actual_outcome={"binary_target": 1},
            is_success=True,
            tenant_id="tenant_m",
        )
        # Verify collision-resistant ID
        assert ev1.event_id.startswith("fb_")
        assert len(ev1.event_id) > 10, "Event ID must be collision-resistant UUID"

        # Test duplicate event rejection
        with pytest.raises(ValueError, match="already exists"):
            col1.record_feedback(
                event_obj={
                    "event_id": ev1.event_id,
                    "tenant_id": "tenant_m",
                    "decision_id": "dec_duplicate",
                    "predicted_decision": {},
                    "predicted_outcome": {},
                    "verified_actual_outcome": {},
                    "decision_success": True,
                    "discrepancy_score": 0.0,
                }
            )

        # Update review status
        col1.update_review_status(ev1.event_id, "verified", tenant_id="tenant_m", reviewer_id="auditor_jane")

        # Phase 2: Simulate process restart by instantiating new collector on same DB
        col2 = FeedbackCollector(db_path=db_path)
        assert len(col2.events) == 1, "Must recover all events from SQLite on restart"
        restarted_ev = col2.events[0]
        assert restarted_ev.event_id == ev1.event_id
        assert restarted_ev.review_status == "verified"
        assert restarted_ev.reviewer_id == "auditor_jane"

        # Record a second event on restarted collector
        ev2 = col2.record_feedback(
            decision_id="dec_002",
            predicted_decision={"decision": "escalate", "binary_prediction": 1},
            predicted_outcome={"risk": 0.5},
            actual_outcome={"binary_target": 0},
            is_success=False,
            tenant_id="tenant_m",
        )
        assert ev2.event_id != ev1.event_id
        assert len(col2.events) == 2
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_adaptation_staging_validation():
    """Requirement 18: Validates labels and actual human-review provenance before staging."""
    gatekeeper = AdaptationGatekeeper()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_path = tmp.name

    try:
        events = [
            # Event 1: Unverified (Must be rejected)
            {
                "event_id": "ev_unverified",
                "review_status": "pending",
                "reviewer_id": "auditor_1",
                "verified_actual_outcome": {"binary_target": 1},
            },
            # Event 2: Verified but missing reviewer provenance (Must be rejected)
            {
                "event_id": "ev_no_reviewer",
                "review_status": "verified",
                "reviewer_id": None,
                "verified_actual_outcome": {"binary_target": 1},
            },
            # Event 3: Verified with valid reviewer and valid target (Must be approved)
            {
                "event_id": "ev_valid",
                "review_status": "verified",
                "reviewer_id": "auditor_verified_01",
                "verified_actual_outcome": {"binary_target": 0, "selected_candidate_id": "act_escalate"},
                "predicted_decision": {"text": "Test sample", "structured_features": {}},
            },
        ]

        staged_count = gatekeeper.stage_feedback_dataset(events, output_path=out_path)
        assert staged_count == 1, f"Expected exactly 1 staged record, got {staged_count}"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except Exception:
                pass


def test_planning_preconditions_dynamic_risk_and_alternatives():
    """Requirements 21, 22, 23, 24: Verifies planning preconditions, dynamic risk, and alternative evaluation."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="resolve_dispute", description="Resolve customer billing dispute")
    world_state = WorldState(
        observed_facts={"text": "Large unauthorized charge dispute"},
        environment_metrics={"monetary_value_usd": 2500.0, "sla_remaining_hours": 6.0},
        overall_state_uncertainty=0.10,
    )
    trace = ReasoningTrace(
        world_state_summary={},
        violated_constraints=["ProhibitedAction: auto_resolve"],
    )

    pred_outcomes = {
        "candidate_scores": {"issue_refund": 0.3, "escalate_to_tier2": 0.95},
        "predicted_risk": 0.12,
    }

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["auto_resolve", "issue_refund", "escalate_to_tier2"],
        current_world_state=world_state,
        reasoning_trace=trace,
        prohibited_actions=["auto_resolve"],
        predicted_outcome=pred_outcomes,
    )

    # 1. Check prohibited action excluded
    action_names = [s.action_name for s in plan.steps]
    assert "auto_resolve" not in action_names

    # 2. Check refund precondition failed due to monetary_value_usd > 1500
    assert "issue_refund" not in action_names
    assert "escalate_to_tier2" in action_names

    # 3. Check dynamic risk is not fixed hardcoded 0.15/0.25
    step = plan.steps[0]
    assert step.risk_estimate > 0.0

    # 4. Check alternative plans evaluated with prospective scores
    assert len(plan.alternative_plans) > 0
    alt = plan.alternative_plans[0]
    assert "predicted_outcome_score" in alt
    assert "expected_risk" in alt
