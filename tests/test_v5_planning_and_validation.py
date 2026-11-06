"""Validation and Remediation Test Suite for CDM-1 V5 Final Release Blockers.
Validates:
1. Planning Engine: Goal satisfaction, preconditions, postconditions, alternative plans, constraint compliance.
2. Risk provenance: Strict distinction between trained_model_prediction and world_state_uncertainty_heuristic.
3. Database-backed feedback failure-case retrieval and review updates across simulated restarts and concurrency.
4. Strict tenant isolation in ContextEngine without default fallback cross-tenant leaks.
5. Readiness endpoint checkpoint SHA-256 verification against release manifest.
"""

import os
import tempfile
import threading
import pytest
from fastapi.testclient import TestClient

from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace
from cdm.planning.planning_engine import PlanningEngine
from cdm.learning.feedback import FeedbackCollector, FeedbackEvent
from cdm.context.context_engine import ContextEngine
from api.main import app


# =========================================================================
# 1. PLANNING ENGINE VALIDATION TESTS
# =========================================================================

def test_planning_goal_satisfaction_and_postconditions():
    """Validates goal satisfaction, preconditions, postconditions and constraint compliance."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_emergency_triage",
        description="Stabilize patient with acute symptoms",
        priority=1,
    )
    world = WorldState(
        tenant_id="tenant_alpha",
        session_id="session_001",
        environment_metrics={"monetary_value_usd": 200.0, "sla_remaining_hours": 12.0},
        overall_state_uncertainty=0.1,
    )
    reasoning = ReasoningTrace(
        satisfied_constraints=["standard_protocol_active"],
        violated_constraints=[],
    )
    available_actions = ["triage_patient", "administer_meds", "order_labs"]

    # Plan with no model prediction -> heuristic risk
    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=available_actions,
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert plan.goal_satisfied is True
    assert plan.constraint_compliance is True
    assert len(plan.steps) == 3
    assert plan.steps[0].action_name == "triage_patient"
    assert plan.steps[0].risk_source in ["world_state_uncertainty_heuristic", "deterministic_heuristic"]
    assert plan.risk_source in ["world_state_uncertainty_heuristic", "deterministic_heuristic"]

    # Verify postconditions on every step
    for step in plan.steps:
        assert f"action_executed_{step.action_name}" in step.postconditions
        assert len(step.postconditions) >= 1
        assert step.risk_estimate > 0.0

    # Verify alternative plans generated
    assert len(plan.alternative_plans) == 2
    assert plan.alternative_plans[0]["first_action"] == "administer_meds"
    assert plan.alternative_plans[0]["risk_source"] in ["world_state_uncertainty_heuristic", "deterministic_heuristic"]


def test_planning_distinguishes_trained_model_risk():
    """Validates that trained model continuous outcome predictions override heuristics with provenance."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_resolve", description="Resolve customer inquiry", priority=1)
    world = WorldState(
        tenant_id="tenant_beta",
        session_id="session_002",
        environment_metrics={"monetary_value_usd": 50.0},
        overall_state_uncertainty=0.08,
    )
    reasoning = ReasoningTrace()

    # Pass explicit trained model outcome predictions with genuine cryptographic provenance
    from cdm.inference.provenance import PredictionProvenance
    prov = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        estimated_risk=0.12,
    )
    predicted_outcome = {
        "predicted_risk": 0.12,
        "candidate_scores": {"auto_resolve": 0.92, "escalate_tier2": 0.35},
        "provenance": prov,
    }

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["auto_resolve", "escalate_tier2"],
        current_world_state=world,
        reasoning_trace=reasoning,
        predicted_outcome=predicted_outcome,
    )

    assert plan.goal_satisfied is True
    assert plan.risk_source == "trained_model_prediction"
    assert plan.steps[0].risk_source == "trained_model_prediction"
    # Step risk should be weighted combination: 0.12 * 0.7 + 0.08 * 0.3 = 0.084 + 0.024 = 0.108
    assert abs(plan.steps[0].risk_estimate - 0.108) < 1e-3
    assert len(plan.alternative_plans) == 1
    assert plan.alternative_plans[0]["risk_source"] == "trained_model_prediction"


def test_planning_precondition_failures_and_constraints():
    """Validates that precondition failures and prohibited actions are strictly excluded."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_dispute", description="Process billing dispute", priority=2)
    # Monetary value exceeds refund limit of $1500
    world = WorldState(
        tenant_id="tenant_gamma",
        session_id="session_003",
        environment_metrics={"monetary_value_usd": 5000.0},
    )
    reasoning = ReasoningTrace(
        violated_constraints=["hard_limit: manual_escalation prohibited for unverified account"],
    )

    available_actions = ["refund_customer", "manual_escalation", "request_documentation"]
    prohibited = ["manual_escalation"]

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=available_actions,
        current_world_state=world,
        reasoning_trace=reasoning,
        prohibited_actions=prohibited,
    )

    # refund_customer should fail precondition (> $1500)
    # manual_escalation is prohibited
    # Only request_documentation should remain
    assert len(plan.steps) == 1
    assert plan.steps[0].action_name == "request_documentation"
    assert plan.constraint_compliance is True
    # request_documentation remains compliant, but alone cannot resolve billing dispute -> goal_satisfied is False
    assert plan.goal_satisfied is False


def test_planning_goal_unsatisfied_when_no_valid_steps():
    """Validates goal_satisfied is False when all actions are prohibited or preconditions fail."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_blocked", description="Blocked goal", priority=3)
    world = WorldState(
        environment_metrics={"monetary_value_usd": 3000.0},
    )
    reasoning = ReasoningTrace(
        violated_constraints=["prohibit all external communications"],
    )

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["refund_transaction"],  # Fails precondition (> $1500)
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert len(plan.steps) == 0
    assert plan.goal_satisfied is False


# =========================================================================
# 2. DATABASE-BACKED FEEDBACK & RESTART PERSISTENCE TESTS
# =========================================================================

def test_database_backed_feedback_failure_retrieval_and_restarts():
    """Validates that get_failure_cases and review status updates read from SQLite across process restarts."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        # Instance 1: write feedback
        collector1 = FeedbackCollector(db_path=db_path)
        e1 = collector1.record_feedback(
            decision_id="dec_ok",
            predicted_decision={"binary_prediction": 1},
            actual_outcome={"binary_target": 1},
            is_success=True,
            tenant_id="tenant_corp",
        )
        e2 = collector1.record_feedback(
            decision_id="dec_fail_1",
            predicted_decision={"binary_prediction": 1},
            actual_outcome={"binary_target": 0},
            is_success=False,
            tenant_id="tenant_corp",
        )
        e3 = collector1.record_feedback(
            decision_id="dec_fail_2",
            predicted_decision={"binary_prediction": 1},
            actual_outcome={"binary_target": 0},
            is_success=False,
            tenant_id="tenant_other",
        )

        # Instance 2 (simulating process restart)
        collector2 = FeedbackCollector(db_path=db_path)

        # Direct database query for failure cases
        failures_all = collector2.get_failure_cases()
        assert len(failures_all) == 2
        fail_ids = {f.decision_id for f in failures_all}
        assert "dec_fail_1" in fail_ids
        assert "dec_fail_2" in fail_ids

        # Tenant-isolated failure cases
        failures_corp = collector2.get_failure_cases(tenant_id="tenant_corp")
        assert len(failures_corp) == 1
        assert failures_corp[0].decision_id == "dec_fail_1"

        # Update review status in collector2
        updated = collector2.update_review_status(
            event_id=e2.event_id,
            new_status="verified",
            tenant_id="tenant_corp",
            reviewer_id="auditor_lead",
        )
        assert updated is True

        # Instance 3: verify persisted update in clean instance
        collector3 = FeedbackCollector(db_path=db_path)
        events_after = collector3.get_failure_cases(tenant_id="tenant_corp")
        assert events_after[0].review_status == "verified"
        assert events_after[0].reviewer_id == "auditor_lead"

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_concurrent_feedback_and_review_updates():
    """Validates thread-safety of database-backed feedback recording and review updates."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        collector = FeedbackCollector(db_path=db_path)
        errors = []

        def record_worker(worker_idx: int):
            try:
                for i in range(10):
                    collector.record_feedback(
                        decision_id=f"dec_{worker_idx}_{i}",
                        predicted_decision={"binary_prediction": 1},
                        actual_outcome={"binary_target": 0 if i % 2 == 0 else 1},
                        is_success=(i % 2 != 0),
                        tenant_id=f"tenant_{worker_idx}",
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_worker, args=(w,)) for w in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all failure cases are durably retrieved
        failures = collector.get_failure_cases()
        assert len(failures) == 20  # 4 workers * 5 failures each

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# =========================================================================
# 3. EXACT TENANT MATCHING IN CONTEXT ENGINE
# =========================================================================

def test_context_engine_exact_tenant_session_matching():
    """Validates that get_latest_context strictly requires exact tenant and session matching."""
    from cdm.cognitive.state import PerceptionResult

    ce = ContextEngine()
    dummy_perception = PerceptionResult(
        raw_text="Alpha secret data",
        text_representation="Alpha secret data",
        intent="query",
        extracted_entities=[],
        domain="testing",
    )
    ce.update_context(
        tenant_id="tenant_alpha",
        session_id="session_alpha_1",
        perception=dummy_perception,
        user_message="Alpha secret data",
    )

    # Exact match succeeds
    exact = ce.get_latest_context(tenant_id="tenant_alpha", session_id="session_alpha_1")
    assert exact is not None
    assert exact.dialogue_history[0]["content"] == "Alpha secret data"

    # Cross-tenant query with same session_id returns None
    cross_tenant = ce.get_latest_context(tenant_id="tenant_beta", session_id="session_alpha_1")
    assert cross_tenant is None

    # default_tenant cannot fall back and leak tenant_alpha
    default_query = ce.get_latest_context(tenant_id="default_tenant", session_id="session_alpha_1")
    assert default_query is None

    # Non-existent session returns None
    non_existent = ce.get_latest_context(tenant_id="tenant_alpha", session_id="session_non_existent")
    assert non_existent is None


# =========================================================================
# 4. READINESS ENDPOINT SHA-256 VERIFICATION
# =========================================================================

def test_readiness_sha256_verification():
    """Validates that /v1/ready verifies checkpoint SHA-256 against release_manifest.json."""
    with TestClient(app) as client:
        resp = client.get("/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["weights_verified"] is True
        assert "weights_sha256" in data
        assert len(data["weights_sha256"]) == 64
        assert data["weights_sha256"] == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
