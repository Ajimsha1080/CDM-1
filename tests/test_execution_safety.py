"""Tests for Plan Dispatcher and Execution Safety Gates (Gate 5).
Validates:
1. Dispatcher executes ONLY approved plans.
2. Rejection of proposed, abstained, prohibited, and expired plans.
3. Explicit authorization requirement for sensitive actions.
4. Prevention of planning and approval bypass attempts.
"""

import time
import pytest

from cdm.cognitive.state import Plan, PlanStep, WorldState
from cdm.planning.planning_engine import PlanningEngine
from cdm.execution.dispatcher import PlanDispatcher, PlanExecutionError


def create_sample_plan(stage="proposed", actions=None):
    if actions is None:
        actions = ["auto_resolve"]
    steps = [
        PlanStep(
            step_number=i + 1,
            action_name=act,
            parameters={},
            expected_outcome={},
        )
        for i, act in enumerate(actions)
    ]
    return Plan(
        target_goal_id="goal_101",
        plan_stage=stage,
        status=stage,
        steps=steps,
    )


def test_dispatcher_rejects_proposed_plan():
    """Verify dispatcher rejects plans still in proposed stage."""
    dispatcher = PlanDispatcher()
    plan = create_sample_plan(stage="proposed")

    with pytest.raises(PlanExecutionError, match="Attempted to dispatch unapproved plan in 'proposed' stage"):
        dispatcher.dispatch_plan(plan)


def test_dispatcher_rejects_abstained_plan():
    """Verify dispatcher rejects abstained plans."""
    dispatcher = PlanDispatcher()
    plan = create_sample_plan(stage="abstained")

    with pytest.raises(PlanExecutionError, match="Attempted to dispatch plan marked 'abstained'"):
        dispatcher.dispatch_plan(plan)


def test_dispatcher_rejects_expired_plan():
    """Verify dispatcher rejects plans whose expiration timestamp has passed."""
    dispatcher = PlanDispatcher()
    plan = create_sample_plan(stage="approved")
    plan.expires_at = time.time() - 100.0  # Expired in past

    with pytest.raises(PlanExecutionError, match="expired"):
        dispatcher.dispatch_plan(plan)
    assert plan.plan_stage == "expired"


def test_dispatcher_rejects_prohibited_actions():
    """Verify dispatcher rejects plans containing prohibited actions."""
    dispatcher = PlanDispatcher()
    plan = create_sample_plan(stage="approved", actions=["auto_resolve", "delete_database"])

    with pytest.raises(PlanExecutionError, match="explicitly prohibited"):
        dispatcher.dispatch_plan(plan, prohibited_actions=["delete_database"])
    assert plan.plan_stage == "prohibited"


def test_dispatcher_requires_authorization_for_sensitive_actions():
    """Verify sensitive actions (e.g. refunds) require authorization token or authorized role."""
    dispatcher = PlanDispatcher()
    plan = create_sample_plan(stage="approved", actions=["issue_refund"])

    # Without token: rejected
    with pytest.raises(PlanExecutionError, match="requires explicit authorization"):
        dispatcher.dispatch_plan(plan, executor_role="standard_agent")

    # With valid authorization token: accepted
    dispatched = dispatcher.dispatch_plan(plan, authorization_token="AUTH-REFUND-992")
    assert dispatched.plan_stage == "dispatched"


def test_planning_engine_approval_and_dispatch_flow():
    """Verify end-to-end planning engine lifecycle transitions from proposed -> approved -> dispatched."""
    pe = PlanningEngine(world_model=WorldState())
    plan = create_sample_plan(stage="proposed", actions=["auto_resolve"])

    # Step recording directly on proposed plan is blocked
    with pytest.raises(ValueError, match="Execution Safety Violation.*Only approved/dispatched plans may execute"):
        pe.record_step_execution(plan, step_number=1, observed_outcome={"status": "done"})

    # Approve plan
    approved_plan = pe.approve_plan(plan, approver_id="safety_auditor")
    assert approved_plan.plan_stage == "approved"
    assert approved_plan.approved_by == "safety_auditor"

    # Dispatch plan
    dispatched_plan = pe.dispatch_plan(approved_plan)
    assert dispatched_plan.plan_stage == "dispatched"

    # Record execution on dispatched plan succeeds
    pe.record_step_execution(dispatched_plan, step_number=1, observed_outcome={"status": "done"})
    assert len(dispatched_plan.executed_steps) == 1
    assert dispatched_plan.plan_stage == "executed"
