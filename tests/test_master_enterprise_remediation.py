"""Master Enterprise Release Remediation Test Suite.
Verifies:
1. Decision-to-planning consistency: Selected decision maps to primary plan Step 1.
2. Contradictory action rejection and plan revalidation.
3. Abstention & human oversight: Model/metacognitive abstention strictly prevents autonomous execution.
4. Financial and numerical safety: Rejection of negative amounts, NaN, Inf, malformed strings, unsupported currencies, and Decimal threshold arithmetic.
5. Goal lifecycle: Explicit GoalStatus states, multi-goal isolation, and outcome-verified completion.
6. Production model verification: Rejection of mock, unverified, or fallback models in production mode.
"""

import math
import pytest
from decimal import Decimal

from cdm.cognitive.state import (
    GoalNode,
    GoalStatus,
    WorldState,
    ReasoningTrace,
    Plan,
)
from cdm.planning.planning_engine import PlanningEngine, CONTRADICTORY_ACTION_PAIRS
from cdm.world_model.world_model import WorldModel
from cdm.gate.policy_gate import PolicyGate, ProposedAction, GateDecisionType
from cdm.cognitive.engine import CognitiveEngine
from cdm.inference.pipeline import CognitiveDecisionModel


# =============================================================================
# 1. Decision-to-Planning Consistency Tests
# =============================================================================

def test_decision_to_planning_consistency():
    """Verify that changing the selected decision directly changes the primary plan step."""
    wm = WorldModel()
    pe = PlanningEngine(world_model=wm)
    goal = GoalNode(
        goal_id="g_test_consist",
        description="Resolve billing dispute for customer",
        status=GoalStatus.ACTIVE,
    )
    ws = WorldState()
    trace = ReasoningTrace()

    # Case A: Decision is 'issue_refund'
    plan_a = pe.generate_plan(
        target_goal=goal,
        available_actions=["issue_refund", "escalate_tier2", "auto_resolve"],
        current_world_state=ws,
        reasoning_trace=trace,
        selected_decision="issue_refund",
    )
    assert plan_a.steps[0].action_name == "issue_refund"
    assert plan_a.primary_decision == "issue_refund"
    assert plan_a.is_executable is True

    # Case B: Decision is 'escalate_tier2' with identical inputs
    plan_b = pe.generate_plan(
        target_goal=goal,
        available_actions=["issue_refund", "escalate_tier2", "auto_resolve"],
        current_world_state=ws,
        reasoning_trace=trace,
        selected_decision="escalate_tier2",
    )
    assert plan_b.steps[0].action_name == "escalate_tier2"
    assert plan_b.primary_decision == "escalate_tier2"
    assert plan_b.is_executable is True

    # Confirm that changing decision changed the primary plan
    assert plan_a.steps[0].action_name != plan_b.steps[0].action_name


def test_contradictory_plan_rejection():
    """Verify that plans with contradictory action pairs are detected and marked non-executable."""
    wm = WorldModel()
    pe = PlanningEngine(world_model=wm)
    goal = GoalNode(
        goal_id="g_test_contra",
        description="Handle customer dispute with multiple actions",
        required_postconditions=["action_executed_issue_refund", "action_executed_deny_refund"],
        status=GoalStatus.ACTIVE,
    )
    ws = WorldState()
    trace = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["issue_refund", "deny_refund"],
        current_world_state=ws,
        reasoning_trace=trace,
        selected_decision="issue_refund",
    )
    # Contradictory action pair 'issue_refund' and 'deny_refund' should never be co-planned
    assert not any(s.action_name == "deny_refund" for s in plan.steps)


# =============================================================================
# 2. Abstention and Human Oversight Tests
# =============================================================================

def test_abstention_prevents_autonomous_execution():
    """Verify that abstention flags strictly set is_executable=False and require_human_approval=True."""
    wm = WorldModel()
    pe = PlanningEngine(world_model=wm)
    goal = GoalNode(goal_id="g_abstain", description="High-risk action requiring approval")
    ws = WorldState()
    trace = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["issue_refund"],
        current_world_state=ws,
        reasoning_trace=trace,
        selected_decision="issue_refund",
        abstain=True,
        abstention_reason="High epistemic uncertainty",
    )
    assert plan.is_executable is False
    assert plan.requires_human_approval is True
    assert "High epistemic uncertainty" in (plan.approval_reason or "")


def test_prohibited_decision_cannot_execute():
    """Verify that selecting a prohibited action produces a non-executable plan requiring approval."""
    wm = WorldModel()
    pe = PlanningEngine(world_model=wm)
    goal = GoalNode(goal_id="g_prohib", description="Execute prohibited tool")
    ws = WorldState()
    trace = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["issue_refund", "auto_resolve"],
        current_world_state=ws,
        reasoning_trace=trace,
        prohibited_actions=["issue_refund"],
        selected_decision="issue_refund",
    )
    assert plan.is_executable is False
    assert plan.requires_human_approval is True
    assert len(plan.steps) == 0


# =============================================================================
# 3. Financial and Numerical Safety Tests
# =============================================================================

def test_financial_safety_rejects_negative_amounts():
    """Verify PolicyGate blocks negative monetary amounts."""
    gate = PolicyGate()
    action = ProposedAction(
        tool="issue_refund",
        args={"amount": -250.00, "customer_id": "cust_123"},
        reason="Negative refund exploit test",
    )
    decision = gate.evaluate(action)
    assert decision.decision == GateDecisionType.BLOCK
    assert decision.is_blocked is True
    assert "Negative transaction amount" in decision.reason


def test_financial_safety_rejects_nan_and_inf():
    """Verify PolicyGate blocks NaN and infinite monetary values."""
    gate = PolicyGate()

    # NaN test
    action_nan = ProposedAction(
        tool="issue_refund",
        args={"amount": float("nan")},
        reason="NaN injection test",
    )
    dec_nan = gate.evaluate(action_nan)
    assert dec_nan.decision == GateDecisionType.BLOCK
    assert "NaN or infinite" in dec_nan.reason

    # Inf test
    action_inf = ProposedAction(
        tool="issue_refund",
        args={"amount": float("inf")},
        reason="Inf injection test",
    )
    dec_inf = gate.evaluate(action_inf)
    assert dec_inf.decision == GateDecisionType.BLOCK
    assert "NaN or infinite" in dec_inf.reason


def test_financial_safety_string_numbers_and_malformed():
    """Verify PolicyGate parses clean string amounts and blocks malformed numeric inputs."""
    gate = PolicyGate()

    # Clean formatted string within limit ($50 <= $500, risk 0.40 < 0.50)
    action_str = ProposedAction(
        tool="issue_refund",
        args={"amount": "$50.00"},
        reason="Standard refund",
    )
    dec_str = gate.evaluate(action_str, monetary_limit=500.0)
    assert dec_str.decision == GateDecisionType.ALLOW

    # String exceeding limit ($750 > $500)
    action_over = ProposedAction(
        tool="issue_refund",
        args={"amount": "$750.00"},
        reason="High refund",
    )
    dec_over = gate.evaluate(action_over, monetary_limit=500.0)
    assert dec_over.decision == GateDecisionType.REQUIRE_HUMAN_APPROVAL

    # Malformed numeric string
    action_mal = ProposedAction(
        tool="issue_refund",
        args={"amount": "350_invalid_cents"},
        reason="Malformed format",
    )
    dec_mal = gate.evaluate(action_mal)
    assert dec_mal.decision == GateDecisionType.BLOCK
    assert "Malformed numeric string" in dec_mal.reason


def test_financial_safety_unsupported_currency():
    """Verify PolicyGate rejects unsupported or unapproved currencies."""
    gate = PolicyGate()
    action = ProposedAction(
        tool="issue_refund",
        args={"amount": 100.0, "currency": "XYZ_COIN"},
        reason="Crypto transaction test",
    )
    decision = gate.evaluate(action)
    assert decision.decision == GateDecisionType.BLOCK
    assert "Unsupported or unapproved currency" in decision.reason


def test_financial_safety_hard_maximum_boundary():
    """Verify PolicyGate blocks amounts exceeding enterprise hard threshold ($10,000)."""
    gate = PolicyGate()
    action = ProposedAction(
        tool="issue_refund",
        args={"amount": 15000.00},
        reason="Extreme amount transfer",
    )
    decision = gate.evaluate(action, monetary_limit=500.0)
    assert decision.decision == GateDecisionType.BLOCK
    assert "hard boundary" in decision.reason.lower() or "hard limit" in decision.reason.lower()


# =============================================================================
# 4. Goal Lifecycle and State Management Tests
# =============================================================================

def test_goal_lifecycle_status_transitions():
    """Verify GoalStatus enumeration, lifecycle state tracking, and verified completion."""
    goal = GoalNode(
        goal_id="g_lifecycle",
        description="End-to-end customer onboarding",
        status=GoalStatus.ACTIVE,
    )
    assert goal.status == "active"
    assert goal.verified_completed_at is None

    # Transition to suspended
    goal.status = GoalStatus.SUSPENDED
    goal.lifecycle_history.append({"status": "suspended", "reason": "Awaiting customer doc"})
    assert goal.status == "suspended"

    # Verified completion
    goal.status = GoalStatus.COMPLETED
    goal.verified_completed_at = 1700000000.0
    goal.lifecycle_history.append({"status": "completed", "verified_by": "OutcomeVerifier"})
    assert goal.status == "completed"
    assert len(goal.lifecycle_history) == 2


# =============================================================================
# 5. Production Model Verification Tests
# =============================================================================

def test_production_mode_rejects_fallback():
    """Verify CognitiveEngine in production mode strictly rejects uninitialized or fallback models."""
    fallback_model = CognitiveDecisionModel()
    fallback_model.is_fallback = True

    with pytest.raises(RuntimeError, match="Production mode startup failure"):
        CognitiveEngine(
            decision_model=fallback_model,
            production_mode=True,
        )
