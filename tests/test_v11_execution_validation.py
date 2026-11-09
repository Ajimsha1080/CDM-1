"""CDM-1 V11 Execution Outcome Verification, Step-ID, and Target-Metric Test Suite.
Validates:
1. Rejection of automatic conversion of planned postconditions into observed postconditions.
2. Negative test: successful action execution with unsuccessful domain outcome fails verified goal completion.
3. Plan lifecycle separation: proposed, approved, executed, verified_complete, verified_failed.
4. Validation and rejection of duplicate, unknown, missing, and out-of-order execution step IDs.
5. Rejection of incomplete execution records.
6. Validation of numeric target conditions (upper bounds, lower bounds, tolerances, dict operators).
7. Validation of non-numeric target conditions (categorical strings, booleans).
"""

from typing import Tuple
import pytest
from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace, Plan, ExecutedActionStep
from cdm.planning.planning_engine import PlanningEngine

Tuple_Plan_Goal_World = Tuple[PlanningEngine, Plan, GoalNode, WorldState]


def _create_test_plan(num_steps: int = 2) -> Tuple_Plan_Goal_World:
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_test_v11",
        description="Stabilize patient and administer meds",
        priority=1,
        required_postconditions=["vital_signs_stabilized"],
        target_state_metrics={"vital_instability": 0.0, "patient_status": "stable"},
    )
    world = WorldState(
        environment_metrics={"vital_instability": 0.8, "patient_status": "unstable"}
    )
    reasoning = ReasoningTrace()

    actions = ["stabilize_vitals", "administer_meds"][:num_steps]
    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=actions,
        current_world_state=world,
        reasoning_trace=reasoning,
    )
    return pe, plan, goal, world


# =========================================================================
# 1. LIFECYCLE SEPARATION (PROPOSED -> APPROVED -> EXECUTED -> VERIFIED)
# =========================================================================

def test_plan_lifecycle_approval_and_execution_transition():
    """Validates lifecycle stages: proposed -> approved -> in_progress -> executed -> completed."""
    pe, plan, goal, world = _create_test_plan(num_steps=2)

    # Initial proposed stage
    assert plan.plan_stage == "proposed"
    assert plan.is_verified_goal_completion is False

    # Approval stage
    plan = pe.approve_plan(plan)
    assert plan.plan_stage == "approved"

    # Step 1 execution -> in_progress
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"observed_postconditions": ["vital_signs_stabilized"]},
        observed_state=WorldState(environment_metrics={"vital_instability": 0.0}),
        execution_status="success",
    )
    assert plan.plan_stage == "in_progress"

    # Step 2 execution -> executed
    pe.record_step_execution(
        plan=plan,
        step_number=2,
        observed_outcome={"observed_postconditions": ["pharmacotherapy_administered"]},
        observed_state=WorldState(environment_metrics={"vital_instability": 0.0, "patient_status": "stable"}),
        execution_status="success",
    )
    assert plan.plan_stage == "executed"


def test_cannot_approve_already_approved_plan():
    """Approving a non-proposed plan raises ValueError."""
    pe, plan, goal, world = _create_test_plan(num_steps=1)
    pe.approve_plan(plan)
    with pytest.raises(ValueError, match="Cannot approve plan in 'approved' stage"):
        pe.approve_plan(plan)


# =========================================================================
# 2. REMOVAL OF AUTOMATIC POSTCONDITION CONVERSION & NEGATIVE OUTCOME TEST
# =========================================================================

def test_successful_execution_with_unsuccessful_outcome_fails_verification():
    """CRITICAL REQUIREMENT:
    Action executes with execution_status='success' (e.g. refund HTTP 200),
    but real-world outcome fails (e.g. bank rejected transaction, postcondition absent).
    Verification MUST fail, and goal completion MUST remain False!
    """
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_dispute",
        description="Resolve billing dispute",
        priority=1,
        required_postconditions=["billing_dispute_resolved"],
        target_state_metrics={"dispute_active": 0.0},
    )
    world = WorldState(environment_metrics={"dispute_active": 1.0})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["refund_customer"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # Plan proposed postconditions include 'billing_dispute_resolved'
    assert "billing_dispute_resolved" in plan.steps[0].postconditions

    # Action executes successfully (status="success"), BUT observed outcome failed!
    unsuccessful_observed_outcome = {
        "status": "success",
        "api_response_code": 200,
        "refund_gateway_status": "DECLINED_BY_BANK",
        "observed_postconditions": [],  # NO 'billing_dispute_resolved' achieved!
    }

    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome=unsuccessful_observed_outcome,
        observed_state=WorldState(environment_metrics={"dispute_active": 1.0}),
        execution_status="success",
    )

    # Verify completion against observed world
    is_completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=WorldState(environment_metrics={"dispute_active": 1.0}),
    )

    assert is_completed is False
    assert plan.is_verified_goal_completion is False
    assert plan.plan_stage == "failed"
    assert "billing_dispute_resolved" in details["missing_required_postconditions"]


# =========================================================================
# 3. STEP-ID & DEPENDENCY VALIDATION (POSITIVE & NEGATIVE TESTS)
# =========================================================================

def test_reject_duplicate_step_id_in_record_execution():
    """Recording the same step_number twice must raise ValueError."""
    pe, plan, goal, world = _create_test_plan(num_steps=2)

    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"vital_signs_stabilized": True},
        execution_status="success",
    )

    # Recording step 1 again raises ValueError
    with pytest.raises(ValueError, match="Duplicate execution step ID"):
        pe.record_step_execution(
            plan=plan,
            step_number=1,
            observed_outcome={"vital_signs_stabilized": True},
            execution_status="success",
        )


def test_reject_unknown_step_id_in_record_execution():
    """Recording a step number that is not part of the plan must raise ValueError."""
    pe, plan, goal, world = _create_test_plan(num_steps=2)

    with pytest.raises(ValueError, match="Unknown execution step ID"):
        pe.record_step_execution(
            plan=plan,
            step_number=99,
            observed_outcome={"data": 1},
            execution_status="success",
        )


def test_reject_out_of_order_execution():
    """Attempting to execute step 2 before step 1 raises ValueError."""
    pe, plan, goal, world = _create_test_plan(num_steps=2)

    # Execute step 2 first
    with pytest.raises(ValueError, match="Execution order violation"):
        pe.record_step_execution(
            plan=plan,
            step_number=2,
            observed_outcome={"data": 1},
            execution_status="success",
        )


def test_reject_incomplete_execution_record():
    """Non-dict outcome or invalid execution_status raises ValueError."""
    pe, plan, goal, world = _create_test_plan(num_steps=1)

    # Outcome not a dict
    with pytest.raises(ValueError, match="Incomplete execution record: observed_outcome must be a valid dictionary"):
        pe.record_step_execution(
            plan=plan,
            step_number=1,
            observed_outcome="invalid_string",
            execution_status="success",
        )

    # Invalid execution_status
    with pytest.raises(ValueError, match="Incomplete execution record: invalid execution_status"):
        pe.record_step_execution(
            plan=plan,
            step_number=1,
            observed_outcome={"data": 1},
            execution_status="invalid_status",
        )


def test_verify_rejects_missing_step_ids():
    """Verifying a plan where planned steps were omitted fails verification."""
    pe, plan, goal, world = _create_test_plan(num_steps=2)

    # Only step 1 executed
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"observed_postconditions": ["vital_signs_stabilized"]},
        execution_status="success",
    )

    is_completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=world,
    )
    assert is_completed is False
    assert details["error_code"] == "MISSING_STEP_ID"
    assert plan.is_verified_goal_completion is False


# =========================================================================
# 4. TARGET-METRIC VALIDATION (NUMERIC & NON-NUMERIC)
# =========================================================================

def test_numeric_target_metric_bounds_validation():
    """Validates upper bounds (e.g. exposure <= 0.1), lower bounds, and dict operators."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_metrics",
        description="Audit metrics",
        priority=1,
        required_postconditions=["action_executed_isolate_cluster"],
        target_state_metrics={
            "system_exposure": 0.10,          # Upper bound: must be <= 0.10
            "cluster_health": {">=": 0.95},   # Dict operator: must be >= 0.95
        },
    )
    world = WorldState()
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["isolate_cluster"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # 1. When observed metrics satisfy targets
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={
            "observed_postconditions": ["action_executed_isolate_cluster"],
            "system_exposure": 0.08,
            "cluster_health": 0.99,
        },
        observed_state=WorldState(environment_metrics={"system_exposure": 0.08, "cluster_health": 0.99}),
        execution_status="success",
    )

    is_completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=WorldState(environment_metrics={"system_exposure": 0.08, "cluster_health": 0.99}),
    )
    assert is_completed is True
    assert details["metric_evaluations"]["system_exposure"]["satisfied"] is True
    assert details["metric_evaluations"]["cluster_health"]["satisfied"] is True


def test_numeric_metric_exceeded_fails_verification():
    """When an observed metric exceeds the target limit, verification must fail."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_sec_limit",
        description="Reduce exposure limit",
        priority=1,
        target_state_metrics={"system_exposure": 0.10},
    )
    world = WorldState()
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # Observed exposure is 0.50 (> 0.10)
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"healthcheck_ok": True},
        observed_state=WorldState(environment_metrics={"system_exposure": 0.50}),
        execution_status="success",
    )

    is_completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=WorldState(environment_metrics={"system_exposure": 0.50}),
    )

    assert is_completed is False
    assert plan.is_verified_goal_completion is False
    assert any("system_exposure" in err for err in details["failed_metrics"])


def test_non_numeric_target_conditions():
    """Validates string / categorical and boolean conditions."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_non_numeric",
        description="Verify categorical status",
        priority=1,
        target_state_metrics={
            "patient_status": "stable",
            "triage_completed": True,
        },
    )
    world = WorldState()
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # Case A: Non-numeric condition matches
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"data": 1},
        observed_state=WorldState(environment_metrics={"patient_status": "stable", "triage_completed": True}),
        execution_status="success",
    )

    is_ok, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=WorldState(environment_metrics={"patient_status": "stable", "triage_completed": True}),
    )
    assert is_ok is True
    assert details["metric_evaluations"]["patient_status"]["satisfied"] is True
    assert details["metric_evaluations"]["triage_completed"]["satisfied"] is True

    # Case B: Non-numeric condition mismatch fails
    plan_mismatch = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )
    pe.record_step_execution(
        plan=plan_mismatch,
        step_number=1,
        observed_outcome={"data": 1},
        observed_state=WorldState(environment_metrics={"patient_status": "critical", "triage_completed": True}),
        execution_status="success",
    )

    is_ok, details = pe.verify_plan_execution_completion(
        plan=plan_mismatch,
        target_goal=goal,
        final_observed_world_state=WorldState(environment_metrics={"patient_status": "critical", "triage_completed": True}),
    )
    assert is_ok is False
    assert plan_mismatch.is_verified_goal_completion is False
    assert any("patient_status" in err for err in details["failed_metrics"])
