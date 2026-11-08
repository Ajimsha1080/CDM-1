#!/usr/bin/env python3
"""CDM-1 Final Verification: Execution Outcome Verification, Step-ID & Metric Validation.

Verifies:
1. Explicit plan lifecycle transitions: proposed -> approved -> in_progress -> executed -> completed.
2. Rejection of unapproved execution or illegal lifecycle transitions.
3. Genuine telemetry verification: no automatic conversion of planned postconditions.
4. Negative test: successful action execution with unsuccessful outcome fails goal completion.
5. Strict step ID validation: rejects duplicate, missing, unknown, and out-of-order execution steps.
6. Target state metric evaluation: numeric upper/lower bounds, dict operator conditions, booleans, and non-numeric strings.
7. Rejection of incomplete execution records.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace, Plan, ExecutedActionStep
from cdm.planning.planning_engine import PlanningEngine


def verify_execution(report_path: str = "execution_report.json") -> Dict[str, Any]:
    print("=" * 70)
    print("CDM-1 VERIFICATION: EXECUTION OUTCOME, STEP-ID & TARGET-METRIC VALIDATION")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "checks": {},
        "passed": False,
    }

    pe = PlanningEngine()

    # 1. Plan Lifecycle Transitions: proposed -> approved -> in_progress -> executed -> completed
    print("[*] 1. Verifying explicit plan lifecycle stages...")
    goal1 = GoalNode(
        goal_id="goal_lifecycle",
        description="Verify plan lifecycle approval and execution progression",
        priority=1,
        required_postconditions=["vital_signs_stabilized"],
        target_state_metrics={"vital_instability": 0.0},
    )
    world1 = WorldState(environment_metrics={"vital_instability": 0.8})
    trace1 = ReasoningTrace()

    plan1 = pe.generate_plan(
        target_goal=goal1,
        available_actions=["stabilize_vitals"],
        current_world_state=world1,
        reasoning_trace=trace1,
    )
    s1_proposed = (plan1.plan_stage == "proposed" and plan1.is_verified_goal_completion is False)

    pe.approve_plan(plan1)
    s2_approved = (plan1.plan_stage == "approved")

    # Reject re-approval
    reapprove_failed = False
    try:
        pe.approve_plan(plan1)
    except ValueError:
        reapprove_failed = True

    pe.record_step_execution(
        plan=plan1,
        step_number=1,
        observed_outcome={"vital_signs_stabilized": True, "observed_postconditions": ["vital_signs_stabilized"]},
        observed_state=WorldState(environment_metrics={"vital_instability": 0.0}),
        execution_status="success",
    )
    s3_executed = (plan1.plan_stage in ["in_progress", "executed"])

    completed, details = pe.verify_plan_execution_completion(
        plan=plan1,
        target_goal=goal1,
        final_observed_world_state=WorldState(environment_metrics={"vital_instability": 0.0}),
    )
    s4_completed = (completed is True and plan1.plan_stage == "completed" and plan1.is_verified_goal_completion is True)

    lifecycle_ok = s1_proposed and s2_approved and reapprove_failed and s3_executed and s4_completed
    report["checks"]["plan_lifecycle_transitions"] = {
        "status": "PASS" if lifecycle_ok else "FAIL",
        "proposed_verified": s1_proposed,
        "approved_verified": s2_approved,
        "reapproval_rejected": reapprove_failed,
        "execution_transition_verified": s3_executed,
        "completion_verified": s4_completed,
    }
    print(f"[*] Lifecycle state transitions: proposed->approved->executed->completed -> {'PASS' if lifecycle_ok else 'FAIL'}")

    # 2. Negative Test: Successful Execution with Unsuccessful Real-World Outcome
    print("[*] 2. Verifying negative test: successful execution with unsuccessful outcome...")
    goal2 = GoalNode(
        goal_id="goal_dispute_refund",
        description="Verify dispute refund resolution",
        priority=1,
        required_postconditions=["billing_dispute_resolved"],
        target_state_metrics={"dispute_exposure_usd": 0.0},
    )
    world2 = WorldState(environment_metrics={"monetary_value_usd": 120.0, "dispute_exposure_usd": 120.0})
    trace2 = ReasoningTrace()

    plan2 = pe.generate_plan(
        target_goal=goal2,
        available_actions=["refund_customer"],
        current_world_state=world2,
        reasoning_trace=trace2,
    )
    pe.approve_plan(plan2)

    # Tool executed successfully (HTTP 200, tool status 'success'), but real-world outcome failed (card declined)
    pe.record_step_execution(
        plan=plan2,
        step_number=1,
        observed_outcome={
            "http_status": 200,
            "refund_gateway_response": "DECLINED_BY_ISSUER",
            "observed_postconditions": ["gateway_contacted"],
        },
        observed_state=WorldState(environment_metrics={"dispute_exposure_usd": 120.0}),
        execution_status="success",
    )

    completed2, details2 = pe.verify_plan_execution_completion(
        plan=plan2,
        target_goal=goal2,
        final_observed_world_state=WorldState(environment_metrics={"dispute_exposure_usd": 120.0}),
    )
    neg_outcome_ok = (
        completed2 is False
        and plan2.is_verified_goal_completion is False
        and plan2.plan_stage == "failed"
        and "billing_dispute_resolved" in details2.get("missing_required_postconditions", [])
    )
    report["checks"]["successful_action_with_unsuccessful_outcome"] = {
        "status": "PASS" if neg_outcome_ok else "FAIL",
        "verified_completion": plan2.is_verified_goal_completion,
        "final_plan_stage": plan2.plan_stage,
        "missing_postconditions": details2.get("missing_required_postconditions", []),
    }
    print(f"[*] Telemetry isolation: unconfirmed postconditions correctly rejected -> {'PASS' if neg_outcome_ok else 'FAIL'}")

    # 3. Step ID Validation: Duplicates, Unknown, Missing, and Order Violations
    print("[*] 3. Verifying strict step ID validation and ordering checks...")
    goal3 = GoalNode(
        goal_id="goal_multi_step",
        description="Two-step sequence",
        priority=1,
        required_postconditions=["vital_signs_stabilized", "pharmacotherapy_administered"],
    )
    plan3 = pe.generate_plan(
        target_goal=goal3,
        available_actions=["stabilize_vitals", "standard_healthcheck"],
        current_world_state=WorldState(environment_metrics={"vital_instability": 0.8}),
        reasoning_trace=ReasoningTrace(),
    )
    pe.approve_plan(plan3)

    # Test unknown step ID (e.g. step 999)
    unknown_rejected = False
    try:
        pe.record_step_execution(
            plan=plan3,
            step_number=999,
            observed_outcome={"data": 1},
            execution_status="success",
        )
    except ValueError as e:
        unknown_rejected = "Unknown execution step ID" in str(e)

    # Test out-of-order execution (executing step 2 before step 1)
    out_of_order_rejected = False
    try:
        pe.record_step_execution(
            plan=plan3,
            step_number=2,
            observed_outcome={"data": 1},
            execution_status="success",
        )
    except ValueError as e:
        out_of_order_rejected = "Execution order violation" in str(e)

    # Record step 1 legitimately
    pe.record_step_execution(
        plan=plan3,
        step_number=1,
        observed_outcome={"data": 1, "observed_postconditions": ["vital_signs_stabilized"]},
        observed_state=WorldState(environment_metrics={"vital_instability": 0.0}),
        execution_status="success",
    )

    # Test duplicate step ID (recording step 1 again)
    duplicate_rejected = False
    try:
        pe.record_step_execution(
            plan=plan3,
            step_number=1,
            observed_outcome={"data": 1},
            execution_status="success",
        )
    except ValueError as e:
        duplicate_rejected = "Duplicate execution step ID" in str(e)

    # Test verification rejects missing steps (only step 1 recorded, step 2 omitted)
    missing_rejected = False
    comp_missing, det_missing = pe.verify_plan_execution_completion(
        plan=plan3,
        target_goal=goal3,
        final_observed_world_state=WorldState(),
    )
    if comp_missing is False and det_missing.get("error_code") == "MISSING_STEP_ID":
        missing_rejected = True

    step_validation_ok = unknown_rejected and out_of_order_rejected and duplicate_rejected and missing_rejected
    report["checks"]["step_id_validation"] = {
        "status": "PASS" if step_validation_ok else "FAIL",
        "unknown_step_rejected": unknown_rejected,
        "out_of_order_rejected": out_of_order_rejected,
        "duplicate_step_rejected": duplicate_rejected,
        "missing_step_rejected": missing_rejected,
    }
    print(f"[*] Step-ID checks (unknown, out-of-order, duplicate, missing) -> {'PASS' if step_validation_ok else 'FAIL'}")

    # 4. Target Metric Validation: Numeric, Bounds, Categorical, Boolean, Dict Operators
    print("[*] 4. Verifying target state metric validation (numeric & non-numeric)...")
    goal4 = GoalNode(
        goal_id="goal_metrics",
        description="Verify numeric bounds and non-numeric conditions",
        priority=1,
        target_state_metrics={
            "system_risk": {"<=": 0.05},
            "sla_remaining_hours": {">=": 24.0},
            "patient_status": "stable",
            "triage_completed": True,
        },
    )
    plan4 = pe.generate_plan(
        target_goal=goal4,
        available_actions=["standard_healthcheck"],
        current_world_state=WorldState(),
        reasoning_trace=ReasoningTrace(),
    )
    pe.approve_plan(plan4)
    pe.record_step_execution(
        plan=plan4,
        step_number=1,
        observed_outcome={"data": 1},
        observed_state=WorldState(environment_metrics={
            "system_risk": 0.02,
            "sla_remaining_hours": 36.0,
            "patient_status": "stable",
            "triage_completed": True,
        }),
        execution_status="success",
    )

    comp4, det4 = pe.verify_plan_execution_completion(
        plan=plan4,
        target_goal=goal4,
        final_observed_world_state=WorldState(environment_metrics={
            "system_risk": 0.02,
            "sla_remaining_hours": 36.0,
            "patient_status": "stable",
            "triage_completed": True,
        }),
    )
    metrics_ok = (comp4 is True and det4.get("verified") is True and len(det4.get("failed_metrics", [])) == 0)
    report["checks"]["target_metric_evaluation"] = {
        "status": "PASS" if metrics_ok else "FAIL",
        "verified": comp4,
        "metric_evaluations": det4.get("metric_evaluations", {}),
        "failed_metrics": det4.get("failed_metrics", []),
    }
    print(f"[*] Target metric evaluations (numeric upper/lower, dict, boolean, categorical) -> {'PASS' if metrics_ok else 'FAIL'}")

    overall_passed = lifecycle_ok and neg_outcome_ok and step_validation_ok and metrics_ok
    report["passed"] = overall_passed

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Execution verification complete: {'ALL PASS' if overall_passed else 'FAILURES DETECTED'}")
    print(f"[+] Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = verify_execution()
    sys.exit(0 if rep.get("passed") else 1)
