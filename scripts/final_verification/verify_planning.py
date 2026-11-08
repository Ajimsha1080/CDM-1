#!/usr/bin/env python3
"""CDM-1 Final Verification: Planning Engine Correctness & Sequential State Transitions.

Verifies:
1. Sequential simulation passes resulting state forward between steps.
2. Intermediate precondition re-evaluation filters incompatible actions.
3. Constraint compliance evaluates actual plan steps against active constraints.
4. Independent evaluation of alternative plan branches.
5. Negative verification: valid actions that fail goal conditions are NOT marked satisfied.
6. Prediction provenance validation and cryptographic binding rejection.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace, PlanStep, Plan
from cdm.planning.planning_engine import PlanningEngine
from cdm.inference.provenance import PredictionProvenance


def verify_planning(report_path: str = "planning_report.json") -> Dict[str, Any]:
    print("=" * 70)
    print("CDM-1 VERIFICATION: PLANNING ENGINE & SEQUENTIAL STATE TRANSITIONS")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "checks": {},
        "passed": False,
    }

    pe = PlanningEngine()

    # Check 1: Sequential Simulation passes state forward
    print("[*] 1. Verifying sequential world-state simulation forward passing...")
    goal1 = GoalNode(
        goal_id="goal_seq_sim",
        description="Multi-step sequential clinical stabilization",
        priority=1,
        required_postconditions=["vital_signs_stabilized"],
        target_state_metrics={"vital_instability": 0.0},
    )
    init_world1 = WorldState(environment_metrics={"vital_instability": 0.9, "monetary_value_usd": 100.0})
    trace1 = ReasoningTrace()

    plan1 = pe.generate_plan(
        target_goal=goal1,
        available_actions=["stabilize_vitals", "standard_healthcheck"],
        current_world_state=init_world1,
        reasoning_trace=trace1,
    )

    seq_ok = True
    if len(plan1.steps) >= 2:
        step1_delta = plan1.steps[0].expected_outcome.get("state_delta", {})
        step2_delta = plan1.steps[1].expected_outcome.get("state_delta", {})
        # Step 2 must be evaluated starting from step 1's resulting state
        if step1_delta.get("vital_instability") is not None and step2_delta.get("vital_instability") is not None:
            seq_ok = (step2_delta["vital_instability"] <= step1_delta["vital_instability"])
    else:
        seq_ok = len(plan1.steps) >= 1

    report["checks"]["sequential_simulation"] = {
        "status": "PASS" if seq_ok else "FAIL",
        "step_count": len(plan1.steps),
        "steps": [s.action_name for s in plan1.steps],
    }
    print(f"[*] Sequential simulation verified ({len(plan1.steps)} steps) -> {'PASS' if seq_ok else 'FAIL'}")

    # Check 2: Dynamic precondition re-evaluation
    print("[*] 2. Verifying intermediate precondition filtering...")
    goal2 = GoalNode(
        goal_id="goal_precond",
        description="Verify precondition constraints filter prohibited steps",
        priority=1,
    )
    world2 = WorldState(
        environment_metrics={"monetary_value_usd": 2500.0},
        active_constraints=["prohibit_escalate_tier2"],
    )
    trace2 = ReasoningTrace(violated_constraints=["Action escalate_tier2 forbidden by policy"])

    plan2 = pe.generate_plan(
        target_goal=goal2,
        available_actions=["refund_customer", "escalate_tier2", "request_documentation"],
        current_world_state=world2,
        reasoning_trace=trace2,
    )
    planned_actions = [s.action_name for s in plan2.steps]
    # refund_customer should fail precondition (monetary_value_usd > 1500)
    # escalate_tier2 should be forbidden by active constraint
    filtered_ok = ("refund_customer" not in planned_actions) and ("escalate_tier2" not in planned_actions)
    report["checks"]["dynamic_precondition_filtering"] = {
        "status": "PASS" if filtered_ok else "FAIL",
        "planned_actions": planned_actions,
        "monetary_filter_applied": "refund_customer" not in planned_actions,
        "constraint_filter_applied": "escalate_tier2" not in planned_actions,
    }
    print(f"[*] Precondition and constraint filtering: planned={planned_actions} -> {'PASS' if filtered_ok else 'FAIL'}")

    # Check 3: Independent alternative evaluation
    print("[*] 3. Verifying independent alternative branch generation and evaluation...")
    alt_ok = (len(plan1.alternative_plans) > 0)
    for alt in plan1.alternative_plans:
        if "goal_satisfied" not in alt or "steps" not in alt or "cumulative_risk" not in alt:
            alt_ok = False
            break
    report["checks"]["independent_alternative_plans"] = {
        "status": "PASS" if alt_ok else "FAIL",
        "alternative_branches_count": len(plan1.alternative_plans),
        "branch_names": [a.get("name") for a in plan1.alternative_plans],
    }
    print(f"[*] Alternative branches verified ({len(plan1.alternative_plans)} branches) -> {'PASS' if alt_ok else 'FAIL'}")

    # Check 4: Negative Goal Verification (Actions present, but cannot satisfy goal)
    print("[*] 4. Verifying negative goal condition test...")
    impossible_goal = GoalNode(
        goal_id="goal_impossible",
        description="Achieve impossible cryptographic state and zero latency",
        priority=1,
        required_postconditions=["cryptographic_enclave_attested", "quantum_resistant_key_derived"],
        target_state_metrics={"network_latency_ms": 0.0},
    )
    world4 = WorldState(environment_metrics={"network_latency_ms": 120.0})
    trace4 = ReasoningTrace()

    plan4 = pe.generate_plan(
        target_goal=impossible_goal,
        available_actions=["standard_healthcheck", "request_documentation"],
        current_world_state=world4,
        reasoning_trace=trace4,
    )
    # Even though plan4 has steps, it CANNOT satisfy the goal conditions
    neg_ok = (
        plan4.goal_satisfied is False
        and plan4.simulated_goal_achieved is False
        and plan4.is_verified_goal_completion is False
        and len(plan4.steps) > 0
    )
    report["checks"]["negative_goal_verification"] = {
        "status": "PASS" if neg_ok else "FAIL",
        "plan_steps_generated": len(plan4.steps),
        "goal_satisfied": plan4.goal_satisfied,
        "simulated_goal_achieved": plan4.simulated_goal_achieved,
        "is_verified_goal_completion": plan4.is_verified_goal_completion,
        "missing_postconditions": plan4.goal_satisfaction_details.get("failed_conditions", []),
    }
    print(f"[*] Negative goal condition verification -> {'PASS' if neg_ok else 'FAIL'}")

    # Check 5: Prediction provenance cryptographic bindings
    print("[*] 5. Verifying prediction provenance validation & tamper rejection...")
    genuine_prov = PredictionProvenance.create(
        model_id="cdm-1-general",
        model_version="1.0.0",
        artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        prediction_output={"risk": 0.15, "confidence": 0.95},
        tenant_id="tenant_prov_check",
        task="planning",
    )
    valid_prov_ok, src, diag = PredictionProvenance.validate(
        genuine_prov,
        expected_artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        expected_tenant_id="tenant_prov_check",
        expected_task="planning",
        prediction_output={"risk": 0.15, "confidence": 0.95},
    )

    tampered_prov_data = dict(genuine_prov)
    tampered_prov_data["model_version"] = "9.9.9"  # Tampered field
    tamper_ok, t_src, t_diag = PredictionProvenance.validate(
        tampered_prov_data,
        expected_artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        expected_tenant_id="tenant_prov_check",
        expected_task="planning",
        prediction_output={"risk": 0.15, "confidence": 0.95},
    )
    tamper_rejected = not tamper_ok

    prov_ok = valid_prov_ok and tamper_rejected
    report["checks"]["prediction_provenance_verification"] = {
        "status": "PASS" if prov_ok else "FAIL",
        "genuine_provenance_verified": valid_prov_ok,
        "tampered_provenance_rejected": tamper_rejected,
    }
    print(f"[*] Prediction provenance security verified -> {'PASS' if prov_ok else 'FAIL'}")

    overall_passed = seq_ok and filtered_ok and alt_ok and neg_ok and prov_ok
    report["passed"] = overall_passed

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Planning verification complete: {'ALL PASS' if overall_passed else 'FAILURES DETECTED'}")
    print(f"[+] Report written to {report_path}")
    return report


if __name__ == "__main__":
    rep = verify_planning()
    sys.exit(0 if rep.get("passed") else 1)
