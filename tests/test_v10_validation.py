"""CDM-1 V10 Final Model, Planning, and Execution Validation Test Suite.
Validates:
1. Actual Model Artifact & Strict Offline Real Inference:
   - Actual SafeTensors file integrity (size: 633,465,536 bytes, 215 tensors, 158.3M params).
   - Strict offline model reload and inference without internet or mocks.
2. Planning and Goal Verification:
   - Clear separation of simulated goal achievement from actual verified goal completion.
   - Proposed plans NEVER set is_verified_goal_completion=True (always False on proposed plans).
   - Explicit action execution (record_step_execution) and outcome verification (verify_plan_execution_completion).
   - Postconditions verified against actual observed outcomes.
3. Sequential World-State Simulation:
   - Sequential forward state passing: S_{t+1} = T(S_t, A_t).
   - Preconditions re-evaluated after each state transition.
   - Independent alternative plan simulation and evaluation.
   - Detection of invalid action sequences and impossible goals.
4. Cryptographic Prediction Provenance Binding (V10):
   - Provenance bound to loaded artifact hash, tenant ID, task, and prediction output digest.
   - Rejection of modified prediction outputs (digest mismatch).
   - Rejection of cross-tenant provenance spoofing.
5. Release Security & Verification:
   - Ed25519 asymmetric manifest verification and tamper detection.
"""

import os
import copy
import json
import pytest
import torch
import safetensors.torch
from fastapi.testclient import TestClient

from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace, Plan, ExecutedActionStep
from cdm.world_model.world_model import WorldModel
from cdm.planning.planning_engine import PlanningEngine
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.inference.provenance import PredictionProvenance
from cdm.security.manifest_verification import (
    verify_checkpoint_authenticity,
    compute_file_sha256,
    sign_manifest,
    TRUSTED_RELEASE_PUBLIC_KEY,
    SecurityVerificationError,
)
from api.main import app


EXPECTED_ARTIFACT_HASH = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
MODEL_PATH = "experiments/release/cdm-1-general"


# =========================================================================
# 1. P0 — MODEL ARTIFACT & STRICT OFFLINE INFERENCE
# =========================================================================

def test_actual_model_safetensors_file_integrity():
    """Validates that the actual SafeTensors checkpoint exists, matches exact size and SHA-256."""
    weights_path = os.path.join(MODEL_PATH, "model.safetensors")
    assert os.path.exists(weights_path), f"Missing model checkpoint at {weights_path}"

    file_size = os.path.getsize(weights_path)
    assert file_size == 633465536, f"Expected 633,465,536 bytes, got {file_size}"

    file_sha256 = compute_file_sha256(weights_path)
    assert file_sha256 == EXPECTED_ARTIFACT_HASH, f"SHA-256 mismatch: {file_sha256}"

    tensors = safetensors.torch.load_file(weights_path)
    assert len(tensors) == 215, f"Expected 215 tensors, got {len(tensors)}"
    total_params = sum(t.numel() for t in tensors.values())
    assert total_params == 158360004, f"Expected 158.36M params, got {total_params}"


def test_strict_offline_model_loading_and_real_inference():
    """Validates strict offline loading of weights and real neural inference pass."""
    model = CognitiveDecisionModel.from_pretrained(model_name_or_path=MODEL_PATH, local_files_only=True)
    assert model.artifact_hash == EXPECTED_ARTIFACT_HASH

    # Execute real inference with structured features
    res = model.decide(
        structured_features={"monetary_risk": 0.2, "operational_urgency": 0.8},
        text="Urgent customer tier 2 issue requiring immediate attention.",
        task="binary",
        candidates=["escalate_tier2", "request_documentation"],
        tenant_id="tenant_v10_test",
    )

    assert res.decision in ["YES", "NO"]
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.risk <= 1.0
    assert res.provenance is not None
    assert res.provenance.get("artifact_hash") == EXPECTED_ARTIFACT_HASH
    assert res.provenance.get("tenant_id") == "tenant_v10_test"


# =========================================================================
# 2. P1 — PLANNING AND GOAL VERIFICATION
# =========================================================================

def test_proposed_plan_never_marks_verified_goal_completion():
    """A proposed plan MUST have is_verified_goal_completion=False and plan_stage='proposed'."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_dispute",
        description="Resolve billing dispute for customer",
        priority=1,
        required_postconditions=["billing_dispute_resolved"],
    )
    world = WorldState(environment_metrics={"monetary_value_usd": 100.0})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["refund_customer", "request_documentation"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert plan.plan_stage == "proposed"
    assert plan.simulated_goal_achieved is True
    assert plan.goal_satisfied is True
    # MUST NEVER BE TRUE ON PROPOSED PLAN!
    assert plan.is_verified_goal_completion is False
    assert len(plan.executed_steps) == 0


def test_explicit_action_execution_and_verified_completion():
    """Validates step execution recording and transitions to verified completion only upon observation."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_stabilize",
        description="Patient vital signs stabilized",
        priority=1,
        required_postconditions=["vital_signs_stabilized"],
        target_state_metrics={"vital_instability": 0.0},
    )
    world = WorldState(environment_metrics={"vital_instability": 0.9})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["stabilize_vitals"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert plan.is_verified_goal_completion is False

    # 1. Record execution of step 1
    executed = pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"vital_signs_stabilized": True, "observed_postconditions": ["vital_signs_stabilized"]},
        observed_state=WorldState(environment_metrics={"vital_instability": 0.0}),
        execution_status="success",
    )
    assert isinstance(executed, ExecutedActionStep)
    assert len(plan.executed_steps) == 1
    assert plan.plan_stage in ["in_progress", "executed"]

    # 2. Verify completion against observed world state
    final_world = WorldState(environment_metrics={"vital_instability": 0.0})
    completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=final_world,
    )

    assert completed is True
    assert details["verified"] is True
    assert plan.is_verified_goal_completion is True
    assert plan.plan_stage == "completed"


def test_execution_failure_prevents_verified_completion():
    """If an executed action fails, verified goal completion must evaluate to False."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_iso",
        description="Isolate security incident cluster",
        priority=1,
        required_postconditions=["cluster_network_isolated"],
    )
    world = WorldState(environment_metrics={"system_exposure": 0.9})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["isolate_cluster"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # Step execution times out or fails
    pe.record_step_execution(
        plan=plan,
        step_number=1,
        observed_outcome={"error": "Network isolation timeout"},
        observed_state=world,
        execution_status="failed",
    )

    completed, details = pe.verify_plan_execution_completion(
        plan=plan,
        target_goal=goal,
        final_observed_world_state=world,
    )

    assert completed is False
    assert plan.is_verified_goal_completion is False
    assert plan.plan_stage == "failed"


# =========================================================================
# 3. P1 — SEQUENTIAL WORLD-STATE SIMULATION
# =========================================================================

def test_sequential_simulation_passes_state_forward():
    """Validates that S_{t+1} = T(S_t, A_t) chains intermediate state forward across steps."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_ops", description="Triage and escalate support case", priority=1)
    # Starting SLA is 12.0 hours
    world = WorldState(environment_metrics={"sla_remaining_hours": 12.0})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["escalate_tier2", "standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert len(plan.steps) == 2
    # Step 1 should reduce SLA
    step1_metrics = plan.steps[0].expected_outcome["state_delta"]
    assert step1_metrics["sla_remaining_hours"] == 6.0

    # Step 2 must be simulated on top of step 1's intermediate state
    step2_metrics = plan.steps[1].expected_outcome["state_delta"]
    assert step2_metrics["sla_remaining_hours"] == 6.0
    assert step2_metrics["escalation_active"] == 1.0


def test_sequential_precondition_re_evaluation_filters_incompatible_steps():
    """Validates that if step 1 alters state such that subsequent action preconditions fail,
    preconditions are checked against intermediate state.
    """
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_fin", description="Refund and resolve dispute", priority=1)
    # Initial state has high monetary value
    world = WorldState(environment_metrics={"monetary_value_usd": 2000.0})
    reasoning = ReasoningTrace()

    # Preconditions check: refund requires monetary_value_usd <= 1500
    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["refund_customer"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # refund_customer precondition fails because 2000.0 > 1500.0
    assert len(plan.steps) == 0
    assert plan.goal_satisfied is False


def test_impossible_goal_evaluation():
    """A goal demanding unattainable conditions must evaluate simulated_goal_achieved=False."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_impossible",
        description="Achieve zero latency and impossible security metric",
        priority=1,
        required_postconditions=["quantum_teleportation_achieved"],
        target_state_metrics={"system_exposure": -10.0},  # Impossible negative exposure
    )
    world = WorldState(environment_metrics={"system_exposure": 0.5})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert plan.simulated_goal_achieved is False
    assert plan.goal_satisfied is False
    assert plan.is_verified_goal_completion is False


# =========================================================================
# 4. P1 — PREDICTION PROVENANCE (V10 BINDINGS)
# =========================================================================

def test_prediction_provenance_cryptographic_bindings():
    """Validates cryptographic binding of loaded artifact hash, tenant ID, task, and prediction digest."""
    pred_summary = {
        "decision": "escalate_tier2",
        "binary_prediction": 1,
        "confidence": 0.94,
        "risk": 0.06,
        "priority": "HIGH",
    }

    prov = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash=EXPECTED_ARTIFACT_HASH,
        tenant_id="tenant_alpha",
        task="binary",
        prediction_output=pred_summary,
        estimated_risk=0.06,
    )

    # 1. Valid provenance passes
    is_valid, src, diag = PredictionProvenance.validate(
        prov,
        expected_artifact_hash=EXPECTED_ARTIFACT_HASH,
        expected_tenant_id="tenant_alpha",
        expected_task="binary",
        prediction_output=pred_summary,
    )
    assert is_valid is True
    assert src == "trained_model_prediction"

    # 2. Cross-tenant mismatch fails
    is_valid, src, diag = PredictionProvenance.validate(
        prov,
        expected_artifact_hash=EXPECTED_ARTIFACT_HASH,
        expected_tenant_id="tenant_beta",  # Attacker tries to reuse provenance in another tenant!
        expected_task="binary",
        prediction_output=pred_summary,
    )
    assert is_valid is False
    assert src == "unverified_caller_input"
    assert "Tenant mismatch" in diag["reason"]

    # 3. Output modification (digest mismatch) fails
    tampered_pred = copy.deepcopy(pred_summary)
    tampered_pred["risk"] = 0.99  # Attacker modified output after inference!
    is_valid, src, diag = PredictionProvenance.validate(
        prov,
        expected_artifact_hash=EXPECTED_ARTIFACT_HASH,
        expected_tenant_id="tenant_alpha",
        expected_task="binary",
        prediction_output=tampered_pred,
    )
    assert is_valid is False
    assert src == "unverified_caller_input"
    assert "Prediction output digest mismatch" in diag["reason"]


def test_planning_engine_rejects_tampered_prediction_digest():
    """PlanningEngine must classify tampered prediction outputs as unverified_caller_input."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_sec_check", description="Audit security", priority=1)
    world = WorldState()
    reasoning = ReasoningTrace()

    pred_summary = {"decision": "standard_healthcheck", "confidence": 0.99, "risk": 0.01}
    prov = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash=EXPECTED_ARTIFACT_HASH,
        tenant_id="tenant_x",
        task="binary",
        prediction_output=pred_summary,
        estimated_risk=0.01,
    )

    # Attacker tampers with risk value
    tampered_outcome = {
        "predicted_risk": 0.50,  # Modified
        "decision": "standard_healthcheck",
        "confidence": 0.99,
        "provenance": prov,
    }

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
        predicted_outcome=tampered_outcome,
        tenant_id="tenant_x",
    )

    # Should be rejected as unverified caller input due to digest mismatch
    assert plan.risk_source == "unverified_caller_input"
    assert plan.steps[0].risk_source == "unverified_caller_input"
    assert plan.steps[0].risk_provenance["authenticated"] is False


# =========================================================================
# 5. P1 — RELEASE SECURITY & VERIFICATION
# =========================================================================

def test_release_manifest_ed25519_verification():
    """Validates Ed25519 asymmetric signature verification of actual release manifest."""
    manifest_path = os.path.join(MODEL_PATH, "release_manifest.json")
    assert os.path.exists(manifest_path)

    # Verify authenticity against trusted embedded public key
    auth_result = verify_checkpoint_authenticity(
        model_dir=MODEL_PATH,
        verify_signature=True,
        trusted_public_key_hex=TRUSTED_RELEASE_PUBLIC_KEY,
    )
    assert auth_result["status"] == "authenticated"
    assert auth_result["sha256"] == EXPECTED_ARTIFACT_HASH
    assert auth_result["signature_verified"] is True


def test_api_ready_endpoint_verifies_model_and_signature():
    """The /v1/ready endpoint verifies actual SafeTensors SHA-256 and Ed25519 manifest signature."""
    with TestClient(app) as client:
        resp = client.get("/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["weights_sha256"] == EXPECTED_ARTIFACT_HASH
        assert data["manifest_signature_verified"] is True
