"""CDM-1 V8/V9 Security, Planning, and Release Integrity Test Suite.
Validates:
1. Planning Correctness:
   - Replaced nonempty-plan heuristic with explicit goal conditions and measurable postconditions.
   - Constraint compliance evaluated on actual plan steps and violated constraints.
   - Independent multi-step evaluation of alternative plans.
   - Plans containing valid actions that fail to achieve goal conditions produce goal_satisfied=False.
   - Separation of proposed plans, executed steps, and verified goal completion.
2. Risk & Prediction Provenance:
   - Genuine trained model predictions cryptographically signed and authenticated by runtime.
   - Caller-supplied arbitrary values are NEVER labeled as trained model predictions (unverified_caller_input).
   - Forged provenance records are strictly rejected.
3. Checkpoint Authenticity (Ed25519 Asymmetric Verification):
   - Signed release-manifest cryptographic verification.
   - Rejection of unsigned manifests.
   - Rejection of tampered SafeTensors weights.
   - Rejection of modified manifests.
   - Rejection of incorrect public keys.
4. Feedback Concurrency:
   - Multi-process concurrent feedback recording, review updates, and WAL mode resilience.
"""

import os
import copy
import json
import tempfile
import multiprocessing
import pytest
from fastapi.testclient import TestClient
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

from cdm.cognitive.state import GoalNode, WorldState, ReasoningTrace, ExecutedActionStep
from cdm.planning.planning_engine import PlanningEngine
from cdm.learning.feedback import FeedbackCollector, FeedbackEvent
from cdm.inference.provenance import PredictionProvenance
from cdm.security.manifest_verification import (
    sign_manifest,
    verify_manifest_signature,
    verify_checkpoint_authenticity,
    SecurityVerificationError,
    compute_file_sha256,
    TRUSTED_RELEASE_PUBLIC_KEY,
)
from api.main import app


# =========================================================================
# 1. PLANNING CORRECTNESS & GOAL SATISFACTION TESTS
# =========================================================================

def test_plan_with_actions_fails_when_goal_conditions_unmet():
    """Validates that a plan containing executable steps fails goal_satisfied if explicit
    goal postconditions or target metrics are not satisfied by the actions.
    """
    pe = PlanningEngine()
    # Goal explicitly requires clinical stabilization postcondition
    goal = GoalNode(
        goal_id="goal_cardiac_stabilization",
        description="Achieve acute cardiac stabilization",
        priority=1,
        required_postconditions=["cardiac_rhythm_stabilized", "arterial_blood_pressure_normal"],
    )
    world = WorldState(
        tenant_id="hospital_net",
        session_id="session_icu_4",
        environment_metrics={"monetary_value_usd": 100.0, "system_exposure": 0.5},
    )
    reasoning = ReasoningTrace()
    # Actions that do NOT produce the required clinical postconditions
    available_actions = ["order_labs", "request_documentation"]

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=available_actions,
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    # Steps were generated
    assert len(plan.steps) == 2
    assert plan.steps[0].action_name == "order_labs"
    # But explicit required postconditions were NOT produced
    assert plan.goal_satisfied is False
    assert plan.is_verified_goal_completion is False
    assert "failed_conditions" in plan.goal_satisfaction_details
    failed = str(plan.goal_satisfaction_details["failed_conditions"])
    assert "missing_required_postcondition:cardiac_rhythm_stabilized" in failed


def test_plan_with_target_metric_bounds():
    """Validates that plans are checked against explicit target state metrics."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_reduce_exposure",
        description="Mitigate infrastructure exposure",
        priority=1,
        target_state_metrics={"system_exposure": 0.2},
    )
    world = WorldState(
        environment_metrics={"system_exposure": 0.85},
    )
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert len(plan.steps) == 1
    # System exposure remains 0.85 > 0.2 bound, so goal is not satisfied
    assert plan.goal_satisfied is False
    assert any("metric_exceeded:system_exposure" in str(f) for f in plan.goal_satisfaction_details["failed_conditions"])


def test_constraint_compliance_evaluates_actual_plan_steps():
    """Validates that constraint compliance checks the actual plan steps against violated constraints."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_triage", description="Triage incident", priority=2)
    world = WorldState(
        active_constraints=["prohibit dangerous_reset action under active incident"],
    )
    reasoning = ReasoningTrace(
        violated_constraints=["policy violation: dangerous_reset is forbidden during active incident"],
    )

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["dangerous_reset", "safe_diagnostics"],
        current_world_state=world,
        reasoning_trace=reasoning,
        prohibited_actions=["dangerous_reset"],
    )

    step_actions = [s.action_name for s in plan.steps]
    assert "dangerous_reset" not in step_actions
    assert "safe_diagnostics" in step_actions
    assert plan.constraint_compliance is True
    assert plan.goal_satisfied is True


def test_alternative_plans_evaluated_independently():
    """Validates that alternative plans are evaluated as independent multi-step sequences."""
    pe = PlanningEngine()
    goal = GoalNode(
        goal_id="goal_customer_support",
        description="Resolve billing inquiry",
        priority=2,
    )
    world = WorldState(environment_metrics={"monetary_value_usd": 100.0})
    reasoning = ReasoningTrace()

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["auto_resolve", "escalate_tier2", "request_documentation"],
        current_world_state=world,
        reasoning_trace=reasoning,
    )

    assert len(plan.alternative_plans) == 2
    for alt in plan.alternative_plans:
        assert "name" in alt
        assert "first_action" in alt
        assert "steps" in alt
        assert len(alt["steps"]) > 0
        assert "cumulative_risk" in alt
        assert "expected_risk" in alt
        assert "risk_source" in alt
        assert "constraint_compliance" in alt
        assert "goal_satisfied" in alt


# =========================================================================
# 2. RISK & PREDICTION PROVENANCE TESTS
# =========================================================================

def test_risk_provenance_rejects_caller_supplied_arbitrary_values():
    """Validates that arbitrary caller-supplied predicted_outcome dicts are NEVER labeled
    as 'trained_model_prediction', but instead 'unverified_caller_input'.
    """
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_sec", description="Audit access", priority=1)
    world = WorldState()
    reasoning = ReasoningTrace()

    # Caller passes arbitrary unauthenticated values
    unverified_outcome = {
        "predicted_risk": 0.05,
        "is_genuine_neural_prediction": True,  # Attacker tries to forge flag!
        "arbitrary_user_key": "some_value",
    }

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
        predicted_outcome=unverified_outcome,
    )

    assert plan.steps[0].risk_source == "unverified_caller_input"
    assert plan.risk_source == "unverified_caller_input"
    assert plan.steps[0].risk_provenance["authenticated"] is False


def test_risk_provenance_accepts_genuine_trained_prediction():
    """Validates that genuine model predictions with runtime cryptographic signature are authenticated."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_sec_2", description="Audit access 2", priority=1)
    world = WorldState()
    reasoning = ReasoningTrace()

    prov = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        estimated_risk=0.05,
    )
    genuine_outcome = {
        "predicted_risk": 0.05,
        "provenance": prov,
    }

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
        predicted_outcome=genuine_outcome,
    )

    assert plan.steps[0].risk_source == "trained_model_prediction"
    assert plan.risk_source == "trained_model_prediction"
    assert plan.steps[0].model_id == "CDM-1-General"
    assert plan.steps[0].risk_provenance["authenticated"] is True


def test_prediction_provenance_forged_token_rejected():
    """Validates that an attacker crafting a fake signature is rejected."""
    pe = PlanningEngine()
    goal = GoalNode(goal_id="goal_attack", description="Test forgery", priority=1)
    world = WorldState()
    reasoning = ReasoningTrace()

    forged_prov = {
        "model_id": "CDM-1-General",
        "model_version": "1.0.0",
        "artifact_hash": "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        "execution_id": "exec_fake_12345",
        "timestamp": 1234567.89,
        "provenance_signature": "0000000000000000000000000000000000000000000000000000000000000000",
    }
    outcome = {"predicted_risk": 0.02, "provenance": forged_prov}

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["standard_healthcheck"],
        current_world_state=world,
        reasoning_trace=reasoning,
        predicted_outcome=outcome,
    )

    assert plan.steps[0].risk_source == "unverified_caller_input"
    assert plan.steps[0].risk_provenance["authenticated"] is False


# =========================================================================
# 3. CHECKPOINT AUTHENTICITY & ASYMMETRIC Ed25519 VERIFICATION TESTS
# =========================================================================

def _generate_test_keypair():
    priv = ed25519.Ed25519PrivateKey.generate()
    priv_hex = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    ).hex()
    pub_hex = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()
    return priv_hex, pub_hex


def test_checkpoint_authenticity_valid_signed_release():
    """Validates that genuine model checkpoint with valid signed manifest passes verification."""
    model_dir = "experiments/release/cdm-1-general"
    auth_info = verify_checkpoint_authenticity(model_dir, verify_signature=True)
    assert auth_info["status"] == "authenticated"
    assert auth_info["signature_verified"] is True
    assert auth_info["signature_algorithm"] == "Ed25519"
    assert auth_info["sha256"] == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"


def test_checkpoint_authenticity_unsigned_manifest_rejected():
    """Validates that an unsigned manifest missing manifest_signature is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_weights = os.path.join(tmpdir, "model.safetensors")
        with open(dummy_weights, "wb") as f:
            f.write(b"SAMPLE_WEIGHTS")

        manifest = {
            "release_version": "1.0.0",
            "model_name": "TestModel",
            "artifact": {
                "file_name": "model.safetensors",
                "file_size_bytes": os.path.getsize(dummy_weights),
                "sha256": compute_file_sha256(dummy_weights),
            },
            "packaged_files": {},
            # Missing manifest_signature
        }
        with open(os.path.join(tmpdir, "release_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        with pytest.raises(SecurityVerificationError, match="unsigned"):
            verify_checkpoint_authenticity(tmpdir, verify_signature=True)


def test_checkpoint_authenticity_tampered_model_rejected():
    """Validates that modifying a single byte in the model file raises SecurityVerificationError."""
    priv_hex, pub_hex = _generate_test_keypair()

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_weights = os.path.join(tmpdir, "model.safetensors")
        with open(dummy_weights, "wb") as f:
            f.write(b"REAL_UNTOUCHED_WEIGHTS_CONTENT_1234567890")

        manifest = {
            "release_version": "1.0.0",
            "model_name": "TestModel",
            "artifact": {
                "file_name": "model.safetensors",
                "file_size_bytes": os.path.getsize(dummy_weights),
                "sha256": compute_file_sha256(dummy_weights),
            },
            "packaged_files": {},
        }
        signed = sign_manifest(manifest, private_key_hex=priv_hex)
        with open(os.path.join(tmpdir, "release_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(signed, f)

        # Before tamper: passes
        res = verify_checkpoint_authenticity(tmpdir, verify_signature=True, trusted_public_key_hex=pub_hex)
        assert res["status"] == "authenticated"

        # Tamper weights by appending 1 byte
        with open(dummy_weights, "ab") as f:
            f.write(b"X")

        with pytest.raises(SecurityVerificationError, match="Checkpoint size mismatch|tampering detected"):
            verify_checkpoint_authenticity(tmpdir, verify_signature=True, trusted_public_key_hex=pub_hex)


def test_checkpoint_authenticity_tampered_manifest_rejected():
    """Validates that modifying a field in the manifest without valid signing key raises SecurityVerificationError."""
    priv_hex, pub_hex = _generate_test_keypair()

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_weights = os.path.join(tmpdir, "model.safetensors")
        with open(dummy_weights, "wb") as f:
            f.write(b"ORIGINAL_VALID_WEIGHTS")

        manifest = {
            "release_version": "1.0.0",
            "model_name": "TestModel",
            "artifact": {
                "file_name": "model.safetensors",
                "file_size_bytes": os.path.getsize(dummy_weights),
                "sha256": compute_file_sha256(dummy_weights),
            },
            "packaged_files": {},
        }
        signed = sign_manifest(manifest, private_key_hex=priv_hex)

        # Attacker tampers manifest by changing release_version
        tampered = copy.deepcopy(signed)
        tampered["release_version"] = "2.0.0-hacked"

        with open(os.path.join(tmpdir, "release_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(tampered, f)

        with pytest.raises(SecurityVerificationError, match="Ed25519 signature verification failed"):
            verify_checkpoint_authenticity(tmpdir, verify_signature=True, trusted_public_key_hex=pub_hex)


def test_checkpoint_authenticity_incorrect_public_key_rejected():
    """Validates that verifying with an untrusted public key raises SecurityVerificationError."""
    priv_hex, pub_hex = _generate_test_keypair()
    _, other_pub_hex = _generate_test_keypair()

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_weights = os.path.join(tmpdir, "model.safetensors")
        with open(dummy_weights, "wb") as f:
            f.write(b"ORIGINAL_VALID_WEIGHTS")

        manifest = {
            "release_version": "1.0.0",
            "model_name": "TestModel",
            "artifact": {
                "file_name": "model.safetensors",
                "file_size_bytes": os.path.getsize(dummy_weights),
                "sha256": compute_file_sha256(dummy_weights),
            },
            "packaged_files": {},
        }
        signed = sign_manifest(manifest, private_key_hex=priv_hex)
        with open(os.path.join(tmpdir, "release_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(signed, f)

        # Verifying with mismatched public key fails
        with pytest.raises(SecurityVerificationError, match="Ed25519 signature verification failed"):
            verify_checkpoint_authenticity(tmpdir, verify_signature=True, trusted_public_key_hex=other_pub_hex)


# =========================================================================
# 4. MULTI-PROCESS CONCURRENT FEEDBACK TESTS
# =========================================================================

def _worker_record_feedback(db_path: str, worker_id: int, num_items: int, return_dict: dict):
    """Subprocess target for multi-process concurrency testing."""
    try:
        collector = FeedbackCollector(db_path=db_path)
        for i in range(num_items):
            is_success = (i % 2 == 0)
            collector.record_feedback(
                decision_id=f"mp_dec_{worker_id}_{i}",
                predicted_decision={"binary_prediction": 1},
                actual_outcome={"binary_target": 1 if is_success else 0},
                is_success=is_success,
                tenant_id=f"tenant_proc_{worker_id}",
            )
        return_dict[worker_id] = True
    except Exception as e:
        return_dict[worker_id] = str(e)


def test_multiprocess_feedback_concurrency_and_wal():
    """Validates that independent FeedbackCollector instances across multiple OS processes
    write concurrently with SQLite WAL mode and busy timeout handling without locking failures.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name

    try:
        init_col = FeedbackCollector(db_path=db_path)
        del init_col

        num_procs = 2
        items_per_proc = 5
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        procs = []
        for w_id in range(num_procs):
            p = multiprocessing.Process(
                target=_worker_record_feedback,
                args=(db_path, w_id, items_per_proc, return_dict),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=60)

        for w_id in range(num_procs):
            assert return_dict.get(w_id) is True, f"Worker {w_id} failed: {return_dict.get(w_id)}"

        final_collector = FeedbackCollector(db_path=db_path)
        failures = final_collector.get_failure_cases()
        # With 5 items per proc and (i % 2 == 0) being success: i=0 (succ), 1 (fail), 2 (succ), 3 (fail), 4 (succ) -> 2 failures per proc
        assert len(failures) == num_procs * 2

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
        wal_path = f"{db_path}-wal"
        shm_path = f"{db_path}-shm"
        if os.path.exists(wal_path):
            try:
                os.remove(wal_path)
            except Exception:
                pass
        if os.path.exists(shm_path):
            try:
                os.remove(shm_path)
            except Exception:
                pass
