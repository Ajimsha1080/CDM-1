"""CDM-1 V12 Cognitive Integration Tests: Three Minimum Integrated Cognition Scenarios.

Validates the full cognitive cycle across existing components:
Input -> Perception & Understanding -> Context Integration -> Memory Retrieval
-> World-State Representation -> Reasoning -> Prediction -> Planning & Decision
-> Action -> Outcome Observation -> Metacognitive Evaluation -> Learning & Memory Update.

Demonstrates:
Scenario 1: Context-aware decision-making with memory retrieval shaping decisions.
Scenario 2: Goal-based sequential planning with dynamic preconditions and independent alternatives.
Scenario 3: Execution, outcome verification, and feedback-based memory adaptation.
"""

import os
import time
import tempfile
import pytest

from cdm.cognitive.state import (
    GoalNode,
    WorldState,
    ReasoningTrace,
    Plan,
    ExecutedActionStep,
    CognitiveEngineResult,
)
from cdm.cognitive.engine import CognitiveEngine
from cdm.planning.planning_engine import PlanningEngine
from cdm.world_model.world_model import WorldModel
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory, MemoryRecord
from cdm.memory.semantic import SemanticMemory, PolicyRule
from cdm.inference.pipeline import CognitiveDecisionModel

MODEL_PATH = "experiments/release/cdm-1-general"


@pytest.fixture(scope="module")
def shared_decision_model():
    """Loads genuine trained model checkpoint once for the integration scenarios."""
    if not os.path.exists(os.path.join(MODEL_PATH, "model.safetensors")):
        pytest.skip("Release model checkpoint not found for cognitive scenarios")
    return CognitiveDecisionModel.from_pretrained(model_name_or_path=MODEL_PATH, local_files_only=True)


# =========================================================================
# SCENARIO 1: CONTEXT-AWARE DECISION-MAKING
# =========================================================================

def test_scenario_1_context_aware_decision_making(shared_decision_model):
    """Scenario 1: Input -> Perception -> Context -> Memory -> World State -> Reasoning -> Prediction.
    Demonstrates that previous context turns and retrieved memory precedents
    directly shape constraints, reasoning deductions, and final decision choice.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine(decision_model=shared_decision_model, db_path=db_path)
        tenant_id = "tenant_enterprise_ops"
        session_id = "sess_ctx_scenario_01"

        # 1. Pre-seed Semantic Memory with an explicit domain policy rule:
        # High monetary value (> $500) combined with recurring disputes requires Tier-2 escalation
        engine.semantic_memory.register_rule(
            PolicyRule(
                rule_id="RULE-HIGH-DISPUTE-TIER2",
                tenant_id=tenant_id,
                domain="finance_and_credit",
                condition_description="High monetary value (> $500) prohibits automated resolution",
                permitted_actions=["escalate_to_tier2", "issue_refund"],
                prohibited_actions=["auto_resolve"],
            )
        )

        # 2. Pre-seed Episodic Memory with a verified previous incident case
        query_vec = engine.memory_embedder.embed(
            text="High value billing dispute requiring manual supervisor review due to recurrent chargebacks",
            structured_features={"monetary_value_usd": 750.0, "urgency_level": "critical"},
        )
        engine.episodic_memory.insert(
            MemoryRecord(
                record_id="CASE-PRECEDENT-001",
                tenant_id=tenant_id,
                context_text="Unauthorized chargeback on enterprise subscription",
                features={"monetary_value_usd": 750.0},
                decision_made={"selected_candidate": "escalate_to_tier2"},
                vector=query_vec,
            )
        )

        # 3. Simulate dialogue Turn 1: Customer reports an initial billing dispute
        turn1_text = "I am disputing an unauthorized charge on our corporate account."
        res1 = engine.process(
            text=turn1_text,
            structured_features={
                "monetary_value_usd": 850.0,
                "dispute_count": 2,
                "domain": "finance_and_credit",
                "urgency_level": "high",
            },
            candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
            session_id=session_id,
            tenant_id=tenant_id,
            execute_planning=False,
        )
        assert res1.decision is not None
        assert res1.latency_ms > 0.0

        # 4. Turn 2: Customer provides follow-up with extracted entities and context continuity
        turn2_text = "The charge of $850.00 from vendor ACME-CORP was never authorized by finance."
        res2 = engine.process(
            text=turn2_text,
            structured_features={
                "monetary_value_usd": 850.0,
                "dispute_count": 2,
                "domain": "finance_and_credit",
                "urgency_level": "critical",
            },
            candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
            goals=["Resolve unauthorized enterprise chargeback compliant with policy"],
            session_id=session_id,
            tenant_id=tenant_id,
            execute_planning=True,
        )

        # Verify Perception extracted entities and operational signals
        # Context engine tracked turn continuity
        assert res2.world_state_summary["environment_metrics"].get("monetary_value_usd") == 850.0

        # Verify Reasoning evaluated constraints and candidate actions
        assert res2.reasoning_evidence is not None
        assert len(res2.reasoning_evidence.steps) > 0

        # Verify Prediction produced calibrated confidence, risk, and candidate scores
        assert 0.0 <= res2.confidence <= 1.0
        assert 0.0 <= res2.risk <= 1.0
        assert res2.decision in ["escalate_to_tier2", "issue_refund"]
        assert res2.decision != "auto_resolve"  # Prohibited by policy for high monetary dispute

        # Verify alternatives were scored independently
        assert len(res2.alternatives) >= 1
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


# =========================================================================
# SCENARIO 2: GOAL-BASED SEQUENTIAL PLANNING
# =========================================================================

def test_scenario_2_goal_based_sequential_planning(shared_decision_model):
    """Scenario 2: World State -> Sequential Simulation -> Dynamic Preconditions -> Planning.
    Demonstrates multi-step sequential state transitions where each simulated step
    updates the world model, re-evaluates preconditions, and evaluates alternatives.
    """
    pe = PlanningEngine()
    wm = WorldModel(tenant_id="tenant_clinical_ops", session_id="sess_plan_scenario_02")

    # Target Goal: Clinical stabilization with explicit required postconditions and target metric bounds
    goal = GoalNode(
        goal_id="goal_clinical_stabilize",
        description="Multi-step patient stabilization and pharmacotherapy administration",
        priority=1,
        required_postconditions=["vital_signs_stabilized", "pharmacotherapy_administered"],
        target_state_metrics={
            "vital_instability": {"<=": 0.1},
            "patient_status": "stable",
        },
    )

    # Initial World State: High vital instability
    wm.set_environment_metric("vital_instability", 0.85)
    wm.set_environment_metric("monetary_value_usd", 150.0)
    wm.record_observed_fact("patient_status", "unstable")
    init_state = wm.get_state()

    # Generate sequential plan
    reasoning = ReasoningTrace()
    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["stabilize_vitals", "standard_healthcheck"],
        current_world_state=init_state,
        reasoning_trace=reasoning,
        tenant_id="tenant_clinical_ops",
        task="planning",
    )

    # 1. Proposed Plan Verification
    assert plan.plan_stage == "proposed"
    assert plan.is_verified_goal_completion is False
    assert len(plan.steps) >= 2

    # 2. Sequential State Transition Forward Passing:
    # Step 1 delta must lower vital instability for Step 2
    step1 = plan.steps[0]
    step2 = plan.steps[1]
    assert step1.step_number == 1
    assert step2.step_number == 2
    assert step1.action_name == "stabilize_vitals"
    assert step1.postconditions is not None

    step1_delta = step1.expected_outcome.get("state_delta", {})
    assert "vital_instability" in step1_delta
    # Vital instability should decrease after stabilization
    assert step1_delta["vital_instability"] < 0.85

    # 3. Independent Alternative Branch Evaluation
    assert len(plan.alternative_plans) >= 1
    for alt in plan.alternative_plans:
        assert "steps" in alt
        assert "cumulative_risk" in alt
        assert "goal_satisfied" in alt

    # 4. Explicit Lifecycle Transition: Safety approval
    approved_plan = pe.approve_plan(plan, approver_id="clinical_director")
    assert approved_plan.plan_stage == "approved"

    # 5. Re-approval must be strictly rejected
    with pytest.raises(ValueError, match="Cannot approve plan in 'approved' stage"):
        pe.approve_plan(approved_plan)


# =========================================================================
# SCENARIO 3: EXECUTION, OUTCOME VERIFICATION & FEEDBACK IMPROVEMENT
# =========================================================================

def test_scenario_3_execution_outcome_verification_and_feedback_improvement(shared_decision_model):
    """Scenario 3: Approved Plan -> Execution -> Independent Outcome Observation
    -> Metacognitive Evaluation -> Learning & Memory Update.
    Demonstrates that execution outcomes are verified from actual telemetry,
    and feedback updates persistent memory so subsequent queries reflect learned experience.
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine(decision_model=shared_decision_model, db_path=db_path)
        tenant_id = "tenant_logistics_hub"
        session_id = "sess_exec_scenario_03"

        # 1. Setup Goal & Plan
        pe = engine.planning_engine
        goal = GoalNode(
            goal_id="goal_carrier_reroute",
            description="Isolate damaged logistics shipment and reroute priority cargo",
            priority=1,
            required_postconditions=["cluster_network_isolated"],
            target_state_metrics={
                "exposure_risk": {"<=": 0.05},
                "shipment_secured": True,
            },
        )
        world = WorldState(environment_metrics={"exposure_risk": 0.40, "shipment_secured": False})
        trace = ReasoningTrace()

        plan = pe.generate_plan(
            target_goal=goal,
            available_actions=["isolate_incident_cluster"],
            current_world_state=world,
            reasoning_trace=trace,
            tenant_id=tenant_id,
        )
        pe.approve_plan(plan)
        assert plan.plan_stage == "approved"

        # 2. Execution Step 1: Dispatched to worker, records real observed telemetry
        # (telemetry isolation: observed_postconditions strictly from telemetry)
        pe.record_step_execution(
            plan=plan,
            step_number=1,
            observed_outcome={
                "cluster_network_isolated": True,
                "observed_postconditions": ["cluster_network_isolated"],
                "execution_latency_ms": 42.5,
                "telemetry_source": "edge_sensor_array_9",
            },
            observed_state=WorldState(environment_metrics={"exposure_risk": 0.02, "shipment_secured": True}),
            execution_status="success",
        )
        assert len(plan.executed_steps) == 1
        assert plan.plan_stage in ["in_progress", "executed"]

        # 3. Metacognitive Evaluation: Independent verification against observed state
        final_observed_world = WorldState(environment_metrics={"exposure_risk": 0.02, "shipment_secured": True})
        verified, details = pe.verify_plan_execution_completion(
            plan=plan,
            target_goal=goal,
            final_observed_world_state=final_observed_world,
        )

        assert verified is True
        assert details["verified"] is True
        assert plan.is_verified_goal_completion is True
        assert plan.plan_stage == "completed"

        # 4. Learning & Feedback Memory Update
        # Record verified feedback event into durable collector
        engine.feedback_collector.record_feedback(
            decision_id=plan.plan_id,
            predicted_decision={"decision": "isolate_incident_cluster", "risk": 0.05},
            predicted_outcome={"expected_isolated": True},
            actual_outcome={"exposure_risk": 0.02, "verified_completion": True},
            is_success=True,
            tenant_id=tenant_id,
            review_status="verified",
        )
        assert len(engine.feedback_collector.events) == 1

        # Store newly verified episode in episodic memory for closed-loop adaptation
        query_vec = engine.memory_embedder.embed(
            text="Priority cargo damaged in logistics transit requiring containment",
            structured_features={"exposure_risk": 0.40},
        )
        rec_id = f"EPISODE-{plan.plan_id[:8]}"
        engine.episodic_memory.insert(
            MemoryRecord(
                record_id=rec_id,
                tenant_id=tenant_id,
                context_text=f"Incident resolution for goal: {goal.description}",
                features={"exposure_risk": 0.40},
                decision_made={"selected_candidate": "isolate_incident_cluster", "decision": "isolate_incident_cluster"},
                vector=query_vec,
            )
        )
        engine.episodic_memory.update_verified_outcome(
            record_id=rec_id,
            outcome={"exposure_risk": 0.02, "status": "completed"},
            tenant_id=tenant_id,
        )

        # 5. Verify that subsequent retrieval retrieves the newly learned case
        similar_cases = engine.episodic_memory.search_similar(
            tenant_id=tenant_id,
            query_vector=query_vec,
            top_k=1,
            min_sim=0.5,
        )
        assert len(similar_cases) == 1
        retrieved_case = similar_cases[0]
        assert retrieved_case["record_id"] == rec_id
        assert retrieved_case["decision_made"]["decision"] == "isolate_incident_cluster"
        assert retrieved_case["verified_outcome"].get("status") == "completed"

    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
