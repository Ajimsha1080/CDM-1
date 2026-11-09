"""CDM-1 Comprehensive Cognitive Capabilities Validation Suite.
Validates all 9 core cognitive capabilities and the 7 end-to-end operational scenarios:
1. Causal Analysis (Mechanism vs Correlation, Confounders, Counterfactuals, Temporal Ordering)
2. Predictive Intelligence (Outcome distribution, Decision influence, Multi-step horizon)
3. Neural Decision-Making (Candidate scoring, Argument isolation, Abstention)
4. Goal and Constraint Management (Hierarchy, Conflicting goals, Strict enforcement)
5. Sequential Planning (Pre/post conditions, Dynamic replanning on divergence)
6. Autonomous Execution (Sandbox boundary containment, Tool validation, Audit trail)
7. Independent Outcome Verification (4-state lifecycle, Unverified != Verified)
8. Metacognition (Miscalibration intervention, Recurring failure detection)
9. Experience-Driven Adaptation (Memory consolidation, Rejection of unverified data, Tenant isolation)

Scenarios:
- Scenario A: Causal decision
- Scenario B: Predictive planning
- Scenario C: Multi-step autonomous task
- Scenario D: Failure recovery
- Scenario E: Metacognitive correction
- Scenario F: Experience-driven adaptation
- Scenario G: Safety and abstention
"""

import os
import json
import time
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, List

from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import (
    WorldState,
    ReasoningTrace,
    Plan,
    PlanStep,
    GoalNode,
)
from cdm.causal.causal_engine import (
    CausalEngine,
    AssociationType,
    CounterfactualQuery,
)
from cdm.world_model.world_model import WorldModel
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult
from cdm.planning.planning_engine import PlanningEngine
from cdm.memory.working_memory import WorkingMemory
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory
from cdm.gate.policy_gate import PolicyGate, ProposedAction, GateDecisionType
from cdm.execution.tools.file_tool import FileWriteTool, FileReadTool
from cdm.execution.tools.engineering_tools import FileCreateTool, EnhancedFileReadTool
from cdm.execution.verifier import OutcomeVerifier, VerificationSpec, ExecutionOutcomeState, VerificationStatus
from cdm.learning.feedback import FeedbackCollector
from cdm.learning.adaptation import AdaptationGatekeeper
from cdm.agent.loop import AgentControlLoop, AgentStore, AgentRunState, AgentActionProposal
from cdm.llm.base import BaseLLMProvider


class MockScriptedLLM(BaseLLMProvider):
    """Deterministic scriptable LLM provider for agent tests."""
    def __init__(self, responses: List[Any]):
        super().__init__()
        self.responses = list(responses)
        self.call_count = 0

    def _call_model(self, messages, temperature=0.0):
        self.call_count += 1
        idx = min(self.call_count - 1, len(self.responses) - 1)
        resp = self.responses[idx]
        if isinstance(resp, str):
            return resp, 10, 10
        elif isinstance(resp, dict):
            return json.dumps(resp), 10, 10
        return "{}", 0, 0


@pytest.fixture(scope="module")
def decision_core():
    return CognitiveDecisionModel.from_pretrained(allow_fallback=True)


# =========================================================================
# Capability 1: Causal Analysis
# =========================================================================

def test_causal_mechanism_vs_correlation():
    """Validates that CausalEngine distinguishes direct causal mechanisms from correlations."""
    engine = CausalEngine()

    hyp_mech = engine.evaluate_causality("issue_refund", "refund_issued")
    assert hyp_mech.association_type == AssociationType.CAUSAL_MECHANISM
    assert hyp_mech.confidence >= 0.95
    assert hyp_mech.status == "intervention_verified"

    hyp_corr = engine.evaluate_causality("high_monetary_value", "distress_level")
    assert hyp_corr.association_type == AssociationType.CORRELATION
    assert hyp_corr.confidence < 0.90
    assert hyp_corr.status == "supported"


def test_causal_confounding_isolation():
    """Validates identifying common causes (confounders) and isolating spurious correlations."""
    engine = CausalEngine()

    engine.add_causal_relation("server_overload", "high_latency", AssociationType.CAUSAL_MECHANISM, 0.95, "Resource saturation delays requests", True)
    engine.add_causal_relation("server_overload", "500_errors", AssociationType.CAUSAL_MECHANISM, 0.90, "Resource saturation causes worker crash", True)
    engine.add_causal_relation("high_latency", "500_errors", AssociationType.CORRELATION, 0.70, "Observed co-occurrence without direct mechanism", False)

    confounders = engine.detect_confounders("high_latency", "500_errors")
    assert "server_overload" in confounders

    isolation = engine.isolate_confounder("high_latency", "500_errors", "server_overload")
    assert isolation["is_spurious_or_confounded"] is True
    assert isolation["has_direct_causal_mechanism"] is False
    assert isolation["adjusted_causal_effect"] == 0.0


def test_causal_counterfactual_justified():
    """Validates counterfactual claims are only justified when backed by verified causal mechanisms."""
    engine = CausalEngine()

    q_justified = CounterfactualQuery(
        action_taken="auto_resolve",
        observed_outcome={"status": "resolved"},
        hypothetical_action="issue_refund",
        context_state={"dispute_active": True},
    )
    res_justified = engine.evaluate_counterfactual(q_justified)
    assert res_justified.justified is True
    assert res_justified.confidence >= 0.8
    assert "refund_issued" in res_justified.predicted_outcome
    assert len(res_justified.causal_pathway) > 0

    q_unjustified = CounterfactualQuery(
        action_taken="auto_resolve",
        observed_outcome={"status": "resolved"},
        hypothetical_action="unsupported_quantum_action",
    )
    res_unjustified = engine.evaluate_counterfactual(q_unjustified)
    assert res_unjustified.justified is False
    assert res_unjustified.confidence == 0.0
    assert res_unjustified.limitation_note is not None


def test_causal_temporal_ordering():
    """Validates that cause must strictly precede effect in time; backward ordering is rejected."""
    engine = CausalEngine()

    engine.temporal_history.append(("event_trigger", "triggered", 100.0))
    engine.temporal_history.append(("event_consequence", "observed", 105.0))

    valid, msg = engine.validate_temporal_ordering("event_trigger", "event_consequence")
    assert valid is True

    invalid, reason = engine.validate_temporal_ordering("event_consequence", "event_trigger")
    assert invalid is False
    assert "Temporal ordering violated" in reason

    hyp_viol = engine.evaluate_causality("event_consequence", "event_trigger")
    assert hyp_viol.status == "refuted"
    assert any("Temporal violation" in ce for ce in hyp_viol.counter_evidence)


# =========================================================================
# Capability 2: Predictive Intelligence
# =========================================================================

def test_predictive_outcome_distribution(decision_core):
    """Validates predictive intelligence generates calibrated outcome distributions and uncertainty."""
    res = decision_core.decide(
        text="Customer reports complete service outage and unauthorized charges",
        structured_features={"monetary_value_usd": 850.0, "urgency_level": "critical", "sentiment_score": -0.8},
        task="binary",
    )
    assert res.predicted_outcome is not None
    assert 0.0 <= res.predicted_outcome["binary_outcome_probability"] <= 1.0
    assert res.predicted_outcome["predicted_resolution_time"] >= 0.0
    assert 0.0 <= res.risk <= 1.0
    assert res.priority in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    assert res.provenance is not None


def test_predictive_influences_decision(decision_core):
    """Validates that high predicted risk penalizes action selection in cognitive engine."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine(db_path=db_path, decision_core=decision_core)
        res = engine.process(
            text="CRITICAL SECURITY BREACH: System compromised, unauthorized root access detected!",
            structured_features={"monetary_value_usd": 5000.0, "urgency_level": "critical", "sla_remaining_hours": 0.5, "sentiment_score": -0.9},
            candidates=["auto_resolve", "isolate_cluster"],
            task="choice",
        )
        assert res.risk >= 0.20
        assert res.decision == "isolate_cluster"
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_predictive_multi_step_horizon():
    """Validates multi-step prospective horizon simulation in planning."""
    wm = WorldModel()
    wm.set_environment_metric("system_exposure", 0.9)
    wm.set_environment_metric("monetary_risk", 100.0)

    pe = PlanningEngine(world_model=wm)
    target_goal = GoalNode(goal_id="g_horizon", description="Reduce exposure and resolve dispute", priority=1)

    plan = pe.generate_plan(
        target_goal=target_goal,
        available_actions=["isolate_cluster", "issue_refund"],
        current_world_state=wm.get_state(),
        reasoning_trace=ReasoningTrace(),
    )
    assert plan is not None
    assert len(plan.steps) >= 1
    assert plan.cumulative_risk is not None
    assert plan.steps[0].expected_outcome is not None


# =========================================================================
# Capability 3: Neural Decision-Making
# =========================================================================

def test_neural_decision_candidate_scoring(decision_core):
    """Validates candidate scoring and ranking with prohibited candidate filtering."""
    cands = ["auto_resolve", "issue_refund", "escalate_to_tier2"]
    proh = ["auto_resolve"]

    res = decision_core.decide(
        text="Customer requests refund for incorrect charges",
        task="choice",
        candidates=cands,
        prohibited_actions=proh,
    )
    assert res.decision != "auto_resolve"
    assert res.candidate_scores["auto_resolve"] == 0.0
    assert res.selected_candidate in ["issue_refund", "escalate_to_tier2"]


def test_neural_decision_argument_isolation(decision_core):
    """Validates that candidate tools receive independent arguments and never reuse parameters from other tools."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_db = os.path.join(tmp_dir, "agent.db")
        store = AgentStore(store_db)

        class MockToolA:
            description = "Tool A"
            def execute(self, args, tenant_id=None):
                from cdm.agent.loop import ToolResult
                return ToolResult(success=True, output=f"Tool A ran with {args}", tool_name="tool_a")

        class MockToolB:
            description = "Tool B"
            def execute(self, args, tenant_id=None):
                from cdm.agent.loop import ToolResult
                return ToolResult(success=True, output=f"Tool B ran with {args}", tool_name="tool_b")

        tools = {"tool_a": MockToolA(), "tool_b": MockToolB()}
        llm = MockScriptedLLM([{"tool": "tool_a", "arguments": {"tool_a_specific_arg": "unique_val"}, "reason": "Try tool A"}])

        engine = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            tools=tools,
            store=store,
            sandbox_root=tmp_dir,
        )

        run = loop.create_run(
            goal="Test argument isolation",
            task_description="Execute safely",
            tenant_id="tenant_arg_test",
        )
        # Active plan has step 1 'tool_b'
        run.active_plan = {
            "plan_id": "p_isolate",
            "target_goal_id": "g_isolate",
            "steps": [{"step_number": 1, "action_name": "tool_b", "parameters": {}}],
        }
        run.current_plan_step = 0
        store.save_run(run)

        run = loop.step(run.run_id)

        assert len(run.history) >= 1
        executed_step = run.history[-1]
        assert executed_step["tool"] == "tool_b"
        # Tool B must NOT inherit Tool A's arguments
        assert "tool_a_specific_arg" not in executed_step["arguments"]
        assert executed_step["arguments"] == {}


# =========================================================================
# Capability 4: Goal and Constraint Management
# =========================================================================

def test_goal_decomposition_hierarchy():
    """Validates goal decomposition and parent-subgoal hierarchy in WorkingMemory."""
    wm = WorkingMemory(max_active_goals=5)
    g_root = wm.push_goal("g_root", "Resolve incident ticket ACC-100", priority=1)
    g_sub1 = wm.push_goal("g_sub1", "Verify customer identity", priority=2, parent_goal_id="g_root")
    g_sub2 = wm.push_goal("g_sub2", "Check transaction logs", priority=3, parent_goal_id="g_root")

    assert len(wm.state.active_goals) == 3
    assert wm.state.active_goals[1].parent_goal_id == "g_root"
    assert wm.state.active_goals[2].parent_goal_id == "g_root"

    wm.complete_goal("g_sub1")
    wm.clear_transient()
    assert len(wm.state.active_goals) == 2
    assert wm.state.active_goals[0].goal_id == "g_root"


def test_constraint_strict_enforcement():
    """Validates constraints strictly prevent execution regardless of utility or neural score."""
    gate = PolicyGate()
    action = ProposedAction(tool="issue_refund", args={"amount": 500.0}, reason="High reward action")
    world_state = WorldState()

    res = gate.evaluate(
        action=action,
        tenant_id="tenant_strict",
        world_state=world_state,
        prohibited_actions=["issue_refund"],
    )
    assert res.decision == GateDecisionType.BLOCK
    assert "prohibited" in res.reason.lower()


# =========================================================================
# Capability 5: Sequential Planning
# =========================================================================

def test_sequential_plan_pre_post_conditions():
    """Validates that sequential plans check preconditions and establish postconditions."""
    pe = PlanningEngine()
    wm = WorldState()
    wm.observed_facts["tier1_triage_completed"] = True
    wm.environment_metrics["monetary_value_usd"] = 2000.0  # Exceeds max refund limit

    preconds_esc, met_esc = pe._evaluate_preconditions("escalate_to_tier2", wm, set())
    assert met_esc is True
    assert "tier1_triage_completed" in preconds_esc

    preconds_ref, met_ref = pe._evaluate_preconditions("issue_refund", wm, set())
    assert met_ref is False


def test_dynamic_replanning_on_state_divergence():
    """Validates dynamic replanning is triggered when world state constraints invalidate active plan."""
    pe = PlanningEngine()
    plan = Plan(
        plan_id="p_test",
        target_goal_id="g_incident",
        goal="Resolve incident",
        steps=[PlanStep(step_number=1, action_name="auto_resolve", description="Close inquiry")],
    )

    ws_normal = WorldState()
    replan_normal, _ = pe.should_replan(plan, ws_normal)
    assert replan_normal is False

    ws_diverged = WorldState()
    ws_diverged.active_constraints.append("ProhibitedAction: auto_resolve")
    replan_diverged, reason = pe.should_replan(plan, ws_diverged)
    assert replan_diverged is True
    assert "prohibited by updated world state" in reason


# =========================================================================
# Capability 6: Autonomous Execution
# =========================================================================

def test_tool_execution_sandbox_boundary():
    """Validates tools strictly enforce filesystem sandbox boundaries and reject directory traversal."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox_path = os.path.join(tmp_dir, "sandbox")
        os.makedirs(sandbox_path, exist_ok=True)

        tool = FileCreateTool(sandbox_root=sandbox_path)

        # Attempt directory traversal out of sandbox
        res = tool.execute({"path": "../outside.txt", "content": "malicious content"})
        assert res.success is False
        assert "traversal" in res.error.lower() or "outside" in res.error.lower()

        # Legitimate file inside sandbox
        res_legit = tool.execute({"path": "valid.txt", "content": "legitimate file inside"})
        assert res_legit.success is True
        assert os.path.exists(os.path.join(sandbox_path, "valid.txt"))


# =========================================================================
# Capability 7: Independent Outcome Verification
# =========================================================================

def test_four_state_verification():
    """Validates the 4 distinct execution states: attempted -> executed -> action result verified -> task goal verified."""
    s1 = OutcomeVerifier.classify_execution_state(attempted=True, executed=False)
    assert s1 == ExecutionOutcomeState.ATTEMPTED

    s2 = OutcomeVerifier.classify_execution_state(attempted=True, executed=True, action_verified=False)
    assert s2 == ExecutionOutcomeState.EXECUTED

    s3 = OutcomeVerifier.classify_execution_state(attempted=True, executed=True, action_verified=True, goal_verified=False)
    assert s3 == ExecutionOutcomeState.ACTION_RESULT_VERIFIED

    s4 = OutcomeVerifier.classify_execution_state(attempted=True, executed=True, action_verified=True, goal_verified=True)
    assert s4 == ExecutionOutcomeState.TASK_GOAL_VERIFIED


def test_unverified_never_reported_as_verified():
    """Validates that unverified outcomes are rejected and never reported as verified_complete."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        spec = VerificationSpec(check_type="file_exists", target="non_existent_output.log")
        res = OutcomeVerifier.verify(spec=spec, sandbox_root=tmp_dir)

        assert res.verified is False
        assert res.status == VerificationStatus.FAILED
        assert res.status != VerificationStatus.VERIFIED_COMPLETE


# =========================================================================
# Capability 8: Metacognition
# =========================================================================

def test_metacognitive_miscalibration_intervention(decision_core):
    """Validates that detected context contradictions trigger metacognitive review."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine(db_path=db_path, decision_core=decision_core)
        # Turn 1: Normal statement
        p1 = engine.perception_engine.process("I am inquiring about order delivery.")
        ctx1 = engine.context_engine.update_context("sess_meta", "tenant_meta", p1, "I am inquiring about order delivery.")

        # Turn 2: Contradictory interaction history
        p2 = engine.perception_engine.process("I have never contacted support before, this is my first time reporting anything!", structured_features={"interaction_count": 5})
        ctx2 = engine.context_engine.update_context("sess_meta", "tenant_meta", p2, "I have never contacted support before, this is my first time reporting anything!")

        assert len(ctx2.detected_contradictions) >= 1

        reasoning = engine.reasoning_engine.reason(
            world_state=WorldState(),
            context=ctx2,
            semantic_rules=[],
            candidate_actions=["auto_resolve", "escalate_to_tier2"],
        )
        assert len(reasoning.contradictions_found) >= 1
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_metacognitive_recurring_failure_detection(decision_core):
    """Validates that 3 consecutive action failures trigger metacognitive intervention and suspend execution."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AgentStore(os.path.join(tmp_dir, "agent.db"))
        engine = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=MockScriptedLLM([]),
            cognitive_engine=engine,
            tools={},
            store=store,
            sandbox_root=tmp_dir,
        )

        run = loop.create_run(
            goal="Test recurring failure",
            task_description="Execute failing steps",
            tenant_id="tenant_meta_fail",
        )
        # Record 3 consecutive failed steps in history
        for i in range(1, 4):
            run.history.append({"step": i, "tool": "fail_tool", "success": False, "error": "Simulated error"})
            run.step_count += 1
        store.save_run(run)

        res_run = loop.step(run.run_id)
        assert res_run.status == "waiting_for_approval"
        assert "Metacognitive intervention" in res_run.pending_reason
        assert res_run.history[-1]["tool"] == "metacognitive_intervention"


# =========================================================================
# Capability 9: Experience-Driven Adaptation
# =========================================================================

def test_experience_consolidation_tenant_isolated():
    """Validates that recorded experiences are strictly tenant-isolated and cannot leak across boundaries."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteMemoryStorage(db_path=db_path)
        ep_mem = EpisodicMemory(storage=storage)

        # Tenant A records experience
        ep_mem.record_episode(
            session_id="sess_A",
            goal="Resolve billing issue",
            action_taken="issue_refund({'amount': 25})",
            observed_outcome="Customer satisfied, refund processed",
            reward_signal=1.0,
            tenant_id="tenant_ALPHA",
        )

        # Tenant B searches memory
        results_b = ep_mem.search_episodes(query_text="billing refund", tenant_id="tenant_BETA")
        assert len(results_b) == 0

        # Tenant A searches memory
        results_a = ep_mem.search_episodes(query_text="billing refund", tenant_id="tenant_ALPHA")
        assert len(results_a) == 1
        assert "issue_refund" in results_a[0].action_taken
        assert results_a[0].reward_signal == 1.0
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_unverified_experience_rejected_from_adaptation():
    """Validates that unverified or failed experiences are rejected by AdaptationGatekeeper."""
    gatekeeper = AdaptationGatekeeper()
    collector = FeedbackCollector()

    collector.record_feedback(
        decision_id="dec_unverified_01",
        predicted_decision={"tool": "write_file"},
        predicted_outcome={"expected": "created"},
        actual_outcome={"output": "File written without verification", "verified": False},
        is_success=False,
        tenant_id="tenant_adapt_test",
        review_status="rejected_for_training",
    )

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        out_jsonl = tmp.name

    try:
        staged_count = gatekeeper.stage_feedback_dataset(collector.events, output_path=out_jsonl)
        assert staged_count == 0  # Unverified/rejected feedback is excluded from model adaptation!
    finally:
        if os.path.exists(out_jsonl):
            os.unlink(out_jsonl)


# =========================================================================
# Cross-Capability Scenarios A through G
# =========================================================================

def test_scenario_a_causal_decision():
    """Scenario A: Causal Decision
    Demonstrates correlation vs causation distinction and causal mechanism guiding action selection.
    """
    engine = CausalEngine()
    metrics = {"refund_issued": 0.0, "customer_distress": 0.8}
    consequences = engine.analyze_candidate_consequences(
        candidate_actions=["issue_refund", "auto_resolve"],
        current_metrics=metrics,
    )
    assert consequences["issue_refund"]["has_verified_causal_mechanism"] is True
    direct_effects = consequences["issue_refund"]["direct_causal_effects"]
    assert any(e["target_variable"] == "refund_issued" for e in direct_effects)


def test_scenario_b_predictive_planning():
    """Scenario B: Predictive Planning
    Predictive model output shapes multi-step plan steps and contingency selection.
    """
    wm = WorldModel()
    wm.set_environment_metric("active_incident", 1.0)
    pe = PlanningEngine(world_model=wm)
    target_goal = GoalNode(goal_id="g_incident", description="Contain incident and diagnostic telemetry", priority=1)

    plan = pe.generate_plan(
        target_goal=target_goal,
        available_actions=["isolate_cluster", "safe_diagnostics"],
        current_world_state=wm.get_state(),
        reasoning_trace=ReasoningTrace(),
        predicted_outcome={"expected_risk": 0.85, "candidate_scores": {"isolate_cluster": 0.9}},
    )
    assert plan is not None
    assert plan.steps[0].action_name == "isolate_cluster"


def test_scenario_c_multistep_autonomous_task(decision_core):
    """Scenario C: Multi-Step Autonomous Task
    Executes full loop: perception -> plan -> tool execution in sandbox -> verifier -> completion.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox = os.path.join(tmp_dir, "sandbox")
        os.makedirs(sandbox, exist_ok=True)
        store = AgentStore(os.path.join(tmp_dir, "agent.db"))

        tools = {
            "create_file": FileCreateTool(sandbox_root=sandbox),
            "read_file": EnhancedFileReadTool(sandbox_root=sandbox),
        }
        llm = MockScriptedLLM([
            {"tool": "create_file", "arguments": {"path": "result.txt", "content": "Task completed successfully"}, "reason": "Write output"},
            {"tool": "complete_task", "arguments": {}, "reason": "Declare task goal finished", "claim_goal_met": True},
        ])
        cog = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=cog,
            tools=tools,
            store=store,
            sandbox_root=sandbox,
        )

        run = loop.create_run(
            goal="Create result.txt verifying task completion",
            task_description="Multi-step creation and completion",
            tenant_id="tenant_scenario_c",
            verification_spec={"check_type": "file_exists", "target": "result.txt"},
        )
        res_run = loop.run_until_pause_or_finish(run.run_id, max_steps=5)

        assert res_run.status == "completed"
        assert res_run.verification_status == "verified_complete"
        assert os.path.exists(os.path.join(sandbox, "result.txt"))


def test_scenario_d_failure_recovery(decision_core):
    """Scenario D: Failure Recovery
    Tool failure triggers dynamic replanning and recovery to alternative action.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox = os.path.join(tmp_dir, "sandbox")
        os.makedirs(sandbox, exist_ok=True)
        store = AgentStore(os.path.join(tmp_dir, "agent.db"))

        class FailingTool:
            description = "Tool that fails"
            def execute(self, args, tenant_id=None):
                from cdm.agent.loop import ToolResult
                return ToolResult(success=False, error="Connection timeout on endpoint", tool_name="fail_net")

        class RecoveryTool:
            description = "Tool that succeeds"
            def execute(self, args, tenant_id=None):
                from cdm.agent.loop import ToolResult
                return ToolResult(success=True, output="Fallback succeeded", tool_name="recover_tool")

        tools = {"fail_net": FailingTool(), "recover_tool": RecoveryTool()}
        llm = MockScriptedLLM([
            {"tool": "fail_net", "arguments": {}, "reason": "Initial attempt"},
            {"tool": "recover_tool", "arguments": {}, "reason": "Recovery attempt"},
        ])
        cog = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=cog,
            tools=tools,
            store=store,
            sandbox_root=sandbox,
        )

        run = loop.create_run(goal="Recover from failure", task_description="Test recovery", tenant_id="tenant_scen_d")
        run = loop.step(run.run_id)
        assert run.history[-1]["success"] is False

        run = loop.step(run.run_id)
        assert run.history[-1]["success"] is True
        assert run.history[-1]["tool"] == "recover_tool"


def test_scenario_e_metacognitive_correction(decision_core):
    """Scenario E: Metacognitive Correction
    Detects systemic repeated failures and intervenes before uncontrolled execution.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AgentStore(os.path.join(tmp_dir, "agent.db"))
        cog = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=MockScriptedLLM([{"tool": "broken_tool", "arguments": {}, "reason": "Attempt"}]),
            cognitive_engine=cog,
            tools={},
            store=store,
            sandbox_root=tmp_dir,
        )

        run = loop.create_run(goal="Metacognitive test", task_description="Catch loop", tenant_id="tenant_scen_e")
        for i in range(1, 4):
            run.history.append({"step": i, "tool": "broken_tool", "arguments": {}, "success": False, "error": "Crash"})
            run.step_count += 1
        store.save_run(run)

        res_run = loop.step(run.run_id)
        assert res_run.status == "waiting_for_approval"
        assert "Systemic failure detected" in res_run.pending_reason


def test_scenario_f_experience_driven_adaptation():
    """Scenario F: Experience-Driven Adaptation
    Verified task completion consolidates experience into episodic memory; subsequent query retrieves it.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "cog.db")
        cog = CognitiveEngine(db_path=db_path)

        # 1. Record verified task episode
        cog.episodic_memory.record_episode(
            session_id="sess_f_1",
            goal="Configure production firewall",
            action_taken="apply_firewall_policy({'strict': True})",
            observed_outcome="Policy applied, port 22 closed to external",
            reward_signal=1.0,
            tenant_id="tenant_scen_f",
        )

        # 2. Subsequent task query retrieves verified experience
        recalled = cog.episodic_memory.search_episodes(
            query_text="Configure firewall ports",
            limit=1,
            tenant_id="tenant_scen_f",
        )
        assert len(recalled) == 1
        assert "apply_firewall_policy" in recalled[0].action_taken
        assert recalled[0].reward_signal == 1.0


def test_scenario_g_safety_and_abstention(decision_core):
    """Scenario G: Safety and Abstention
    Policy constraint violation triggers immediate block and safe abstention.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        sandbox = os.path.join(tmp_dir, "sandbox")
        os.makedirs(sandbox, exist_ok=True)
        store = AgentStore(os.path.join(tmp_dir, "agent.db"))

        class DeleteAllTool:
            description = "Destructive tool"
            def execute(self, args, tenant_id=None):
                from cdm.agent.loop import ToolResult
                return ToolResult(success=True, output="Deleted", tool_name="delete_all")

        tools = {"delete_all": DeleteAllTool()}
        llm = MockScriptedLLM([{"tool": "delete_all", "arguments": {}, "reason": "Clear disk"}])
        cog = CognitiveEngine(db_path=os.path.join(tmp_dir, "cog.db"), decision_core=decision_core)
        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=cog,
            tools=tools,
            store=store,
            sandbox_root=sandbox,
        )

        run = loop.create_run(
            goal="Clean disk",
            task_description="Dangerous task",
            tenant_id="tenant_scen_g",
            prohibited_actions=["delete_all"],
        )
        res_run = loop.step(run.run_id)

        assert len(res_run.history) >= 1
        step_rec = res_run.history[-1]
        assert step_rec["gate_decision"] in ("block", "abstain")
        assert step_rec["success"] is False
        assert "prohibit" in step_rec["error"].lower() or "blocked" in step_rec["error"].lower()
