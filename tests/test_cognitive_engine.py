"""Test Suite for Integrated Cognitive AI Engine & Capabilities.
Verifies Perception, Context, Working Memory, World Model, Reasoning, Planning,
and End-to-End Orchestration with CDM-1 as the Decision Core.
"""

import pytest
import os
import tempfile
import torch

from cdm.cognitive.state import (
    SharedCognitiveState,
    PerceptionResult,
    ContextFrame,
    WorkingMemoryState,
    GoalNode,
    WorldState,
    ReasoningTrace,
    Plan,
    CognitiveEngineResult,
)
from cdm.perception.perception_engine import PerceptionEngine
from cdm.context.context_engine import ContextEngine
from cdm.memory.working_memory import WorkingMemory
from cdm.world_model.world_model import WorldModel
from cdm.reasoning.reasoning_engine import ReasoningEngine
from cdm.planning.planning_engine import PlanningEngine
from cdm.cognitive.engine import CognitiveEngine
from cdm.inference.pipeline import CognitiveDecisionModel


def test_perception_intelligence():
    """Verifies entity extraction, operational signal calculation, and perception confidence."""
    pe = PerceptionEngine()
    text = "Customer ACC-48291 reports $250.00 unauthorized charge on their account. Urgent resolution needed immediately!"
    struct = {"monetary_value_usd": 250.0, "urgency_level": "critical", "sla_remaining_hours": 1.0}

    result = pe.process(text=text, structured_features=struct)

    assert isinstance(result, PerceptionResult)
    assert len(result.extracted_entities) >= 2  # account_id and monetary
    ent_types = [e.entity_type for e in result.extracted_entities]
    assert "account_id" in ent_types or "monetary" in ent_types

    # Signals
    assert result.signals["text_urgency"] > 0.3
    assert result.modality_confidence >= 0.8
    assert result.validated_tabular["monetary_value_usd"] == 250.0


def test_contextual_intelligence_and_contradiction():
    """Verifies dialogue tracking, temporal delta, and contradiction detection."""
    ce = ContextEngine()
    pe = PerceptionEngine()

    session_id = "test_session_001"
    turn1_text = "Hello, I am asking about my account."
    p1 = pe.process(turn1_text, {"interaction_count": 5, "urgency_level": "low"})
    ctx1 = ce.update_context(session_id, "tenant_a", p1, turn1_text, "How can I help you today?")

    assert ctx1.turn_index == 1
    assert len(ctx1.dialogue_history) == 2
    assert len(ctx1.detected_contradictions) == 0

    # Turn 2: User claims first time contacting despite interaction_count=5
    turn2_text = "This is my first time reporting any issue, I never contacted support before!"
    p2 = pe.process(turn2_text, {"interaction_count": 5, "urgency_level": "low"})
    ctx2 = ce.update_context(session_id, "tenant_a", p2, turn2_text)

    assert ctx2.turn_index == 2
    assert len(ctx2.detected_contradictions) >= 1
    assert "never contacted" in ctx2.detected_contradictions[0] or "first time" in ctx2.detected_contradictions[0]


def test_working_memory_lifecycle():
    """Verifies goal stack management, scratchpad eviction, and attention tracking."""
    wm = WorkingMemory(max_scratchpad_items=3, max_active_goals=3)

    # Goal hierarchy
    g1 = wm.push_goal("goal_main", "Resolve enterprise outage", priority=1)
    g2 = wm.push_goal("goal_sub1", "Quarantine infected node", priority=2, parent_goal_id="goal_main")

    assert len(wm.state.active_goals) == 2
    assert "goal_sub1" in g1.subgoal_ids

    # Scratchpad and eviction
    wm.write_scratchpad("k1", "v1")
    wm.write_scratchpad("k2", "v2")
    wm.write_scratchpad("k3", "v3")
    wm.write_scratchpad("k4", "v4")  # Should evict k1
    assert wm.read_scratchpad("k1") is None
    assert wm.read_scratchpad("k4") == "v4"

    # Complete goal
    assert wm.complete_goal("goal_sub1") is True
    wm.clear_transient()
    assert len(wm.state.active_goals) == 1
    assert wm.state.active_goals[0].goal_id == "goal_main"


def test_world_model_state_and_counterfactual_transitions():
    """Verifies explicit world state representation and counterfactual S_{t+1} transition simulation."""
    wm = WorldModel()

    wm.update_entity("cust_1", "customer", {"tier": "platinum"}, uncertainty=0.1)
    wm.update_entity("srv_1", "server", {"status": "degraded"}, uncertainty=0.2)
    wm.add_relation("cust_1", "impacts", "srv_1", confidence=0.9)
    wm.set_environment_metric("sla_remaining_hours", 4.0)

    state = wm.get_state()
    assert len(state.entities) == 2
    assert len(state.relations) == 1
    assert 0.0 < state.overall_state_uncertainty < 0.5

    # Simulate transition: 'escalate'
    sim_escalate = wm.simulate_transition("escalate_to_tier2")
    assert sim_escalate.environment_metrics["sla_remaining_hours"] < 4.0
    assert sim_escalate.environment_metrics["escalation_active"] == 1.0

    # Ground truth world state remains unmutated
    assert wm.get_state().environment_metrics["sla_remaining_hours"] == 4.0


def test_reasoning_intelligence_deduction_and_proof():
    """Verifies rule-based deduction, policy constraint evaluation, and proof step generation."""
    re = ReasoningEngine()
    wm = WorldModel()
    wm.set_environment_metric("monetary_value_usd", 1200.0)
    world_state = wm.get_state()

    ctx = ContextFrame(session_id="s1", tenant_id="t1")
    rules = [
        {
            "rule_id": "RULE-FIN-01",
            "description": "Auto-resolve prohibited for transactions > $500",
            "condition": {"monetary_value_usd": 500.0},
            "prohibited_action": "auto_resolve",
        }
    ]

    trace = re.reason(
        world_state=world_state,
        context=ctx,
        semantic_rules=rules,
        candidate_actions=["auto_resolve", "issue_refund", "escalate_to_tier2"],
        user_constraints=["prohibit issue_refund"],
    )

    assert isinstance(trace, ReasoningTrace)
    assert len(trace.steps) >= 3
    assert any("auto_resolve" in v for v in trace.violated_constraints)
    assert any("issue_refund" in v for v in trace.violated_constraints)
    assert trace.reasoning_confidence > 0.8


def test_planning_intelligence_generation_and_replanning():
    """Verifies multi-step plan generation, risk aggregation, and replanning triggers."""
    pe = PlanningEngine()
    wm = WorldModel()
    wm.set_environment_metric("sla_remaining_hours", 2.0)
    world_state = wm.get_state()

    goal = GoalNode(goal_id="g_incident", description="Mitigate payment gateway failure", priority=1)
    trace = ReasoningTrace(violated_constraints=["auto_resolve is forbidden"])

    plan = pe.generate_plan(
        target_goal=goal,
        available_actions=["auto_resolve", "escalate_to_tier2", "notify_customers"],
        current_world_state=world_state,
        reasoning_trace=trace,
    )

    assert isinstance(plan, Plan)
    assert plan.target_goal_id == "g_incident"
    assert len(plan.steps) == 2  # auto_resolve excluded due to violation
    assert plan.steps[0].action_name == "escalate_to_tier2"
    assert plan.cumulative_risk > 0.0

    # Test replanning trigger
    should_replan, reason = pe.should_replan(plan, world_state)
    assert should_replan is False

    # Simulate critical security deterioration triggering replan
    wm.set_environment_metric("system_exposure", 0.95)
    should_replan2, reason2 = pe.should_replan(plan, wm.get_state())
    assert should_replan2 is True
    assert "exposure" in reason2.lower()


def test_integrated_cognitive_engine_end_to_end():
    """Verifies the complete coordinated Cognitive AI Engine with real CDM-1 checkpoint decision core."""
    model_path = "experiments/release/cdm-1-general"
    if not os.path.exists(model_path):
        pytest.skip("Release checkpoint not found")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine.from_pretrained(model_name_or_path=model_path, db_path=db_path)

        res = engine.process(
            text="Customer reports item arrived cracked and damaged with photos attached.",
            structured_features={
                "monetary_value_usd": 85.0,
                "sla_remaining_hours": 12.0,
                "domain": "customer_support",
                "urgency_level": "medium",
                "interaction_count": 1.0,
            },
            candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
            prohibited_actions=["auto_resolve"],
            goals=["Resolve customer damage claim within SLA"],
            session_id="session_test_integration",
            tenant_id="tenant_retail",
        )

        assert isinstance(res, CognitiveEngineResult)
        assert res.decision in ["issue_refund", "escalate_to_tier2"]
        assert res.decision != "auto_resolve"  # Prohibited constraint strictly enforced
        assert 0.0 <= res.confidence <= 1.0
        assert 0.0 <= res.risk <= 1.0
        assert res.selected_plan is not None
        assert len(res.selected_plan.steps) >= 1
        assert res.reasoning_evidence is not None
        assert res.world_state_summary["entity_count"] >= 0
        assert res.latency_ms > 0.0

        # Verify feedback recording
        engine.record_feedback(
            decision_id=res.execution_id,
            predicted_result=res,
            actual_outcome={"resolution_success": True, "customer_satisfaction": 5},
            is_success=True,
        )
        assert len(engine.feedback_collector.events) == 1
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_integrated_cognitive_engine_metacognitive_abstention():
    """Verifies that high contradiction and low confidence trigger metacognitive abstention."""
    model_path = "experiments/release/cdm-1-general"
    if not os.path.exists(model_path):
        pytest.skip("Release checkpoint not found")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine.from_pretrained(model_name_or_path=model_path, db_path=db_path)

        # Pre-seed contradiction in context
        session_id = "session_abstain_test"
        res1 = engine.process(
            text="I already called three times today and nobody answered!",
            structured_features={"interaction_count": 0, "urgency_level": "low"},
            session_id=session_id,
        )

        # Second request with all candidates prohibited
        res2 = engine.process(
            text="Second ambiguous query with conflicting claims.",
            structured_features={"monetary_value_usd": 10000.0, "urgency_level": "critical"},
            candidates=["action_a", "action_b"],
            prohibited_actions=["action_a", "action_b"],
            session_id=session_id,
        )

        assert res2.abstain is True
        assert res2.decision == "ABSTAIN" or "prohibited" in (res2.abstention_reason or "").lower()
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
