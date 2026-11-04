"""Unit tests for Cognitive Memory Subsystems and Feedback Learning.
"""

import time
import pytest
from cdm.memory.episodic import EpisodicMemory, MemoryRecord
from cdm.memory.semantic import SemanticMemory, PolicyRule
from cdm.memory.procedural import ProceduralMemory, OperationalProcedure
from cdm.learning.feedback import FeedbackCollector
from cdm.learning.adaptation import AdaptationGatekeeper


def test_episodic_memory_vector_retrieval():
    mem = EpisodicMemory()
    rec1 = MemoryRecord(
        record_id="rec_1",
        context_text="Refund request under 30 days.",
        decision_made={"binary_decision": 1},
        vector=[1.0, 0.0, 0.0],
    )
    rec2 = MemoryRecord(
        record_id="rec_2",
        context_text="Password reset inquiry.",
        decision_made={"binary_decision": 0},
        vector=[0.0, 1.0, 0.0],
    )
    mem.insert(rec1)
    mem.insert(rec2)

    # Query with vector close to rec1
    matches = mem.retrieve_similar([0.9, 0.1, 0.0], top_k=1)
    assert len(matches) == 1
    assert matches[0]["record_id"] == "rec_1"

    # Outcome updating
    updated = mem.update_verified_outcome("rec_1", {"resolved_in_hours": 0.5})
    assert updated


def test_semantic_memory_rules():
    sem = SemanticMemory()
    sem.register_rule(
        PolicyRule(
            rule_id="RULE-01",
            domain="customer_support",
            condition_description="Cash refund past 30 days is prohibited",
            permitted_actions=["store_credit"],
            prohibited_actions=["cash_refund"],
        )
    )

    prohibited = sem.get_prohibited_actions("customer_support")
    assert "cash_refund" in prohibited


def test_procedural_memory():
    proc_mem = ProceduralMemory()
    proc_mem.register_procedure(
        OperationalProcedure(
            procedure_id="SOP-SECURITY-ESCALATE",
            domain="communication",
            trigger_condition="Active server penetration",
            action_sequence=["quarantine_session", "page_oncall", "isolate_vpc"],
            required_validations=["2fa_manager_auth"],
        )
    )

    p = proc_mem.get_procedure("SOP-SECURITY-ESCALATE")
    assert p is not None
    assert len(p.action_sequence) == 3


def test_feedback_collector():
    collector = FeedbackCollector()
    ev = collector.record_feedback(
        decision_id="dec_99",
        predicted_decision={"binary_prediction": 1},
        predicted_outcome={"prob": 0.8},
        actual_outcome={"binary_target": 0},
        is_success=False,
    )

    assert ev.discrepancy_score == 1.0
    failures = collector.get_failure_cases()
    assert len(failures) == 1
    assert failures[0].decision_id == "dec_99"


def test_adaptation_gatekeeper():
    gate = AdaptationGatekeeper()
    # Test gate evaluation
    base_metrics = {"binary_decision": {"accuracy": 0.95}}
    good_cand = {"binary_decision": {"accuracy": 0.96}}
    bad_cand = {"binary_decision": {"accuracy": 0.88}}

    assert gate.evaluate_benchmark_gate(good_cand, base_metrics) is True
    assert gate.evaluate_benchmark_gate(bad_cand, base_metrics) is False
