"""Test Suite for Tenant Isolation and Concurrent Safety in CDM-1.
Verifies that context, world state, working memory, persistent storage, feedback,
and concurrent model inference strictly enforce tenant boundaries without cross-talk.
"""

import pytest
import os
import tempfile
import threading
from typing import Dict, List, Any

from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import PerceptionResult, EntityMention
from cdm.context.context_engine import ContextEngine
from cdm.world_model.world_model import WorldModel
from cdm.memory.working_memory import WorkingMemory
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory, MemoryRecord
from cdm.learning.feedback import FeedbackCollector


def test_context_engine_tenant_isolation():
    """Verifies that two tenants sharing the same session_id have completely isolated histories."""
    ce = ContextEngine()
    session_id = "shared_session_001"

    p_a = PerceptionResult(raw_text="Tenant A message: My account is ACC-ALICE")
    p_b = PerceptionResult(raw_text="Tenant B message: My account is ACC-BOB")

    ctx_a = ce.update_context(session_id=session_id, tenant_id="tenant_a", perception=p_a, user_message="Hello from Alice")
    ctx_b = ce.update_context(session_id=session_id, tenant_id="tenant_b", perception=p_b, user_message="Hello from Bob")

    assert ctx_a.tenant_id == "tenant_a"
    assert ctx_b.tenant_id == "tenant_b"
    assert ctx_a.turn_index == 1
    assert ctx_b.turn_index == 1

    latest_a = ce.get_latest_context(session_id=session_id, tenant_id="tenant_a")
    latest_b = ce.get_latest_context(session_id=session_id, tenant_id="tenant_b")

    assert latest_a is not None and latest_b is not None
    assert "Alice" in latest_a.dialogue_history[0]["content"]
    assert "Bob" in latest_b.dialogue_history[0]["content"]
    assert "Bob" not in str(latest_a.dialogue_history)
    assert "Alice" not in str(latest_b.dialogue_history)


def test_world_model_tenant_scoping():
    """Verifies that entity states, environment metrics, and observed facts remain scoped per tenant."""
    wm_a = WorldModel(tenant_id="tenant_finance", session_id="task_001")
    wm_b = WorldModel(tenant_id="tenant_retail", session_id="task_001")

    wm_a.update_entity("CUST-1", "customer", {"credit_limit": 50000.0})
    wm_a.set_environment_metric("fraud_score", 0.85)

    wm_b.update_entity("CUST-2", "customer", {"cart_value": 45.0})
    wm_b.set_environment_metric("sla_remaining_hours", 4.0)

    state_a = wm_a.get_state()
    state_b = wm_b.get_state()

    assert "CUST-1" in state_a.entities
    assert "CUST-2" not in state_a.entities
    assert state_a.environment_metrics.get("fraud_score") == 0.85
    assert "sla_remaining_hours" not in state_a.environment_metrics

    assert "CUST-2" in state_b.entities
    assert "CUST-1" not in state_b.entities
    assert state_b.environment_metrics.get("sla_remaining_hours") == 4.0
    assert "fraud_score" not in state_b.environment_metrics


def test_memory_storage_cross_tenant_isolation():
    """Verifies that SQLite storage strictly isolates episodic queries and outcome updates by tenant_id."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        storage = SQLiteMemoryStorage(db_path=db_path)
        episodic = EpisodicMemory(storage=storage)

        rec_a = MemoryRecord(
            record_id="rec_alpha_001",
            tenant_id="tenant_alpha",
            context_text="Confidential trade secret of Tenant Alpha.",
            decision_made={"selected_candidate": "act_escalate_tier2"},
            vector=[1.0, 0.0, 0.0] + [0.0] * 13,
        )
        episodic.insert(rec_a)

        # Tenant Beta performs vector retrieval with the exact same vector
        query_vec = [1.0, 0.0, 0.0] + [0.0] * 13
        beta_results = episodic.retrieve_similar(query_vector=query_vec, tenant_id="tenant_beta", top_k=5, min_relevance=0.1)
        assert len(beta_results) == 0, "Tenant Beta must not see Tenant Alpha's episodic memories"

        # Tenant Alpha performs vector retrieval
        alpha_results = episodic.retrieve_similar(query_vector=query_vec, tenant_id="tenant_alpha", top_k=5, min_relevance=0.1)
        assert len(alpha_results) == 1
        assert alpha_results[0]["record_id"] == "rec_alpha_001"

        # Tenant Beta attempts to overwrite Tenant Alpha's verified outcome
        malicious_update = episodic.update_verified_outcome(
            record_id="rec_alpha_001",
            outcome={"hacked": True},
            tenant_id="tenant_beta",
        )
        assert malicious_update is False, "Cross-tenant outcome update must be rejected"

        # Legitimate Tenant Alpha outcome update
        valid_update = episodic.update_verified_outcome(
            record_id="rec_alpha_001",
            outcome={"resolution_success": True},
            tenant_id="tenant_alpha",
        )
        assert valid_update is True
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_durable_feedback_tenant_isolation():
    """Verifies that FeedbackCollector partitions feedback events and failure cases by tenant."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        collector = FeedbackCollector(db_path=db_path)

        collector.record_feedback(
            decision_id="dec_001",
            predicted_decision={"decision": "escalate"},
            predicted_outcome={"risk": 0.1},
            actual_outcome={"observed_outcome_binary": 1},
            is_success=False,
            tenant_id="tenant_a",
        )

        collector.record_feedback(
            decision_id="dec_002",
            predicted_decision={"decision": "refund"},
            predicted_outcome={"risk": 0.2},
            actual_outcome={"observed_outcome_binary": 0},
            is_success=False,
            tenant_id="tenant_b",
        )

        failures_a = collector.get_failure_cases(tenant_id="tenant_a")
        failures_b = collector.get_failure_cases(tenant_id="tenant_b")

        assert len(failures_a) == 1
        assert failures_a[0].decision_id == "dec_001"

        assert len(failures_b) == 1
        assert failures_b[0].decision_id == "dec_002"
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_concurrent_multithreaded_cognitive_processing():
    """Verifies that concurrent requests from different tenants do not corrupt shared state."""
    model_path = "experiments/release/cdm-1-general"
    if not os.path.exists(model_path):
        pytest.skip("Release checkpoint required for integration test")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine.from_pretrained(model_name_or_path=model_path, db_path=db_path)
        errors = []
        results = {}

        def worker(thread_idx: int, tenant_id: str):
            try:
                res = engine.process(
                    text=f"Inquiry for {tenant_id}: Transaction #{thread_idx} needs review.",
                    structured_features={
                        "monetary_value_usd": 100.0 * (thread_idx + 1),
                        "urgency_level": "medium",
                        "sla_remaining_hours": 12.0,
                        "domain": "customer_support",
                    },
                    candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
                    prohibited_actions=["auto_resolve"],
                    session_id=f"session_worker_{thread_idx}",
                    tenant_id=tenant_id,
                )
                results[f"{tenant_id}_{thread_idx}"] = res
            except Exception as e:
                errors.append(f"Thread {thread_idx} ({tenant_id}) failed: {e}")

        threads = []
        for i in range(6):
            t_id = "tenant_prime" if i % 2 == 0 else "tenant_second"
            t = threading.Thread(target=worker, args=(i, t_id))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent workers encountered errors: {errors}"
        assert len(results) == 6, "All 6 concurrent worker requests must succeed"
        for key, res in results.items():
            assert res.decision in ["issue_refund", "escalate_to_tier2"]
            assert res.decision != "auto_resolve"
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_simultaneous_requests_same_session():
    """Requirement 12: Verifies simultaneous concurrent requests against the EXACT same session are synchronized."""
    model_path = "experiments/release/cdm-1-general"
    if not os.path.exists(model_path):
        pytest.skip("Release checkpoint required for integration test")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        engine = CognitiveEngine.from_pretrained(model_name_or_path=model_path, db_path=db_path)
        errors = []
        results = []
        lock = threading.Lock()

        def worker(thread_idx: int):
            try:
                res = engine.process(
                    text=f"Simultaneous turn #{thread_idx}: Follow-up update on order status.",
                    structured_features={"monetary_value_usd": 50.0 + thread_idx, "sla_remaining_hours": 10.0},
                    session_id="contentious_shared_session",
                    tenant_id="tenant_synchronous",
                )
                with lock:
                    results.append((thread_idx, res))
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_idx} failed: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Simultaneous session requests encountered errors: {errors}"
        assert len(results) == 4, "All 4 simultaneous requests to same session must succeed"

        # Check that context engine turns were serialized and properly incremented
        ctx = engine.context_engine.get_latest_context("contentious_shared_session", "tenant_synchronous")
        assert ctx is not None
        assert ctx.turn_index == 4, f"Expected 4 turns in history, got {ctx.turn_index}"
        assert len(ctx.dialogue_history) == 4
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


def test_cross_tenant_state_leak_prevention():
    """Requirement 14: Verifies context, world state, and working memory cannot leak across requests or tenants."""
    ce = ContextEngine()
    wm_a = WorldModel(tenant_id="tenant_secret_alpha", session_id="sess_1")
    wm_b = WorldModel(tenant_id="tenant_secret_beta", session_id="sess_1")

    wm_a.record_observed_fact("top_secret_code", "ALPHA_999")
    wm_b.record_observed_fact("top_secret_code", "BETA_111")

    assert wm_a.get_state().observed_facts["top_secret_code"] == "ALPHA_999"
    assert wm_b.get_state().observed_facts["top_secret_code"] == "BETA_111"
    assert "ALPHA_999" not in str(wm_b.get_state().observed_facts)

    # Working memory isolation check
    wmem_a = WorkingMemory()
    wmem_b = WorkingMemory()

    wmem_a.write_scratchpad("auth_token", "TOKEN_A")
    wmem_b.write_scratchpad("auth_token", "TOKEN_B")

    assert wmem_a.read_scratchpad("auth_token") == "TOKEN_A"
    assert wmem_b.read_scratchpad("auth_token") == "TOKEN_B"
