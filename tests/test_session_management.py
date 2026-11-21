"""Tests for CDM-1 Session Management, Conversation Context Persistence, and Tenant Isolation."""

import pytest
from cdm.context.session_manager import SessionManager, ConversationSession
from cdm.context.context_engine import ContextEngine
from cdm.cognitive.state import PerceptionResult


def test_session_lifecycle_and_tenant_isolation(tmp_path):
    """Verify unique session creation, listing, retrieval, and cross-tenant isolation."""
    db_path = str(tmp_path / "sessions_test.sqlite")
    mgr = SessionManager(db_path=db_path)

    # 1. Create sessions for two distinct tenants
    sess_a1 = mgr.create_session(tenant_id="tenant_a", user_id="alice", title="Alice Cloud Debug")
    sess_a2 = mgr.create_session(tenant_id="tenant_a", user_id="alice", title="Alice Billing")
    sess_b1 = mgr.create_session(tenant_id="tenant_b", user_id="bob", title="Bob Security Audit")

    # 2. Verify unique IDs generated
    assert sess_a1.session_id != sess_a2.session_id
    assert sess_a1.session_id != sess_b1.session_id

    # 3. List sessions: Tenant A sees only its own 2 sessions
    list_a = mgr.list_sessions(tenant_id="tenant_a")
    assert len(list_a) == 2
    session_ids_a = {s.session_id for s in list_a}
    assert sess_a1.session_id in session_ids_a
    assert sess_a2.session_id in session_ids_a
    assert sess_b1.session_id not in session_ids_a

    # 4. Cross-tenant retrieval is blocked
    b_as_a = mgr.get_session(session_id=sess_b1.session_id, tenant_id="tenant_a")
    assert b_as_a is None

    # 5. Activity update
    mgr.update_session_activity(
        session_id=sess_a1.session_id,
        tenant_id="tenant_a",
        turn_increment=1,
        goals=["isolate_pod"],
        constraints=["sla_999"],
    )
    updated = mgr.get_session(session_id=sess_a1.session_id, tenant_id="tenant_a")
    assert updated.turn_count == 1
    assert "isolate_pod" in updated.active_goals
    assert "sla_999" in updated.active_constraints

    # 6. Deletion lifecycle
    deleted = mgr.delete_session(session_id=sess_a1.session_id, tenant_id="tenant_a")
    assert deleted is True
    assert mgr.get_session(session_id=sess_a1.session_id, tenant_id="tenant_a") is None
    assert len(mgr.list_sessions(tenant_id="tenant_a")) == 1


def test_context_engine_multi_turn_history():
    """Verify ContextEngine stores dialogue turns and tracks temporal deltas without default sharing."""
    ce = ContextEngine()
    perception = PerceptionResult(raw_text="Check server health", task="choice")

    # Turn 1
    frame1 = ce.update_context(
        session_id="sess_101",
        tenant_id="tenant_alpha",
        perception=perception,
        user_message="Is pod-7 healthy?",
        system_response="Pod-7 is reporting 92% memory.",
    )
    assert frame1.turn_index == 1
    assert len(frame1.dialogue_history) == 2

    # Turn 2
    frame2 = ce.update_context(
        session_id="sess_101",
        tenant_id="tenant_alpha",
        perception=perception,
        user_message="Should we restart it?",
        system_response="Restarting pod-7 is recommended.",
    )
    assert frame2.turn_index == 2
    assert len(frame2.dialogue_history) == 4

    # History retrieval
    history = ce.get_history(session_id="sess_101", tenant_id="tenant_alpha")
    assert len(history) == 2
    assert history[-1].turn_index == 2

    # Tenant isolation in context engine
    assert len(ce.get_history(session_id="sess_101", tenant_id="tenant_beta")) == 0
