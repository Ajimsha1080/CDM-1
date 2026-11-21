"""Tests for Cryptographically Verifiable, Expiring, Action-Bound Approvals in CDM-1."""

import time
import pytest
from cdm.security.approval import (
    generate_action_approval,
    verify_action_approval,
    ActionApproval,
)
from cdm.agent.loop import AgentControlLoop, AgentStore, AgentRunState


def test_action_approval_token_generation_and_verification():
    """Verify standard approval generation and successful HMAC validation."""
    token_str = generate_action_approval(
        tenant_id="tenant_finance",
        user_id="operator_sarah",
        plan_id="run_wire_001",
        action_name="issue_wire_transfer",
        ttl_seconds=300.0,
    )
    assert isinstance(token_str, str)
    assert len(token_str) > 30

    valid, err, app_obj = verify_action_approval(
        token_str=token_str,
        expected_tenant_id="tenant_finance",
        expected_plan_id="run_wire_001",
        expected_action_name="issue_wire_transfer",
    )
    assert valid is True
    assert err is None
    assert app_obj is not None
    assert app_obj.user_id == "operator_sarah"
    assert app_obj.action_name == "issue_wire_transfer"


def test_action_approval_rejection_on_expiration():
    """Verify expired token is rejected with explicit expiration error."""
    expired_token = generate_action_approval(
        tenant_id="tenant_ops",
        user_id="operator_mike",
        plan_id="run_restart_002",
        action_name="reboot_node",
        ttl_seconds=-5.0,  # Already expired
    )
    valid, err, app_obj = verify_action_approval(
        token_str=expired_token,
        expected_tenant_id="tenant_ops",
        expected_plan_id="run_restart_002",
        expected_action_name="reboot_node",
    )
    assert valid is False
    assert "Approval token expired" in (err or "")


def test_action_approval_rejection_on_action_mismatch():
    """Verify token bound to Action A cannot be reused or replayed for Action B."""
    token_action_a = generate_action_approval(
        tenant_id="tenant_ops",
        user_id="operator_mike",
        plan_id="run_003",
        action_name="read_telemetry",
        ttl_seconds=300.0,
    )
    valid, err, app_obj = verify_action_approval(
        token_str=token_action_a,
        expected_tenant_id="tenant_ops",
        expected_plan_id="run_003",
        expected_action_name="delete_database",  # Mismatched action
    )
    assert valid is False
    assert "Action mismatch" in (err or "")


def test_action_approval_rejection_on_tenant_mismatch():
    """Verify token from Tenant A cannot approve an action for Tenant B."""
    token_tenant_a = generate_action_approval(
        tenant_id="tenant_alpha",
        user_id="operator_1",
        plan_id="run_004",
        action_name="restart_service",
        ttl_seconds=300.0,
    )
    valid, err, app_obj = verify_action_approval(
        token_str=token_tenant_a,
        expected_tenant_id="tenant_beta",  # Different tenant
        expected_plan_id="run_004",
        expected_action_name="restart_service",
    )
    assert valid is False
    assert "Tenant mismatch" in (err or "")


def test_action_approval_rejection_on_signature_tampering():
    """Verify cryptographic HMAC tampering detection."""
    token_str = generate_action_approval(
        tenant_id="tenant_ops",
        user_id="operator_1",
        plan_id="run_005",
        action_name="execute_migration",
        ttl_seconds=300.0,
    )
    token_obj = ActionApproval.deserialize(token_str)
    # Tamper with signature
    token_obj.signature = "0" * len(token_obj.signature)
    tampered_str = token_obj.serialize()

    valid, err, app_obj = verify_action_approval(
        token_str=tampered_str,
        expected_tenant_id="tenant_ops",
        expected_plan_id="run_005",
        expected_action_name="execute_migration",
    )
    assert valid is False
    assert "forgery detected" in (err or "").lower() or "mismatch" in (err or "").lower()
