"""Unit and integration tests for Layer 3 Policy Gate and Layer 4 Sandboxed Execution & Verification."""

import os
import tempfile
import pytest

from cdm.gate import PolicyGate, ProposedAction, GateDecisionType
from cdm.cognitive.state import WorldState, WorldEntity
from cdm.execution.tools import (
    SandboxedShellTool,
    FileReadTool,
    FileWriteTool,
    FileEditTool,
    HttpCallTool,
)
from cdm.execution.verifier import OutcomeVerifier, VerificationSpec
from cdm.execution.audit import AuditTrailManager


def test_policy_gate_prohibited_action_blocks():
    """Verify that hard prohibited actions and dangerous commands are immediately blocked."""
    gate = PolicyGate()

    # Explicit prohibited action
    action = ProposedAction(tool="delete_database", args={}, reason="Cleanup database")
    res = gate.evaluate(action, prohibited_actions=["delete_database"])
    assert res.decision == GateDecisionType.BLOCK
    assert res.is_blocked is True
    assert "prohibited" in res.reason.lower()

    # Prohibited dangerous system command
    danger_action = ProposedAction(tool="shell_exec", args={"command": "rm -rf /var/log"}, reason="Clear logs")
    res_danger = gate.evaluate(danger_action)
    assert res_danger.decision == GateDecisionType.BLOCK
    assert res_danger.is_blocked is True


def test_policy_gate_monetary_bound_requires_human_approval():
    """Verify that financial amounts exceeding autonomous limit trigger require_human_approval."""
    gate = PolicyGate()
    action = ProposedAction(tool="issue_refund", args={"ticket_id": "T-10", "amount": 850.0}, reason="Customer dispute")
    res = gate.evaluate(action, monetary_limit=500.0)
    assert res.decision == GateDecisionType.REQUIRE_HUMAN_APPROVAL
    assert res.requires_human is True
    assert "exceeds autonomous limit" in res.reason


def test_policy_gate_precondition_against_world_state():
    """Verify that world state precondition checks block actions when target entity is missing."""
    gate = PolicyGate()
    world = WorldState(
        tenant_id="tenant_retail",
        session_id="sess_01",
        entities={"T-100": WorldEntity(entity_id="T-100", entity_type="ticket", properties={"status": "open"})},
    )
    # Action on unknown entity
    action = ProposedAction(tool="close_ticket", args={"ticket_id": "T-999"}, reason="Resolving")
    res = gate.evaluate(action, world_state=world)
    assert res.decision == GateDecisionType.BLOCK
    assert "does not exist in current world state" in res.reason


def test_sandboxed_file_tools_jail_enforcement():
    """Verify that file tools strictly prevent path traversal outside sandbox root."""
    with tempfile.TemporaryDirectory() as sandbox:
        writer = FileWriteTool(sandbox_root=sandbox)
        reader = FileReadTool(sandbox_root=sandbox)
        editor = FileEditTool(sandbox_root=sandbox)

        # Valid write and read
        w_res = writer.execute({"path": "sub/test.txt", "content": "Hello World!"})
        assert w_res.success is True

        r_res = reader.execute({"path": "sub/test.txt"})
        assert r_res.success is True
        assert r_res.output["content"] == "Hello World!"

        # Valid edit
        e_res = editor.execute({"path": "sub/test.txt", "find": "World", "replace": "CDM"})
        assert e_res.success is True
        r_res2 = reader.execute({"path": "sub/test.txt"})
        assert r_res2.output["content"] == "Hello CDM!"

        # Path traversal attack attempt: must fail without escaping sandbox
        evil_res = writer.execute({"path": "../outside.txt", "content": "escaped!"})
        assert evil_res.success is False
        assert "escapes sandbox root" in evil_res.error


def test_sandboxed_shell_tool_disabled_by_default_and_no_host_execution():
    """Verify shell tool is disabled by default and refuses host execution."""
    tool = SandboxedShellTool(timeout_seconds=2.0)
    assert tool.enabled is False
    res = tool.execute({"command": "echo 'cdm_test'"})
    assert res.success is False
    assert "disabled by default" in res.error.lower()

    # Enabled tool without Docker refuses execution
    tool_enabled = SandboxedShellTool(enabled=True, timeout_seconds=2.0)
    # Ensure it refuses when Docker is unavailable
    if not tool_enabled.is_docker_available():
        res_no_docker = tool_enabled.execute({"command": "echo 'cdm_test'"})
        assert res_no_docker.success is False
        assert "docker is unavailable" in res_no_docker.error.lower()


def test_sandboxed_http_tool_domain_allowlist():
    """Verify that HTTP tool blocks calls to non-allowlisted domains."""
    tool = HttpCallTool(allowed_domains={"localhost", "127.0.0.1"})

    # Attempt to reach external unauthorized domain
    res = tool.execute({"url": "https://unauthorized-evil-api.com/exfiltrate"})
    assert res.success is False
    assert "not in the approved tenant allowlist" in res.error


def test_outcome_verifier():
    """Verify that OutcomeVerifier checks ground truth rather than LLM assertions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = "verified_output.txt"
        spec_exists = VerificationSpec(check_type="file_exists", target=test_file)

        # Before creation: fails
        v_before = OutcomeVerifier.verify(spec_exists, sandbox_root=tmpdir)
        assert v_before.verified is False

        # After creation: passes
        with open(os.path.join(tmpdir, test_file), "w", encoding="utf-8") as f:
            f.write("Status: completed successfully.")

        v_after = OutcomeVerifier.verify(spec_exists, sandbox_root=tmpdir)
        assert v_after.verified is True

        # Check content
        spec_content = VerificationSpec(check_type="file_contains", target=test_file, expected_value="completed")
        v_content = OutcomeVerifier.verify(spec_content, sandbox_root=tmpdir)
        assert v_content.verified is True


def test_audit_trail_manager_cryptographic_verification():
    """Verify that audit trail manager records steps and cryptographically verifies signatures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AuditTrailManager(audit_dir=tmpdir)
        run_id = "test_run_123"

        rec = mgr.record_step(
            run_id=run_id,
            step_number=1,
            tenant_id="tenant_prod",
            proposed_action={"tool": "read_file", "args": {"path": "config.json"}, "reason": "Check config"},
            gate_decision={"decision": "allow", "risk_score": 0.15},
            execution_result={"success": True, "output": "loaded"},
            verification_result={"verified": True},
        )
        assert rec.run_id == run_id
        assert rec.provenance_record["provenance_signature"]

        # Verify audit trail
        valid, issues = mgr.verify_run_audit(run_id)
        assert valid is True
        assert len(issues) == 0
