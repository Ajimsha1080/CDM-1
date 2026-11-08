"""Tests for Layer 4 Agent Control Loop, Persistent Resumption, and REST API Endpoints."""

import os
import tempfile
import pytest
from fastapi.testclient import TestClient

from cdm.agent.loop import AgentControlLoop, AgentStore, AgentRunState
from cdm.llm import MockLLMProvider
from cdm.execution.tools import FileWriteTool, FileReadTool
from api.main import app, KEY_MANAGER

TEST_TENANT_A = "tenant_agent_alpha"
TEST_KEY_A = "cdm_agent_key_alpha_12345"

TEST_TENANT_B = "tenant_agent_beta"
TEST_KEY_B = "cdm_agent_key_beta_67890"


@pytest.fixture(scope="module", autouse=True)
def setup_api_keys():
    KEY_MANAGER.register_key(TEST_KEY_A, TEST_TENANT_A, tier="enterprise", rate_limit_rpm=500)
    KEY_MANAGER.register_key(TEST_KEY_B, TEST_TENANT_B, tier="standard", rate_limit_rpm=50)


def test_agent_control_loop_execution_and_verifier():
    """Verify that AgentControlLoop steps, executes sandboxed tools, and verifier confirms goal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(db_path=os.path.join(tmpdir, "runs.sqlite"))
        loop = AgentControlLoop(
            llm_provider=MockLLMProvider(
                response_canned='{"tool": "write_file", "arguments": {"path": "target.txt", "content": "goal_achieved"}, "reason": "Writing goal file", "claim_goal_met": false}'
            ),
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Write target file to disk",
            task_description="Create target.txt with contents goal_achieved",
            tenant_id="test_tenant",
            verification_spec={"check_type": "file_exists", "target": "target.txt"},
        )

        # Step 1: Executes write_file
        run_after_step = loop.step(run.run_id)
        assert run_after_step.step_count == 1
        assert run_after_step.tool_call_count == 1
        assert run_after_step.status == "completed" # Verifier confirms file exists!

        # Verify physical file exists on disk
        assert os.path.isfile(os.path.join(tmpdir, "target.txt"))

        # Verify run can be reloaded from persistent store
        reloaded = store.load_run(run.run_id)
        assert reloaded is not None
        assert reloaded.status == "completed"
        assert len(reloaded.history) == 1


def test_agent_control_loop_human_approval_pause_and_resume():
    """Verify that high-risk action pauses the loop for human approval, and approve_pending resumes it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(db_path=os.path.join(tmpdir, "runs.sqlite"))
        # Action proposing refund of $1200 (exceeds $500 threshold)
        loop = AgentControlLoop(
            llm_provider=MockLLMProvider(
                response_canned='{"tool": "shell_exec", "arguments": {"command": "echo refund_processed", "amount": 1200.0}, "reason": "High value customer refund", "claim_goal_met": false}'
            ),
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Process large refund",
            task_description="Refund customer $1200",
            tenant_id="test_tenant",
        )

        # Step: Policy gate flags financial bound -> waiting_for_approval
        run_paused = loop.step(run.run_id)
        assert run_paused.status == "waiting_for_approval"
        assert run_paused.pending_action is not None
        assert "exceeds autonomous limit" in run_paused.pending_reason

        # Operator approves the pending action without auto-looping
        run_resumed = loop.approve_pending(run.run_id, resume_loop=False)
        assert run_resumed.status in ("running", "completed")
        assert run_resumed.pending_action is None
        assert len(run_resumed.history) >= 1
        assert run_resumed.history[0]["gate_decision"] == "human_approved"


def test_agent_control_loop_human_rejection():
    """Verify that rejecting a suspended action records the rejection cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(db_path=os.path.join(tmpdir, "runs.sqlite"))
        loop = AgentControlLoop(
            llm_provider=MockLLMProvider(
                response_canned='{"tool": "shell_exec", "arguments": {"command": "echo test", "amount": 999.0}, "reason": "Test", "claim_goal_met": false}'
            ),
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(goal="Test", task_description="Test", tenant_id="test_tenant")
        loop.step(run.run_id) # Pauses for approval

        # Reject
        rejected = loop.reject_pending(run.run_id, rejection_reason="Too risky for today")
        assert rejected.status == "rejected"
        assert "Too risky" in rejected.history[0]["error"]


def test_agent_rest_api_lifecycle_and_tenant_isolation():
    """Verify REST API /v1/agent/runs endpoints and tenant isolation enforcement."""
    with TestClient(app) as client:
        headers_a = {"Authorization": f"Bearer {TEST_KEY_A}"}
        headers_b = {"Authorization": f"Bearer {TEST_KEY_B}"}

        # 1. Create run for Tenant A
        payload = {
            "goal": "Audit user records",
            "task_description": "Inspect user table",
            "step_budget": 5,
            "cost_budget_usd": 2.0,
            "auto_run": False, # Just create without stepping
        }
        r_create = client.post("/v1/agent/runs", json=payload, headers=headers_a)
        assert r_create.status_code == 200
        run_data = r_create.json()
        run_id = run_data["run_id"]
        assert run_data["tenant_id"] == TEST_TENANT_A
        assert run_data["status"] == "running"

        # 2. Get run as Tenant A (allowed)
        r_get = client.get(f"/v1/agent/runs/{run_id}", headers=headers_a)
        assert r_get.status_code == 200
        assert r_get.json()["run_id"] == run_id

        # 3. Tenant B tries to access Tenant A's run -> 403 Forbidden
        r_forbidden = client.get(f"/v1/agent/runs/{run_id}", headers=headers_b)
        assert r_forbidden.status_code == 403

        # 4. Tenant B tries to approve Tenant A's run -> 403 Forbidden
        r_app_forbidden = client.post(f"/v1/agent/runs/{run_id}/approve", headers=headers_b)
        assert r_app_forbidden.status_code == 403


def test_batch_charging_and_cognitive_process_metering():
    """Verify that /v1/batch checks and charges per item, and /v1/cognitive/process records metering."""
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TEST_KEY_A}"}

        # 1. Cognitive process endpoint records metering
        r_cog = client.post(
            "/v1/cognitive/process",
            json={"text": "Investigating server alert", "task": "choice", "candidates": ["isolate", "monitor"]},
            headers=headers,
        )
        assert r_cog.status_code == 200

        # 2. Batch endpoint
        batch_payload = {
            "requests": [
                {"text": "Inquiry 1", "task": "binary"},
                {"text": "Inquiry 2", "task": "binary"},
                {"text": "Inquiry 3", "task": "binary"},
            ]
        }
        r_batch = client.post("/v1/batch", json=batch_payload, headers=headers)
        assert r_batch.status_code == 200
        assert r_batch.json()["count"] == 3

        # Check metering summary has recorded requests
        r_meter = client.get("/v1/metering", headers=headers)
        assert r_meter.status_code == 200
        usage = r_meter.json()["usage"]
        assert usage["total_requests"] >= 4
