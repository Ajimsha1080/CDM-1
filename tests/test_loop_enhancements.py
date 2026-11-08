"""Unit tests for Agent Control Loop Enhancements (Long Tasks, Retries, Budgets, Resumability, Loops)."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock

from cdm.agent.loop import AgentControlLoop, AgentRunState, AgentActionProposal, AgentStore
from cdm.llm.base import BaseLLMProvider, LLMMessage, LLMResponse


class MockLoopLLM(BaseLLMProvider):
    """Configurable mock LLM provider for testing loop mechanics."""

    def __init__(self, responses=None):
        super().__init__()
        self.responses = responses or []
        self.call_count = 0
        self.history_received = []

    def _call_model(self, messages, temperature):
        return "", 0, 0

    def complete(self, messages, schema=None, tenant_id="default_tenant"):
        self.call_count += 1
        self.history_received.append(messages)

        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, LLMResponse):
                return resp
            if isinstance(resp, dict):
                return LLMResponse(
                    raw_text=str(resp),
                    parsed_json=resp,
                    input_tokens=50,
                    output_tokens=25,
                    cost_usd=0.001,
                    model_name=self.config.model_name,
                )
            if isinstance(resp, str):
                return LLMResponse(
                    raw_text=resp,
                    parsed_json=None,
                    input_tokens=50,
                    output_tokens=25,
                    cost_usd=0.001,
                    model_name=self.config.model_name,
                )

        # Default completion action
        return LLMResponse(
            raw_text='{"tool": "complete_task", "arguments": {}, "reason": "done", "claim_goal_met": true}',
            parsed_json={"tool": "complete_task", "arguments": {}, "reason": "done", "claim_goal_met": True},
            input_tokens=50,
            output_tokens=25,
            cost_usd=0.001,
            model_name=self.config.model_name,
        )


@pytest.fixture
def temp_loop_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_runs.sqlite")
        store = AgentStore(db_path=db_path)
        yield tmpdir, store


def test_step_budget_enforcement(temp_loop_env):
    tmpdir, store = temp_loop_env
    cont_action = {
        "tool": "read_file",
        "arguments": {"path": "test.txt"},
        "reason": "keep going",
        "claim_goal_met": False,
    }
    llm = MockLoopLLM(responses=[cont_action, cont_action, cont_action])
    loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)

    run = loop.create_run(goal="Test budget", task_description="Testing step limits", step_budget=2)
    # Step 1
    run = loop.step(run.run_id)
    assert run.step_count == 1
    # Step 2
    run = loop.step(run.run_id)
    assert run.step_count == 2
    # Step 3 exceeds budget
    run = loop.step(run.run_id)
    assert run.status == "step_budget_exceeded"
    assert run.step_count == 2


def test_cost_budget_enforcement(temp_loop_env):
    tmpdir, store = temp_loop_env
    # Each response costs $0.60
    resp = LLMResponse(
        raw_text='{"tool": "read_file", "arguments": {"path": "hello.txt"}, "reason": "inspect", "claim_goal_met": false}',
        parsed_json={"tool": "read_file", "arguments": {"path": "hello.txt"}, "reason": "inspect", "claim_goal_met": False},
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.60,
        model_name="mock",
    )
    llm = MockLoopLLM(responses=[resp, resp, resp])
    loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)

    # Cost budget $1.00
    run = loop.create_run(goal="Test cost", task_description="Testing cost limits", cost_budget_usd=1.00, step_budget=10)
    # Step 1: cost $0.60
    run = loop.step(run.run_id)
    assert run.status == "running"
    # Step 2: cost becomes $1.20 (exceeds budget)
    run = loop.step(run.run_id)
    # Next step should halt due to cost budget exceeded
    run = loop.step(run.run_id)
    assert run.status == "cost_budget_exceeded"


def test_loop_detection_intercepts_repeated_actions(temp_loop_env):
    tmpdir, store = temp_loop_env
    # Propose the exact same action 4 times
    same_action = {
        "tool": "read_file",
        "arguments": {"path": "nonexistent.py"},
        "reason": "keep trying same file",
        "claim_goal_met": False,
    }
    llm = MockLoopLLM(responses=[same_action, same_action, same_action, same_action, same_action])
    loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)

    run = loop.create_run(goal="Test loop", task_description="Looping actions", step_budget=10)

    # Step 1: executed
    run = loop.step(run.run_id)
    assert run.history[-1]["tool"] == "read_file"
    assert run.history[-1]["gate_decision"] == "allow"

    # Step 2: executed
    run = loop.step(run.run_id)
    assert run.history[-1]["gate_decision"] == "allow"

    # Step 3: identical 3rd attempt -> loop detected and blocked!
    run = loop.step(run.run_id)
    assert run.history[-1]["gate_decision"] == "blocked_loop"
    assert "Action loop detected" in run.history[-1]["error"]


def test_context_summarization_on_long_history(temp_loop_env):
    tmpdir, store = temp_loop_env
    loop = AgentControlLoop(store=store, sandbox_root=tmpdir)

    run = loop.create_run(goal="Long task", task_description="Test history compaction", step_budget=20)
    # Populate with 10 synthetic history entries
    for i in range(1, 11):
        run.history.append({
            "step": i,
            "tool": f"tool_{i}",
            "arguments": {"param": i},
            "gate_decision": "allow",
            "success": True,
            "output": f"Output for step {i}",
        })

    summary = loop._build_history_context(run)
    assert "Compact Summary of Earlier Steps 1 to 6" in summary
    assert "Recent Execution Steps:" in summary
    # Step 10 should be in recent steps
    assert "Step 10: Action 'tool_10'" in summary


def test_automatic_retry_of_malformed_llm_output(temp_loop_env):
    tmpdir, store = temp_loop_env
    # First response is malformed JSON, second response is valid JSON
    malformed = "NOT JSON AT ALL {{{ bad syntax"
    valid = {
        "tool": "complete_task",
        "arguments": {},
        "reason": "recovered after retry",
        "claim_goal_met": True,
    }
    llm = MockLoopLLM(responses=[malformed, valid])
    loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)

    run = loop.create_run(goal="Test retry", task_description="Test malformed retry")
    run = loop.step(run.run_id)

    # LLM should have been called twice (1 initial + 1 retry)
    assert llm.call_count == 2
    assert run.status == "completed"


def test_resumable_runs_from_sqlite(temp_loop_env):
    tmpdir, store = temp_loop_env
    cont_action = {
        "tool": "read_file",
        "arguments": {"path": "test.txt"},
        "reason": "keep going",
        "claim_goal_met": False,
    }
    llm = MockLoopLLM(responses=[cont_action, cont_action, cont_action])
    loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)

    run = loop.create_run(goal="Resume task", task_description="Test resumability", step_budget=1)
    run = loop.step(run.run_id)
    # Exhausts budget
    run = loop.step(run.run_id)
    assert run.status == "step_budget_exceeded"

    # Now create a fresh control loop with same store/db (simulating process restart)
    new_loop = AgentControlLoop(llm_provider=llm, store=store, sandbox_root=tmpdir)
    resumed = new_loop.resume_run(run.run_id, additional_steps=5, resume_loop=False)
    assert resumed.status == "running"
    assert resumed.step_budget == 6

    # Continue stepping
    resumed = new_loop.step(run.run_id)
    assert resumed.step_count == 2
