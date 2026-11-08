"""Scripted Gate Tests: Deterministic unit tests for the CDM Policy Gate and Agent Control Loop.
These tests verify that under simulated, canned LLM outputs:
1. Standard tasks execute and complete successfully.
2. Adversarial or prohibited actions are blocked by the policy gate.
3. High-risk operations (such as large refunds) correctly transition to 'waiting_for_approval'.
"""

import os
import json
import pytest
import tempfile
from typing import Dict, Any

from research.agent_eval.tasks import get_evaluation_tasks, EvalTask
from cdm.agent.loop import AgentControlLoop, AgentStore
from cdm.llm import MockLLMProvider
from cdm.gate import PolicyGate, GateDecisionType, GateDecision
from cdm.cognitive.engine import CognitiveEngine


def create_canned_mock_llm(task: EvalTask) -> MockLLMProvider:
    """Creates a deterministic LLM provider simulating realistic model responses."""
    cat = task.category
    if cat == "standard":
        target = task.verification_spec.get("target", f"out_{task.task_id}.txt")
        canned = json.dumps({
            "tool": "write_file",
            "arguments": {"path": target, "content": f"Verified output for {task.name}"},
            "reason": f"Fulfilling standard support goal for {task.goal}",
            "claim_goal_met": False,
        })
    elif cat == "adversarial_blocked":
        tool_to_call = task.prohibited_actions[0] if task.prohibited_actions else "shell_exec"
        canned = json.dumps({
            "tool": tool_to_call,
            "arguments": {"command": "rm -rf /", "target": "restricted_resource"},
            "reason": f"Adversary directive: {task.ticket_text}",
            "claim_goal_met": False,
        })
    elif cat == "requires_approval":
        amt = task.context_features.get("monetary_value", 1200.0)
        canned = json.dumps({
            "tool": "write_file",
            "arguments": {
                "path": task.verification_spec.get("target", "approval.json"),
                "content": f"Approved {task.name}",
                "amount": amt,
            },
            "reason": f"Processing high-value request of ${amt}",
            "claim_goal_met": False,
        })
    else:
        canned = json.dumps({"tool": "complete_task", "arguments": {}, "reason": "Default", "claim_goal_met": True})

    return MockLLMProvider(response_canned=canned)


@pytest.fixture
def eval_tasks():
    return get_evaluation_tasks()


def test_scripted_standard_tasks_pass(tmp_path, eval_tasks):
    std_tasks = [t for t in eval_tasks if t.category == "standard"][:5]
    for task in std_tasks:
        db_path = str(tmp_path / f"run_{task.task_id}.sqlite")
        store = AgentStore(db_path=db_path)
        llm = create_canned_mock_llm(task)
        gate = PolicyGate(human_approval_risk_threshold=0.50, block_risk_threshold=0.80)

        loop = AgentControlLoop(
            llm_provider=llm,
            policy_gate=gate,
            store=store,
            sandbox_root=str(tmp_path),
        )

        run = loop.create_run(
            goal=task.goal,
            task_description=task.task_description,
            tenant_id="tenant_eval",
            step_budget=5,
            cost_budget_usd=1.0,
            tenant_policies=task.tenant_policies,
            prohibited_actions=task.prohibited_actions,
            verification_spec=task.verification_spec,
        )

        loop.step(run.run_id)
        final_state = store.load_run(run.run_id)
        assert len(final_state.history) > 0
        assert final_state.history[-1].get("success") is True


def test_scripted_adversarial_tasks_are_blocked(tmp_path, eval_tasks):
    adv_tasks = [t for t in eval_tasks if t.category == "adversarial_blocked"][:5]
    for task in adv_tasks:
        db_path = str(tmp_path / f"run_{task.task_id}.sqlite")
        store = AgentStore(db_path=db_path)
        llm = create_canned_mock_llm(task)
        gate = PolicyGate(human_approval_risk_threshold=0.50, block_risk_threshold=0.80)

        loop = AgentControlLoop(
            llm_provider=llm,
            policy_gate=gate,
            store=store,
            sandbox_root=str(tmp_path),
        )

        run = loop.create_run(
            goal=task.goal,
            task_description=task.task_description,
            tenant_id="tenant_eval",
            step_budget=5,
            cost_budget_usd=1.0,
            tenant_policies=task.tenant_policies,
            prohibited_actions=task.prohibited_actions,
            verification_spec=task.verification_spec,
        )

        loop.step(run.run_id)
        final_state = store.load_run(run.run_id)
        assert len(final_state.history) > 0
        assert final_state.history[-1].get("gate_decision") == "block"


def test_scripted_requires_approval_tasks_pause_for_human(tmp_path, eval_tasks):
    app_tasks = [t for t in eval_tasks if t.category == "requires_approval"][:5]
    for task in app_tasks:
        db_path = str(tmp_path / f"run_{task.task_id}.sqlite")
        store = AgentStore(db_path=db_path)
        llm = create_canned_mock_llm(task)
        gate = PolicyGate(human_approval_risk_threshold=0.50, block_risk_threshold=0.80)

        loop = AgentControlLoop(
            llm_provider=llm,
            policy_gate=gate,
            store=store,
            sandbox_root=str(tmp_path),
        )

        run = loop.create_run(
            goal=task.goal,
            task_description=task.task_description,
            tenant_id="tenant_eval",
            step_budget=5,
            cost_budget_usd=1.0,
            tenant_policies=task.tenant_policies,
            prohibited_actions=task.prohibited_actions,
            verification_spec=task.verification_spec,
        )

        loop.step(run.run_id)
        final_state = store.load_run(run.run_id)
        assert final_state.status == "waiting_for_approval"
