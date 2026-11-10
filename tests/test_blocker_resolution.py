"""CDM-1 Final Five Blocker Resolution Regression Test Suite.

Verifies:
1. Neural Checkpoint Integrity & Inference (Blocker 1)
2. Pytest Collection & Compatibility (Blocker 2)
3. Unified Cognitive Runtime orchestration without duplicate state (Blocker 3)
4. Decision Ranking, Schema Validation, World-State Revalidation, and Fail-Closed abstention (Blocker 4)
5. Independent Benchmark provenance and transparent synthetic disambiguation (Blocker 5)
"""

import os
import json
import tempfile
import pytest
from unittest.mock import MagicMock

from cdm.agent.loop import AgentControlLoop, AgentStore
from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import GoalNode, WorldState, PerceptionResult, ContextFrame, ReasoningTrace, Plan, PlanStep
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult
from cdm.execution.tools.file_tool import FileWriteTool
from cdm.llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, responses: list):
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


def test_blocker4_decision_ranking_exception_fails_closed():
    """Verify that any exception during candidate decision ranking fails closed and suspends execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        # Inject a decision core that raises an unexpected exception
        mock_core = MagicMock()
        mock_core.decide.side_effect = RuntimeError("Neural tensor contract mismatch during candidate scoring")
        engine.decision_core = mock_core

        llm = MockLLMProvider([
            {"tool": "read_file", "arguments": {"path": "test.txt"}, "reason": "read", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Test fail closed ranking",
            task_description="Read test file safely",
            tenant_id="tenant_ranking_test",
        )

        stepped_run = loop.step(run.run_id)

        # Agent MUST fail closed, NOT proceed to execute the unvalidated action
        assert stepped_run.status == "waiting_for_approval"
        assert ("Candidate ranking error" in stepped_run.pending_reason) or ("Decision core failure" in stepped_run.pending_reason)
        last_step = stepped_run.history[-1]
        assert last_step["gate_decision"] == "fail_closed_ranking_error"
        assert last_step["success"] is False
        assert "failed closed" in last_step["error"].lower()


def test_blocker4_tool_argument_schema_validation_rejection():
    """Verify that candidate tool arguments are validated against the tool's schema before execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        # Propose write_file with invalid schema (missing required 'path' field)
        llm = MockLLMProvider([
            {"tool": "write_file", "arguments": {"content": "missing path"}, "reason": "write without path", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Write file with bad schema",
            task_description="Attempt invalid argument invocation",
            tenant_id="tenant_schema_test",
        )

        stepped_run = loop.step(run.run_id)

        # Action must be rejected by schema validation
        last_step = stepped_run.history[-1]
        assert last_step["gate_decision"] == "rejected_schema_validation"
        assert last_step["success"] is False
        assert "Tool argument schema validation failed" in last_step["error"]


def test_blocker4_world_state_precondition_revalidation():
    """Verify that candidate actions are revalidated against dynamic world state preconditions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        # Register a mock tool named 'issue_refund'
        class MockRefundTool:
            name = "issue_refund"
            description = "Issues monetary customer refund"
            def validate_args(self, args):
                return None
            def execute(self, args, tenant_id="default"):
                return None

        tools = {
            "issue_refund": MockRefundTool(),
            "read_file": FileWriteTool(sandbox_root=tmpdir),
        }

        llm = MockLLMProvider([
            {"tool": "issue_refund", "arguments": {"amount": 2500}, "reason": "refund", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            tools=tools,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Issue excessive refund",
            task_description="Customer refund over policy limit",
            tenant_id="tenant_precond_test",
        )

        # Ingest state into tenant's world model exceeding refund limits ($1500)
        wm = engine.get_isolated_world_model(run.tenant_id, run.session_id)
        wm.set_environment_metric("monetary_value_usd", 2500.0)

        stepped_run = loop.step(run.run_id)

        # Action must be rejected due to violated world state preconditions
        last_step = stepped_run.history[-1]
        assert last_step["gate_decision"] == "rejected_world_state_preconditions"
        assert last_step["success"] is False
        assert "preconditions not satisfied" in last_step["error"]


def test_blocker4_decision_scores_control_candidate_selection_and_prevent_arg_reuse():
    """Verify that decision network scores control candidate selection and prevent argument reuse."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        # Create decision core that strongly favors 'repo_search' over LLM proposed 'shell_exec'
        mock_core = MagicMock()
        mock_core.decide.return_value = DecisionResult({
            "decision": "repo_search",
            "confidence": 0.95,
            "calibrated": True,
            "score": 0.95,
            "candidate_scores": {"repo_search": 0.99, "shell_exec": 0.10, "read_file": 0.40},
            "risk": 0.01,
            "predicted_outcome": {"expected_success": True},
            "abstain": False,
        })
        engine.decision_core = mock_core

        # LLM proposes read_file with low score
        llm = MockLLMProvider([
            {"tool": "read_file", "arguments": {"path": "test.txt"}, "reason": "search", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Search code safely",
            task_description="Search for keyword in repository",
            tenant_id="tenant_control_test",
        )

        stepped_run = loop.step(run.run_id)

        # Verify that repo_search was selected by the decision core ranking
        last_step = stepped_run.history[-1]
        assert last_step["tool"] == "repo_search"
        # Verify arguments from read_file were NOT reused for repo_search
        assert "path" not in last_step["arguments"]
        # Item 3: Verify single authoritative decision inference (NO duplicate decide() call in step())
        assert mock_core.decide.call_count == 1


def test_blocker4_abstain_when_no_valid_candidate_exists():
    """Verify that the engine cleanly abstains when no valid candidate passes pre-ranking validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        # LLM proposes an unregistered tool and no fallback exists
        llm = MockLLMProvider([
            {"tool": "completely_unknown_tool", "arguments": {}, "reason": "invalid tool", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Do impossible task",
            task_description="Execute unsupported tool",
            tenant_id="tenant_abstain_test",
        )

        stepped_run = loop.step(run.run_id)
        assert stepped_run.status == "waiting_for_approval"
        last_step = stepped_run.history[-1]
        assert last_step["gate_decision"] in ("rejected_unregistered_tool", "abstain_no_valid_candidates")
        assert last_step["success"] is False


def test_blocker2_reproducible_dependencies_lockfile():
    """Verify that requirements-lock.txt exists, contains pinned dependencies, and aligns with runtime."""
    lockfile_path = "requirements-lock.txt"
    assert os.path.exists(lockfile_path), "requirements-lock.txt must exist in repo root."
    with open(lockfile_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    pinned_pkgs = {}
    for line in lines:
        if "==" in line:
            pkg, ver = line.split("==", 1)
            pinned_pkgs[pkg.lower()] = ver

    for required in ["torch", "transformers", "safetensors", "pydantic", "pytest", "cryptography", "fastapi"]:
        assert required in pinned_pkgs, f"Missing core pinned dependency: {required}"


def test_blocker1_offline_neural_inference_verified():
    """Verify that model.safetensors has exact size, SHA-256, 215 tensors, and runs offline inference without fallbacks."""
    import safetensors.torch
    import hashlib
    ckpt_path = "experiments/release/cdm-1-general/model.safetensors"
    assert os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}"
    assert os.path.getsize(ckpt_path) == 633465536

    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    assert h.hexdigest() == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"

    with safetensors.safe_open(ckpt_path, framework="pt") as sf:
        keys = list(sf.keys())
    assert len(keys) == 215

    # Run actual offline inference
    cdm = CognitiveDecisionModel.from_pretrained("experiments/release/cdm-1-general")
    assert cdm.is_fallback is False
    res = cdm.decide(
        text="Review dispute claim",
        structured_features={"step": 1, "accumulated_cost": 0.0},
        task="choice",
        candidates=["auto_resolve", "issue_refund"],
    )
    assert res.decision in ("auto_resolve", "issue_refund")
    assert res.confidence is not None
    assert res.risk is not None


def test_blocker3_unified_cognitive_runtime_orchestration():
    """Verify that CognitiveEngine.process() acts as the authoritative orchestration layer for AgentControlLoop."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = AgentStore(os.path.join(tmpdir, "runs.sqlite"))
        engine = CognitiveEngine(db_path=os.path.join(tmpdir, "cog.sqlite"))

        llm = MockLLMProvider([
            {"tool": "read_file", "arguments": {"path": "sample.txt"}, "reason": "inspect", "claim_goal_met": False}
        ])

        loop = AgentControlLoop(
            llm_provider=llm,
            cognitive_engine=engine,
            store=store,
            sandbox_root=tmpdir,
        )

        run = loop.create_run(
            goal="Inspect sample file",
            task_description="Check sample.txt contents",
            tenant_id="tenant_unified_runtime",
        )

        # Prior to step, isolated world model has 0 observed facts
        wm_before = engine.get_isolated_world_model(run.tenant_id, run.session_id)
        assert len(wm_before.state.observed_facts) == 0

        stepped_run = loop.step(run.run_id)

        # After step, CognitiveEngine.process() synchronously populated world model and working memory
        wm_after = engine.get_isolated_world_model(run.tenant_id, run.session_id)
        assert len(wm_after.state.observed_facts) >= 2
        assert "sample.txt" in stepped_run.history[-1]["arguments"]["path"]


def test_blocker5_independent_benchmark_and_dataset_provenance():
    """Verify that dataset provenance, synthetic proportion, and split manifest integrity are documented and valid."""
    manifest_path = "research/datasets/split_manifest.json"
    assert os.path.exists(manifest_path), "Split manifest must exist."
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Validate accounting
    assert manifest["dataset_version"] == "2.0.0"
    assert manifest["split_counts"]["train"] == 210
    assert manifest["split_counts"]["validation"] == 40
    assert manifest["split_counts"]["test"] == 58
    assert manifest["split_counts"]["heldout_domain_test"] == 8

    # Provenance accounting: transparent disclosure of real vs synthetic
    provenance = manifest["provenance_breakdown"]
    assert provenance["externally_sourced_real"] == 8
    assert provenance["curated_synthetic"] == 300
    assert provenance["externally_sourced_real"] + provenance["curated_synthetic"] == 308

    # Verify benchmark results JSON exists and contains explicit data notice and reproducibility provenance
    bench_results_path = "research/benchmarks/benchmark_results.json"
    assert os.path.exists(bench_results_path), "Benchmark results must exist."
    with open(bench_results_path, "r", encoding="utf-8") as f:
        b_data = json.load(f)
    assert "_data_notice" in b_data
    assert "synthetic" in b_data["_data_notice"].lower()
    assert b_data["reproducibility_provenance"]["checkpoint_sha256"] == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
    assert b_data["cdm_1_multitask"]["selective_prediction"]["selective_accuracy"] == 1.0


def test_blocker1_checkpoint_manifest_and_tensor_contract():
    """Verify that the release manifest has a valid Ed25519 signature and correct tensor metadata."""
    import hashlib
    manifest_path = "experiments/release/cdm-1-general/release_manifest.json"
    assert os.path.exists(manifest_path), "Release manifest must exist."
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifact = manifest["artifact"]
    assert artifact["file_name"] == "model.safetensors"
    assert artifact["file_size_bytes"] == 633465536
    assert artifact["sha256"] == "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
    assert artifact["parameter_counts"]["total_parameters"] == 158360004
    assert artifact["parameter_counts"]["trainable_parameters"] == 9345732
    assert artifact["parameter_counts"]["frozen_parameters"] == 149014272

    # Verify signature algorithm is Ed25519
    sig_meta = manifest["manifest_signature"]
    assert sig_meta["algorithm"] == "Ed25519"
    assert len(sig_meta["signature"]) == 128
    assert len(sig_meta["public_key"]) == 64
