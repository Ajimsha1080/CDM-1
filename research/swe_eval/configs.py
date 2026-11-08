"""Evaluation Configurations for SWE-bench Lite Benchmark in CDM-1.

Supported configurations:
- llm_only: Raw LLM predicts a unified patch directly from the issue description without tool loop.
- llm_tools: LLM + Sandboxed engineering tools, but CDM Policy Gate is bypassed.
- llm_tools_gate: LLM + Tools + CDM Policy Gate risk evaluation and loop detection.
- llm_tools_gate_memory: Full CDM-1 Layer 1-4 architecture: LLM + Tools + CDM Gate + Episodic Memory writeback loop.
"""

from typing import Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

from cdm.agent.loop import AgentControlLoop, AgentStore
from cdm.llm.base import BaseLLMProvider
from cdm.llm.factory import get_llm_provider
from cdm.cognitive.engine import CognitiveEngine
from cdm.gate.policy_gate import PolicyGate
from cdm.execution.tools.engineering_tools import (
    RepoSearchTool,
    EnhancedFileReadTool,
    EnhancedFileEditTool,
    FileCreateTool,
    RunTestsTool,
    GitDiffTool,
    GitStatusTool,
    ListDirectoryTool,
)


class EvalConfigName(str, Enum):
    LLM_ONLY = "llm_only"
    LLM_TOOLS = "llm_tools"
    LLM_TOOLS_GATE = "llm_tools_gate"
    LLM_TOOLS_GATE_MEMORY = "llm_tools_gate_memory"


class EvalConfig(BaseModel):
    name: EvalConfigName
    use_tools: bool
    use_gate: bool
    use_memory: bool
    max_steps: int = 30
    step_budget: int = 30
    cost_budget_usd: float = 2.50
    max_context_tokens: int = 12000
    description: str


CONFIG_DEFINITIONS: Dict[str, EvalConfig] = {
    EvalConfigName.LLM_ONLY.value: EvalConfig(
        name=EvalConfigName.LLM_ONLY,
        use_tools=False,
        use_gate=False,
        use_memory=False,
        max_steps=1,
        step_budget=1,
        cost_budget_usd=1.00,
        description="Single-shot direct LLM generation of patch; no tools or feedback loop.",
    ),
    EvalConfigName.LLM_TOOLS.value: EvalConfig(
        name=EvalConfigName.LLM_TOOLS,
        use_tools=True,
        use_gate=False,
        use_memory=False,
        max_steps=30,
        step_budget=30,
        cost_budget_usd=3.00,
        description="Autonomous loop with tools enabled; policy gate disabled.",
    ),
    EvalConfigName.LLM_TOOLS_GATE.value: EvalConfig(
        name=EvalConfigName.LLM_TOOLS_GATE,
        use_tools=True,
        use_gate=True,
        use_memory=False,
        max_steps=30,
        step_budget=30,
        cost_budget_usd=3.00,
        description="Autonomous loop with tools and CDM Policy Gate (safety bounds, risk checks, loop detection).",
    ),
    EvalConfigName.LLM_TOOLS_GATE_MEMORY.value: EvalConfig(
        name=EvalConfigName.LLM_TOOLS_GATE_MEMORY,
        use_tools=True,
        use_gate=True,
        use_memory=True,
        max_steps=30,
        step_budget=30,
        cost_budget_usd=3.00,
        description="Full CDM-1 Architecture: LLM + Tools + CDM Gate + Episodic Memory & World State feedback.",
    ),
}


def build_configured_loop(
    config: EvalConfig,
    sandbox_root: str,
    llm_provider: Optional[BaseLLMProvider] = None,
    container_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> AgentControlLoop:
    """Instantiates an AgentControlLoop configured according to the evaluation setup."""
    llm = llm_provider or get_llm_provider()
    store = AgentStore(db_path=db_path or "experiments/swe_eval_runs.sqlite")

    cognitive_engine = CognitiveEngine() if config.use_memory else None
    policy_gate = None
    if config.use_gate:
        decision_model = getattr(cognitive_engine, "decision_model", None) if cognitive_engine else None
        policy_gate = PolicyGate(decision_model=decision_model)
    else:
        # Pass-through gate
        policy_gate = PolicyGate(decision_model=None)

    tools = {}
    if config.use_tools:
        tools = {
            "repo_search": RepoSearchTool(sandbox_root=sandbox_root),
            "read_file": EnhancedFileReadTool(sandbox_root=sandbox_root),
            "edit_file": EnhancedFileEditTool(sandbox_root=sandbox_root),
            "create_file": FileCreateTool(sandbox_root=sandbox_root),
            "run_tests": RunTestsTool(sandbox_root=sandbox_root, container_id=container_id),
            "git_diff": GitDiffTool(sandbox_root=sandbox_root),
            "git_status": GitStatusTool(sandbox_root=sandbox_root),
            "list_directory": ListDirectoryTool(sandbox_root=sandbox_root),
        }

    return AgentControlLoop(
        llm_provider=llm,
        cognitive_engine=cognitive_engine,
        policy_gate=policy_gate,
        tools=tools,
        store=store,
        sandbox_root=sandbox_root,
    )
