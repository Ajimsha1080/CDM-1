"""Host Execution Safety Tests for CDM-1.

Verifies:
1. ShellTool (SandboxedShellTool) strictly refuses to run on host when Docker is unavailable.
2. RunTestsTool strictly refuses to run on host when Docker is unavailable (default allow_host=False).
3. Runtime tool arguments (e.g. from LLM action proposals or API requests) cannot bypass allow_host.
4. AgentControlLoop registers RunTestsTool with allow_host=False by default.
5. SWE-bench evaluation runner and configs never enable allow_host.
"""

import pytest
from unittest.mock import patch

from cdm.execution.tools.shell_tool import SandboxedShellTool
from cdm.execution.tools.engineering_tools import RunTestsTool
from cdm.agent.loop import AgentControlLoop
from research.swe_eval.configs import CONFIG_DEFINITIONS, build_configured_loop, EvalConfigName


def test_shell_tool_refuses_host_execution_when_docker_unavailable():
    """Verify SandboxedShellTool never falls back to host execution."""
    tool = SandboxedShellTool(enabled=True)
    with patch.object(tool, "is_docker_available", return_value=False):
        res = tool.execute({"command": "echo 'host execute attempt'"})
        assert res.success is False
        assert "refuses to execute on the host" in res.error.lower() or "docker is unavailable" in res.error.lower()


def test_run_tests_tool_refuses_host_execution_when_docker_unavailable():
    """Verify RunTestsTool refuses host execution when Docker is unavailable and allow_host is False."""
    tool = RunTestsTool(allow_host=False)
    with patch.object(tool, "is_docker_available", return_value=False):
        res = tool.execute({"test_command": "pytest"})
        assert res.success is False
        assert "never on the host" in res.error.lower()
        assert "docker is unavailable" in res.error.lower()


def test_runtime_arguments_cannot_enable_allow_host():
    """Verify that passing allow_host in tool arguments payload is ignored and cannot bypass sandbox."""
    tool = RunTestsTool(allow_host=False)
    with patch.object(tool, "is_docker_available", return_value=False):
        # Malicious action proposal attempting to inject allow_host: True
        res = tool.execute({"test_command": "echo 'bypass'", "allow_host": True})
        assert res.success is False
        assert "never on the host" in res.error.lower()


def test_agent_control_loop_default_tools_have_allow_host_disabled():
    """Verify AgentControlLoop defaults RunTestsTool to allow_host=False."""
    loop = AgentControlLoop()
    run_tests_tool = loop.tools.get("run_tests")
    assert isinstance(run_tests_tool, RunTestsTool)
    assert run_tests_tool.allow_host is False


def test_swe_eval_configs_have_allow_host_disabled(tmp_path):
    """Verify that all SWE-bench evaluation setups strictly disable allow_host."""
    for config_name, cfg in CONFIG_DEFINITIONS.items():
        if cfg.use_tools:
            loop = build_configured_loop(cfg, sandbox_root=str(tmp_path))
            run_tests_tool = loop.tools.get("run_tests")
            assert isinstance(run_tests_tool, RunTestsTool)
            assert run_tests_tool.allow_host is False
