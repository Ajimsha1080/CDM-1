"""Tests for Real Container-Sandboxed Shell Execution Tool (cdm/execution/tools/shell_tool.py).

Verifies:
1. Tool is disabled by default.
2. Tool strictly refuses execution when Docker is unavailable and never falls back to host.
3. Argument lists only are used (no shell=True).
4. Container isolation (requires_docker):
   - Network access fails (--network none)
   - Writes outside workspace fail (--read-only)
   - Resource limits: fork bomb (--pids-limit) and memory hog (--memory) are killed
   - Hard timeout forcibly kills container
   - Bypass attempts (rm -r -f /, python -c, curl | sh) cannot affect host
"""

import pytest
import os
import tempfile
from unittest.mock import patch
from cdm.execution.tools.shell_tool import SandboxedShellTool


def test_shell_tool_disabled_by_default():
    """Verify tool is disabled by default and refuses execution."""
    tool = SandboxedShellTool()
    assert tool.enabled is False

    res = tool.execute({"command": "echo 'should_not_run'"})
    assert res.success is False
    assert "disabled by default" in res.error.lower()


def test_shell_tool_refuses_when_docker_unavailable(monkeypatch):
    """Verify tool refuses to run and returns structured error if Docker is unavailable."""
    tool = SandboxedShellTool(enabled=True)
    monkeypatch.setattr(tool, "is_docker_available", lambda: False)

    res = tool.execute({"command": "echo 'host_test'"})
    assert res.success is False
    assert "docker is unavailable" in res.error.lower()
    assert "refuses to execute on the host" in res.error.lower()


def test_shell_tool_requires_command_param(monkeypatch):
    """Verify command parameter validation."""
    tool = SandboxedShellTool(enabled=True)
    monkeypatch.setattr(tool, "is_docker_available", lambda: True)

    res = tool.execute({})
    assert res.success is False
    assert "required" in res.error.lower()

    res_empty = tool.execute({"command": ""})
    assert res_empty.success is False


def test_shell_tool_docker_command_construction(monkeypatch):
    """Verify docker arguments are passed as argument list without shell=True."""
    tool = SandboxedShellTool(
        enabled=True,
        timeout_seconds=5.0,
        working_dir="/tmp/fake_dir",
        memory_limit="64m",
        cpu_limit=0.5,
        pids_limit=32,
    )
    monkeypatch.setattr(tool, "is_docker_available", lambda: True)

    captured_args = []

    def mock_subprocess_run(args, **kwargs):
        captured_args.extend(args)
        assert kwargs.get("shell") is False
        class MockProc:
            returncode = 0
            stdout = "mock_success"
            stderr = ""
        return MockProc()

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    res = tool.execute({"command": "echo hello"})
    assert res.success is True
    assert captured_args[0] == "docker"
    assert captured_args[1] == "run"
    assert "--network" in captured_args
    assert "none" in captured_args
    assert "--read-only" in captured_args
    assert "--cap-drop" in captured_args
    assert "ALL" in captured_args
    assert "--security-opt" in captured_args
    assert "no-new-privileges" in captured_args
    assert "--memory" in captured_args
    assert "64m" in captured_args
    assert "--cpus=0.5" in captured_args
    assert "--pids-limit=32" in captured_args
    assert any("sha256:" in arg for arg in captured_args)


@pytest.mark.requires_docker
def test_docker_sandbox_network_isolation():
    """Verify container cannot access external network (--network none)."""
    tool = SandboxedShellTool(enabled=True, timeout_seconds=10.0)
    if not tool.is_docker_available():
        pytest.skip("Docker daemon is not running")

    # ping or wget to public IP must fail
    res = tool.execute({"command": "ping -c 1 -W 2 8.8.8.8 || wget -T 2 -qO- http://1.1.1.1"})
    assert res.success is True
    assert res.output["exit_code"] != 0


@pytest.mark.requires_docker
def test_docker_sandbox_readonly_root_blocks_writes():
    """Verify writes outside writable mounts fail (--read-only)."""
    tool = SandboxedShellTool(enabled=True, timeout_seconds=10.0)
    if not tool.is_docker_available():
        pytest.skip("Docker daemon is not running")

    res = tool.execute({"command": "touch /root/evil.txt"})
    assert res.success is True
    assert res.output["exit_code"] != 0
    assert "read-only file system" in res.output["stderr"].lower()


@pytest.mark.requires_docker
def test_docker_sandbox_fork_bomb_killed():
    """Verify fork bomb is terminated by pids-limit."""
    tool = SandboxedShellTool(enabled=True, timeout_seconds=10.0, pids_limit=16)
    if not tool.is_docker_available():
        pytest.skip("Docker daemon is not running")

    res = tool.execute({"command": ":(){ :|:& };:"})
    assert res.success is True
    # Either fork error or process killed
    assert res.output["exit_code"] != 0


@pytest.mark.requires_docker
def test_docker_sandbox_memory_hog_killed():
    """Verify memory hog is terminated by memory limit."""
    tool = SandboxedShellTool(enabled=True, timeout_seconds=10.0, memory_limit="32m")
    if not tool.is_docker_available():
        pytest.skip("Docker daemon is not running")

    # Allocate 100MB in a 32MB container
    res = tool.execute({"command": "python3 -c 'a = \"x\" * 100000000' || dd if=/dev/zero of=/dev/null bs=100M count=1"})
    assert res.success is True
    assert res.output["exit_code"] != 0


@pytest.mark.requires_docker
def test_docker_sandbox_hard_timeout_kills_container():
    """Verify hard timeout forcibly kills the container."""
    tool = SandboxedShellTool(enabled=True, timeout_seconds=2.0)
    if not tool.is_docker_available():
        pytest.skip("Docker daemon is not running")

    res = tool.execute({"command": "sleep 15"})
    assert res.success is True
    assert res.output["timed_out"] is True
    assert res.output["exit_code"] == -1


@pytest.mark.requires_docker
def test_docker_sandbox_bypass_attempts_cannot_affect_host():
    """Verify bypass attempts cannot affect host filesystem."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "canary.txt")
        with open(test_file, "w") as f:
            f.write("intact")

        tool = SandboxedShellTool(enabled=True, working_dir=temp_dir, timeout_seconds=10.0)
        if not tool.is_docker_available():
            pytest.skip("Docker daemon is not running")

        # Attempt destructive command
        res = tool.execute({"command": "rm -r -f / 2>/dev/null || true"})
        assert res.success is True

        # Host canary file must be intact
        assert os.path.exists(test_file)
        with open(test_file, "r") as f:
            assert f.read() == "intact"
