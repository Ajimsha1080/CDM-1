"""Unit tests for Layer 4 Sandboxed Engineering Tools."""

import os
import tempfile
import pytest

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
from cdm.execution.tools.base import ToolResult


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test structure
        os.makedirs(os.path.join(tmpdir, "pkg", "subpkg"), exist_ok=True)
        with open(os.path.join(tmpdir, "hello.py"), "w", encoding="utf-8") as f:
            f.write("def hello():\n    return 'world'\n\ndef target():\n    pass\n")
        with open(os.path.join(tmpdir, "pkg", "subpkg", "calc.py"), "w", encoding="utf-8") as f:
            f.write("# Calculator module\ndef add(a, b):\n    return a + b\n")
        yield tmpdir


def test_repo_search(temp_workspace):
    tool = RepoSearchTool(sandbox_root=temp_workspace)
    res = tool.execute({"query": "def add"})
    assert res.success is True
    assert res.output["matches_count"] >= 1
    assert any("calc.py" in m["file"] for m in res.output["matches"])

    # Path traversal protection
    bad_res = tool.execute({"query": "root", "path": "../../../etc"})
    assert bad_res.success is False
    assert "escapes" in bad_res.error or "jail" in bad_res.error.lower()


def test_enhanced_file_read(temp_workspace):
    tool = EnhancedFileReadTool(sandbox_root=temp_workspace)
    # Read slice
    res = tool.execute({"path": "hello.py", "start_line": 1, "end_line": 2})
    assert res.success is True
    assert res.output["lines_returned"] == 2
    assert "def hello():" in res.output["content"]

    # Missing file
    res_missing = tool.execute({"path": "nonexistent.py"})
    assert res_missing.success is False

    # Path traversal attempt
    res_jail = tool.execute({"path": "../secret.txt"})
    assert res_jail.success is False


def test_enhanced_file_edit(temp_workspace):
    tool = EnhancedFileEditTool(sandbox_root=temp_workspace)
    res = tool.execute({
        "path": "hello.py",
        "old_str": "return 'world'",
        "new_str": "return 'cdm-1'",
    })
    assert res.success is True
    assert res.output["replacements_made"] == 1

    # Verify content changed
    with open(os.path.join(temp_workspace, "hello.py"), "r", encoding="utf-8") as f:
        content = f.read()
    assert "return 'cdm-1'" in content

    # Target not found error
    fail_res = tool.execute({
        "path": "hello.py",
        "old_str": "non_existent_function()",
        "new_str": "new_func()",
    })
    assert fail_res.success is False
    assert "not found" in fail_res.error.lower()


def test_file_create(temp_workspace):
    tool = FileCreateTool(sandbox_root=temp_workspace)
    res = tool.execute({
        "path": "nested/dir/new_script.py",
        "content": "print('hello from cdm')\n",
    })
    assert res.success is True
    assert res.output["created"] is True

    # Verify file exists on disk
    dest = os.path.join(temp_workspace, "nested", "dir", "new_script.py")
    assert os.path.isfile(dest)
    with open(dest, "r", encoding="utf-8") as f:
        assert f.read() == "print('hello from cdm')\n"


def test_list_directory(temp_workspace):
    tool = ListDirectoryTool(sandbox_root=temp_workspace)
    res = tool.execute({"path": ".", "max_depth": 3})
    assert res.success is True
    assert any("hello.py" in f for f in res.output["files"])
    assert any("pkg" in d for d in res.output["directories"])


def test_run_tests_docker_unavailable_refusal(temp_workspace):
    # Default behavior without docker: refuses to run on host
    tool = RunTestsTool(sandbox_root=temp_workspace, allow_host=False)
    # If docker daemon is not available, execute() returns structured failure
    if not tool.is_docker_available():
        res = tool.execute({"test_command": "python -m unittest"})
        assert res.success is False
        assert "Docker is unavailable" in res.error
        assert "never on the host" in res.error


def test_run_tests_with_allow_host_mock(temp_workspace):
    tool = RunTestsTool(sandbox_root=temp_workspace, allow_host=True)
    res = tool.execute({"test_command": "python -c \"print('test pass')\""})
    assert res.success is True
    assert res.output["passed"] is True
    assert "test pass" in res.output["stdout_trimmed"]

    # Failing test command
    res_fail = tool.execute({"test_command": "python -c \"import sys; sys.exit(1)\""})
    assert res_fail.success is True
    assert res_fail.output["passed"] is False
    assert res_fail.output["exit_code"] == 1


def test_git_status_and_diff(temp_workspace):
    status_tool = GitStatusTool(sandbox_root=temp_workspace)
    diff_tool = GitDiffTool(sandbox_root=temp_workspace)

    # In a temp dir that isn't a git repo, tools return safe structured envelope without raising
    res_status = status_tool.execute({})
    assert isinstance(res_status, ToolResult)

    res_diff = diff_tool.execute({})
    assert isinstance(res_diff, ToolResult)
