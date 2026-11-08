"""Layer 4: Sandboxed Engineering Tools for Software Development & Coding Agents.

All tools operate strictly within the checked-out repository sandbox root,
enforce path jail restrictions, return structured output envelopes, truncate large outputs,
and never raise uncaught exceptions into the control loop.
"""

from typing import Dict, Any, Optional, List
import os
import re
import subprocess
from pathlib import Path

from cdm.execution.tools.base import BaseTool
from cdm.execution.tools.file_tool import _resolve_sandbox_path


class RepoSearchTool(BaseTool):
    """ripgrep/grep-style text and regex search tool across repository files."""

    def __init__(self, sandbox_root: Optional[str] = None, max_results: int = 50, max_file_size: int = 1048576):
        super().__init__(
            name="repo_search",
            description="Searches for a text pattern or regex across files in the repository. Returns matching files, line numbers, and contents.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()
        self.max_results = max_results
        self.max_file_size = max_file_size

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        query = args.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("Parameter 'query' is required and must be a non-empty string.")

        subpath = args.get("path", "")
        file_pattern = args.get("file_pattern")
        case_sensitive = args.get("case_sensitive", True)
        limit = min(int(args.get("max_results", self.max_results)), 200)

        search_dir = _resolve_sandbox_path(subpath, self.sandbox_root)
        if not os.path.isdir(search_dir):
            raise FileNotFoundError(f"Search directory not found: {subpath or '.'}")

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{query}': {e}")

        matches = []
        truncated = False
        files_scanned = 0

        # Walk repository files safely
        for root, dirs, files in os.walk(search_dir):
            # Skip hidden and VCS directories
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

            for fname in files:
                if fname.startswith(".") or fname.endswith((".pyc", ".safetensors", ".so", ".dll", ".zip", ".tar", ".gz")):
                    continue
                if file_pattern and not re.search(file_pattern, fname):
                    continue

                full_path = os.path.join(root, fname)
                try:
                    if os.path.getsize(full_path) > self.max_file_size:
                        continue
                    files_scanned += 1
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_idx, line in enumerate(f, 1):
                            if pattern.search(line):
                                rel_path = os.path.relpath(full_path, self.sandbox_root)
                                matches.append({
                                    "file": rel_path.replace("\\", "/"),
                                    "line_number": line_idx,
                                    "line_content": line.rstrip()[:500],
                                })
                                if len(matches) >= limit:
                                    truncated = True
                                    break
                except Exception:
                    continue

                if truncated:
                    break
            if truncated:
                break

        return {
            "query": query,
            "matches_count": len(matches),
            "files_scanned": files_scanned,
            "truncated": truncated,
            "matches": matches,
        }


class EnhancedFileReadTool(BaseTool):
    """File reading tool with line range slicing and bounded output."""

    def __init__(self, sandbox_root: Optional[str] = None, max_lines: int = 500, max_bytes: int = 65536):
        super().__init__(
            name="read_file",
            description="Reads text content of a file within the sandbox, supporting start_line and end_line slicing.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()
        self.max_lines = max_lines
        self.max_bytes = max_bytes

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        if not path:
            raise ValueError("Parameter 'path' is required.")

        target = _resolve_sandbox_path(path, self.sandbox_root)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: {path}")

        start_line = max(1, int(args.get("start_line", 1)))
        end_line = args.get("end_line")
        end_line = int(end_line) if end_line is not None else None

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)
        if end_line is None or end_line > total_lines:
            end_line = total_lines

        selected_lines = all_lines[start_line - 1 : end_line]
        truncated = False

        if len(selected_lines) > self.max_lines:
            selected_lines = selected_lines[: self.max_lines]
            truncated = True

        content = "".join(selected_lines)
        if len(content.encode("utf-8")) > self.max_bytes:
            content = content[: self.max_bytes]
            truncated = True

        return {
            "path": path,
            "start_line": start_line,
            "end_line": start_line + len(selected_lines) - 1 if selected_lines else start_line,
            "total_lines": total_lines,
            "lines_returned": len(selected_lines),
            "truncated": truncated,
            "content": content,
        }


class EnhancedFileEditTool(BaseTool):
    """Exact string replacement tool with count verification."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(
            name="edit_file",
            description="Performs exact string replacement in a file within the sandbox.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        old_str = args.get("old_str") or args.get("find")
        new_str = args.get("new_str") if "new_str" in args else args.get("replace")

        if not path or old_str is None or new_str is None:
            raise ValueError("Parameters 'path', 'old_str', and 'new_str' are required.")

        target = _resolve_sandbox_path(path, self.sandbox_root)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: {path}")

        with open(target, "r", encoding="utf-8") as f:
            orig = f.read()

        if old_str not in orig:
            raise ValueError(f"Target string not found in {path}. Exact match required.")

        count = orig.count(old_str)
        updated = orig.replace(old_str, new_str)

        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)

        return {
            "path": path,
            "replacements_made": count,
            "bytes_updated": len(updated.encode("utf-8")),
            "success": True,
        }


class FileCreateTool(BaseTool):
    """File creation tool with automatic parent directory creation."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(
            name="create_file",
            description="Creates a new file or overwrites an existing file in the sandbox.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        content = args.get("content", "")

        if not path:
            raise ValueError("Parameter 'path' is required.")

        target = _resolve_sandbox_path(path, self.sandbox_root)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
            "created": True,
        }


class RunTestsTool(BaseTool):
    """Runs a test command with timeout inside an isolated container and returns a trimmed failure summary."""

    def __init__(
        self,
        sandbox_root: Optional[str] = None,
        default_timeout: float = 60.0,
        container_id: Optional[str] = None,
        allow_host: bool = False,
    ):
        super().__init__(
            name="run_tests",
            description="Runs a test command in the repository inside a sandbox container and returns a structured failure summary.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()
        self.default_timeout = min(default_timeout, 300.0)
        self.container_id = container_id
        self.allow_host = allow_host

    _docker_cached: Optional[bool] = None

    def is_docker_available(self) -> bool:
        """Verify whether Docker CLI and daemon are accessible (cached)."""
        if RunTestsTool._docker_cached is not None:
            return RunTestsTool._docker_cached
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=1.0,
                shell=False,
            )
            RunTestsTool._docker_cached = (res.returncode == 0)
        except Exception:
            RunTestsTool._docker_cached = False
        return RunTestsTool._docker_cached

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        test_command = args.get("test_command") or args.get("command")
        if not test_command or not isinstance(test_command, str):
            raise ValueError("Parameter 'test_command' is required and must be a string.")

        timeout = min(float(args.get("timeout_seconds", self.default_timeout)), 300.0)
        target_container = args.get("container_id", self.container_id)

        # Enforce sandbox container execution: refuse to run on host without container unless allow_host is True
        docker_available = self.is_docker_available()
        if not docker_available and not self.allow_host:
            raise RuntimeError(
                "Docker is unavailable: tests must run inside a sandbox container and never on the host."
            )

        try:
            if target_container:
                # Execute inside running container
                cmd = ["docker", "exec", target_container, "sh", "-c", test_command]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                )
            elif docker_available:
                # Execute inside temporary container mounted to sandbox root
                mount_path = os.path.abspath(self.sandbox_root).replace("\\", "/")
                cmd = [
                    "docker", "run", "--rm",
                    "-v", f"{mount_path}:/workspace",
                    "-w", "/workspace",
                    "--network", "none",
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges",
                    "alpine:latest",
                    "sh", "-c", test_command,
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    shell=False,
                )
            else:
                # Explicit allow_host mode (e.g. for mock unit tests)
                proc = subprocess.run(
                    test_command,
                    shell=True,
                    cwd=self.sandbox_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

            passed = proc.returncode == 0
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            # Extract failure summary lines if tests failed
            failure_summary = ""
            if not passed:
                error_lines = [
                    l for l in (stdout + "\n" + stderr).splitlines()
                    if any(kw in l for kw in ("FAILED", "ERROR", "AssertionError", "Traceback", "Exception", "SyntaxError"))
                ]
                failure_summary = "\n".join(error_lines[:30])

            return {
                "test_command": test_command,
                "exit_code": proc.returncode,
                "passed": passed,
                "failure_summary": failure_summary[:4096],
                "stdout_trimmed": stdout[-8192:] if len(stdout) > 8192 else stdout,
                "stderr_trimmed": stderr[-4096:] if len(stderr) > 4096 else stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "test_command": test_command,
                "exit_code": -1,
                "passed": False,
                "failure_summary": f"Test run timed out after {timeout} seconds.",
                "stdout_trimmed": "",
                "stderr_trimmed": f"Timed out after {timeout}s",
                "timed_out": True,
            }


class GitDiffTool(BaseTool):
    """Produces git diff output for modified files within the repository."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(
            name="git_diff",
            description="Returns git diff output of changes made in the repository.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        subpath = args.get("path")
        staged = bool(args.get("staged", False))

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        if subpath:
            valid_path = _resolve_sandbox_path(subpath, self.sandbox_root)
            cmd.append(os.path.relpath(valid_path, self.sandbox_root))

        try:
            res = subprocess.run(
                cmd,
                cwd=self.sandbox_root,
                capture_output=True,
                text=True,
                timeout=15.0,
                shell=False,
            )
            diff_text = res.stdout or ""
            truncated = len(diff_text) > 32768
            return {
                "diff": diff_text[:32768],
                "truncated": truncated,
                "exit_code": res.returncode,
            }
        except Exception as e:
            return {
                "diff": "",
                "error": str(e),
                "exit_code": 1,
            }


class GitStatusTool(BaseTool):
    """Returns structured git status (staged, unstaged, untracked)."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(
            name="git_status",
            description="Returns porcelain git status of modified, staged, and untracked files.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.sandbox_root,
                capture_output=True,
                text=True,
                timeout=10.0,
                shell=False,
            )
            staged = []
            unstaged = []
            untracked = []

            for line in res.stdout.splitlines():
                if len(line) < 3:
                    continue
                x, y = line[0], line[1]
                filepath = line[3:].strip()
                if x == "?" and y == "?":
                    untracked.append(filepath)
                else:
                    if x in ("M", "A", "D", "R"):
                        staged.append({"status": x, "file": filepath})
                    if y in ("M", "D"):
                        unstaged.append({"status": y, "file": filepath})

            return {
                "clean": len(staged) == 0 and len(unstaged) == 0 and len(untracked) == 0,
                "staged": staged,
                "unstaged": unstaged,
                "untracked": untracked,
            }
        except Exception as e:
            return {"error": str(e), "clean": False}


class ListDirectoryTool(BaseTool):
    """Lists files and directories with depth and entry count limits."""

    def __init__(self, sandbox_root: Optional[str] = None, default_limit: int = 100):
        super().__init__(
            name="list_directory",
            description="Lists contents of a directory in the sandbox with depth and count bounds.",
        )
        self.sandbox_root = sandbox_root or os.getcwd()
        self.default_limit = default_limit

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        subpath = args.get("path", ".")
        max_depth = min(int(args.get("max_depth", 2)), 5)
        limit = min(int(args.get("limit", self.default_limit)), 200)

        target_dir = _resolve_sandbox_path(subpath, self.sandbox_root)
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Directory not found: {subpath}")

        files_list = []
        dirs_list = []
        truncated = False

        start_depth = target_dir.rstrip(os.sep).count(os.sep)

        for root, dirs, files in os.walk(target_dir):
            curr_depth = root.rstrip(os.sep).count(os.sep) - start_depth
            if curr_depth >= max_depth:
                dirs[:] = []
                continue

            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git")]

            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), self.sandbox_root)
                dirs_list.append(rel.replace("\\", "/"))
                if len(dirs_list) + len(files_list) >= limit:
                    truncated = True
                    break

            for f in files:
                rel = os.path.relpath(os.path.join(root, f), self.sandbox_root)
                files_list.append(rel.replace("\\", "/"))
                if len(dirs_list) + len(files_list) >= limit:
                    truncated = True
                    break

            if truncated:
                break

        return {
            "path": subpath,
            "directories": sorted(dirs_list),
            "files": sorted(files_list),
            "total_items": len(dirs_list) + len(files_list),
            "truncated": truncated,
        }
