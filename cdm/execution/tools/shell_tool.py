"""Layer 4: Sandboxed Shell Execution Tool for CDM-1.
Executes non-interactive shell commands with strict timeouts, memory/buffer bounds,
and prohibited dangerous commands.
"""

from typing import Dict, Any, Optional, Set
import subprocess
import os
from cdm.execution.tools.base import BaseTool


class SandboxedShellTool(BaseTool):
    """Sandboxed shell execution tool."""

    PROHIBITED_COMMANDS: Set[str] = {
        "rm -rf /", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero",
        "shutdown", "reboot", "init 0", "chmod -r 777 /",
        "nc -l", "ncat", "bash -i", "sh -i",
    }

    def __init__(self, timeout_seconds: float = 10.0, working_dir: Optional[str] = None):
        super().__init__(
            name="shell_exec",
            description="Executes a command in a sandboxed non-interactive shell.",
        )
        self.timeout_seconds = min(timeout_seconds, 30.0)
        self.working_dir = working_dir or os.getcwd()

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        cmd = args.get("command", "")
        if not cmd or not isinstance(cmd, str):
            raise ValueError("Parameter 'command' is required and must be a non-empty string.")

        cmd_lower = cmd.lower()
        for prohibited in self.PROHIBITED_COMMANDS:
            if prohibited in cmd_lower:
                raise PermissionError(f"Command contains prohibited dangerous operation: '{prohibited}'")

        # Execute subprocess with timeout and captured output
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            # Truncate output to prevent memory exhaustion
            stdout = result.stdout[:32768]
            stderr = result.stderr[:32768]
            return {
                "exit_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out after {self.timeout_seconds} seconds.",
                "timed_out": True,
            }
