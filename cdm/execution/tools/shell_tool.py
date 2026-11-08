"""Layer 4: Real Container-Sandboxed Shell Execution Tool for CDM-1.
Executes non-interactive commands strictly inside an isolated Docker container:
- Zero host execution: strictly requires Docker; never falls back to host
- Disabled by default; requires explicit configuration
- --network none (no outbound or inbound networking)
- --read-only root filesystem
- Hard resource bounds (memory, cpu, pids-limit)
- Non-root execution user (1000:1000)
- Writable tmpfs /tmp or mounted workspace
- Hard timeout that forcibly kills the container
- Argument lists only (zero shell=True invocation)
"""

from typing import Dict, Any, Optional
import subprocess
import os
import uuid
from cdm.execution.tools.base import BaseTool


class SandboxedShellTool(BaseTool):
    """Container-sandboxed shell execution tool."""

    def __init__(
        self,
        enabled: Optional[bool] = None,
        timeout_seconds: float = 10.0,
        working_dir: Optional[str] = None,
        image: Optional[str] = None,
        memory_limit: str = "128m",
        cpu_limit: float = 1.0,
        pids_limit: int = 64,
        user: str = "1000:1000",
    ):
        super().__init__(
            name="shell_exec",
            description="Executes a command strictly inside an isolated Docker sandbox container.",
        )
        if enabled is None:
            self.enabled = os.getenv("CDM_SHELL_TOOL_ENABLED", "false").lower() in ("true", "1", "yes")
        else:
            self.enabled = bool(enabled)

        self.timeout_seconds = min(float(timeout_seconds), 30.0)
        self.working_dir = working_dir
        self.image = image or os.getenv("CDM_SANDBOX_IMAGE", "alpine:latest")
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.pids_limit = pids_limit
        self.user = user

    def is_docker_available(self) -> bool:
        """Verify if Docker CLI and daemon are accessible."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3.0,
                shell=False,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        if not self.enabled:
            raise PermissionError(
                "Shell execution tool is disabled by default. "
                "Enable via explicit configuration (enabled=True or CDM_SHELL_TOOL_ENABLED=true)."
            )

        if not self.is_docker_available():
            raise RuntimeError(
                "Docker is unavailable or daemon is not running. "
                "Shell execution strictly requires container isolation and refuses to execute on the host."
            )

        cmd = args.get("command", "")
        if not cmd or not isinstance(cmd, str):
            raise ValueError("Parameter 'command' is required and must be a non-empty string.")

        # Generate unique container name for lifecycle tracking and hard kill
        container_id = f"cdm-sandbox-{uuid.uuid4().hex[:12]}"

        docker_args = [
            "docker", "run",
            "--name", container_id,
            "--rm",
            "--network", "none",
            "--read-only",
            "--memory", self.memory_limit,
            f"--cpus={self.cpu_limit}",
            f"--pids-limit={self.pids_limit}",
            "--user", self.user,
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
        ]

        if self.working_dir and os.path.exists(self.working_dir):
            abs_dir = os.path.abspath(self.working_dir)
            docker_args.extend(["-v", f"{abs_dir}:/workspace:rw", "-w", "/workspace"])
        else:
            docker_args.extend(["-w", "/tmp"])

        # Execute using argument lists only - no shell=True
        docker_args.extend([self.image, "sh", "-c", cmd])

        try:
            proc = subprocess.run(
                docker_args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout[:32768],
                "stderr": proc.stderr[:32768],
                "timed_out": False,
                "sandboxed": True,
            }
        except subprocess.TimeoutExpired:
            # Force kill the container by name
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=5.0, shell=False)
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Execution timed out and container was forcibly terminated after {self.timeout_seconds}s.",
                "timed_out": True,
                "sandboxed": True,
            }
