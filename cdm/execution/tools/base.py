"""Layer 4: Base Execution Tool Interface for CDM-1 Agent Loop.
All tools return structured ToolResult envelopes and guarantee never to raise exceptions
into the control loop.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import time
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger("cdm.execution.tools")


class ToolResult(BaseModel):
    success: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    tool_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseTool(ABC):
    """Abstract sandboxed tool."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def _run(self, args: Dict[str, Any], tenant_id: str) -> Any:
        """Internal execution method. Can raise exceptions that are caught safely by execute()."""
        pass

    def execute(self, args: Dict[str, Any], tenant_id: str = "default_tenant") -> ToolResult:
        """Safe tool execution envelope catching all errors."""
        start_time = time.time()
        try:
            output = self._run(args, tenant_id)
            latency_ms = (time.time() - start_time) * 1000.0
            return ToolResult(
                success=True,
                output=output,
                latency_ms=round(latency_ms, 2),
                tool_name=self.name,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            logger.error(f"Error executing tool '{self.name}': {e}", exc_info=False)
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                latency_ms=round(latency_ms, 2),
                tool_name=self.name,
            )
