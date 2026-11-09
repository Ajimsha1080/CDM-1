"""Layer 4: Sandboxed File Operations Tools for CDM-1.
Guarantees sandbox path jail enforcement, preventing path traversal attacks.
"""

from typing import Dict, Any, Optional
import os
from cdm.execution.tools.base import BaseTool


def _resolve_sandbox_path(rel_or_abs_path: str, sandbox_root: str) -> str:
    """Resolves path and verifies it strictly resides within the sandboxed directory."""
    root_canon = os.path.realpath(sandbox_root)
    full_path = os.path.realpath(os.path.join(root_canon, rel_or_abs_path))
    if not (full_path == root_canon or full_path.startswith(root_canon + os.sep)):
        raise PermissionError(f"Access denied: path '{rel_or_abs_path}' escapes sandbox root '{sandbox_root}'.")
    return full_path


class FileReadTool(BaseTool):
    """Sandboxed file reading tool."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(name="read_file", description="Reads text contents of a file within sandbox.")
        self.sandbox_root = sandbox_root or os.getcwd()

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        base_err = super().validate_args(args)
        if base_err:
            return base_err
        if not args.get("path") or not isinstance(args.get("path"), str):
            return "Parameter 'path' is required and must be a non-empty string."
        return None

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        if not path:
            raise ValueError("Parameter 'path' is required.")
        target = _resolve_sandbox_path(path, self.sandbox_root)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: {path}")

        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(65536) # Max 64KB
        return {"path": path, "content": content, "size_bytes": os.path.getsize(target)}


class FileWriteTool(BaseTool):
    """Sandboxed file writing tool."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(name="write_file", description="Writes text content to a file within sandbox.")
        self.sandbox_root = sandbox_root or os.getcwd()

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        base_err = super().validate_args(args)
        if base_err:
            return base_err
        if not args.get("path") or not isinstance(args.get("path"), str):
            return "Parameter 'path' is required and must be a non-empty string."
        return None

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        content = args.get("content", "")
        if not path:
            raise ValueError("Parameter 'path' is required.")
        target = _resolve_sandbox_path(path, self.sandbox_root)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path, "bytes_written": len(content.encode("utf-8")), "success": True}


class FileEditTool(BaseTool):
    """Sandboxed file search-and-replace tool."""

    def __init__(self, sandbox_root: Optional[str] = None):
        super().__init__(name="edit_file", description="Replaces occurrences of text in a file within sandbox.")
        self.sandbox_root = sandbox_root or os.getcwd()

    def validate_args(self, args: Dict[str, Any]) -> Optional[str]:
        base_err = super().validate_args(args)
        if base_err:
            return base_err
        if not args.get("path") or not isinstance(args.get("path"), str):
            return "Parameter 'path' is required and must be a non-empty string."
        if args.get("find") is None or args.get("replace") is None:
            return "Parameters 'find' and 'replace' are required."
        return None

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        path = args.get("path")
        old_str = args.get("find")
        new_str = args.get("replace")
        if not path or old_str is None or new_str is None:
            raise ValueError("Parameters 'path', 'find', and 'replace' are required.")
        target = _resolve_sandbox_path(path, self.sandbox_root)
        if not os.path.isfile(target):
            raise FileNotFoundError(f"File not found: {path}")

        with open(target, "r", encoding="utf-8") as f:
            orig = f.read()

        if old_str not in orig:
            raise ValueError(f"Target string not found in {path}")

        updated = orig.replace(old_str, new_str)
        with open(target, "w", encoding="utf-8") as f:
            f.write(updated)
        return {"path": path, "replacements": orig.count(old_str), "success": True}
