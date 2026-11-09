"""Layer 4: Ground-Truth Outcome Verification Subsystem for CDM-1.
After each action, verifies the actual physical or state outcome against task specifications.
Explicitly refuses to accept the LLM's self-reported claim of success.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
from pydantic import BaseModel, Field
import os
import time
from enum import Enum
from pathlib import Path


class ExecutionOutcomeState(str, Enum):
    """The four distinct execution lifecycle states."""
    ATTEMPTED = "attempted"                           # Action initiated / dispatched to sandbox
    EXECUTED = "executed"                             # Tool executed with output returned, unverified
    ACTION_RESULT_VERIFIED = "action_result_verified"  # Physical action postcondition independently verified
    TASK_GOAL_VERIFIED = "task_goal_verified"          # End-to-end task goal independently verified


class VerificationStatus:
    PENDING = "pending"
    RUNNING = "running"
    ATTEMPTED = "attempted"
    EXECUTED = "executed"
    ACTION_SUCCEEDED = "action_succeeded"
    ACTION_RESULT_VERIFIED = "action_result_verified"
    VERIFICATION_PENDING = "verification_pending"
    VERIFIED_COMPLETE = "verified_complete"
    TASK_GOAL_VERIFIED = "task_goal_verified"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    REQUIRES_HUMAN_APPROVAL = "requires_human_approval"


class VerificationSpec(BaseModel):
    check_type: str = Field(..., description="'file_exists', 'file_contains', 'exit_code_zero', 'state_equals', 'json_field_equals'")
    target: str = Field(..., description="File path, metric name, or command output key")
    expected_value: Optional[Any] = Field(default=None, description="Expected literal value")
    scope: str = Field(default="goal", description="'action' for per-action postconditions or 'goal' for task completion criteria")
    postconditions: Optional[List[str]] = Field(default=None, description="Optional action postconditions to verify")


class VerificationResult(BaseModel):
    verified: bool
    check_type: str
    status: str = Field(default=VerificationStatus.VERIFIED_COMPLETE)
    details: Dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)


class OutcomeVerifier:
    """Verifies actual execution outcomes independently of LLM assertions."""

    VERIFIER_VERSION = "1.4.0"

    @classmethod
    def classify_execution_state(
        cls,
        attempted: bool = True,
        executed: bool = False,
        action_verified: bool = False,
        goal_verified: bool = False,
    ) -> ExecutionOutcomeState:
        """Determines the exact execution state across the 4-tier lifecycle hierarchy:
        attempted -> executed -> action_result_verified -> task_goal_verified.
        """
        if goal_verified:
            return ExecutionOutcomeState.TASK_GOAL_VERIFIED
        if action_verified:
            return ExecutionOutcomeState.ACTION_RESULT_VERIFIED
        if executed:
            return ExecutionOutcomeState.EXECUTED
        return ExecutionOutcomeState.ATTEMPTED

    @classmethod
    def resolve_sandbox_path(cls, target: str, sandbox_root: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """Resolves target path within sandbox_root and ensures strict containment.
        Returns (canonical_path, error_reason). If path escapes sandbox, returns (None, error).
        """
        if not target or not isinstance(target, str):
            return None, "Target path is empty or invalid."

        raw_root = sandbox_root or os.getcwd()
        try:
            root_path = Path(raw_root).resolve(strict=False)
        except Exception as e:
            return None, f"Failed to resolve sandbox root: {e}"

        # 1. Reject traversal patterns
        normalized_target = target.replace("\\", "/")
        if "../" in normalized_target or normalized_target == ".." or normalized_target.startswith("../") or "/.." in normalized_target:
            return None, f"Security violation: path traversal ('..') detected in target '{target}'."

        target_obj = Path(target)
        try:
            if target_obj.is_absolute():
                resolved_target = target_obj.resolve(strict=False)
            else:
                resolved_target = (root_path / target_obj).resolve(strict=False)
        except Exception as e:
            return None, f"Failed to resolve target path: {e}"

        # 2. Strict containment check
        try:
            resolved_target.relative_to(root_path)
        except ValueError:
            return None, f"Security violation: target path '{target}' escapes sandbox root '{root_path}'."

        # 3. Prevent symlink escapes
        curr = resolved_target
        while curr != root_path and curr != curr.parent:
            if curr.is_symlink():
                try:
                    symlink_target = curr.resolve()
                    symlink_target.relative_to(root_path)
                except ValueError:
                    return None, f"Security violation: symlink '{curr}' points outside sandbox root '{root_path}'."
            curr = curr.parent

        return str(resolved_target), None

    @classmethod
    def verify(
        cls,
        spec: VerificationSpec,
        execution_output: Optional[Any] = None,
        world_state_metrics: Optional[Dict[str, Any]] = None,
        sandbox_root: Optional[str] = None,
    ) -> VerificationResult:
        """Runs concrete ground-truth verification check."""
        check = spec.check_type.lower()
        prov = {
            "verifier_version": cls.VERIFIER_VERSION,
            "timestamp": time.time(),
            "check_type": check,
            "target": spec.target,
            "scope": getattr(spec, "scope", "goal"),
        }

        # 1. Physical file existence
        if check == "file_exists":
            full_path, path_err = cls.resolve_sandbox_path(spec.target, sandbox_root)
            if path_err:
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    status=VerificationStatus.FAILED,
                    details={"target": spec.target, "security_violation": True},
                    failure_reason=path_err,
                    provenance=prov,
                )
            exists = os.path.exists(full_path)
            return VerificationResult(
                verified=exists,
                check_type=check,
                status=VerificationStatus.VERIFIED_COMPLETE if exists else VerificationStatus.FAILED,
                details={"target": spec.target, "exists": exists, "canonical_path": full_path},
                failure_reason=None if exists else f"File does not exist on disk: {spec.target}",
                provenance=prov,
            )

        # 2. File content inspection
        elif check == "file_contains":
            full_path, path_err = cls.resolve_sandbox_path(spec.target, sandbox_root)
            if path_err:
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    status=VerificationStatus.FAILED,
                    details={"target": spec.target, "security_violation": True},
                    failure_reason=path_err,
                    provenance=prov,
                )
            if not os.path.isfile(full_path):
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    status=VerificationStatus.FAILED,
                    details={"target": spec.target, "file_found": False},
                    failure_reason=f"File not found: {spec.target}",
                    provenance=prov,
                )
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                contains = str(spec.expected_value) in content
                return VerificationResult(
                    verified=contains,
                    check_type=check,
                    status=VerificationStatus.VERIFIED_COMPLETE if contains else VerificationStatus.FAILED,
                    details={"target": spec.target, "expected": str(spec.expected_value), "matched": contains},
                    failure_reason=None if contains else f"Pattern '{spec.expected_value}' not found in {spec.target}",
                    provenance=prov,
                )
            except Exception as e:
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    status=VerificationStatus.FAILED,
                    details={"error": str(e)},
                    failure_reason=f"Failed to read file: {e}",
                    provenance=prov,
                )

        # 3. Process exit code
        elif check == "exit_code_zero":
            if isinstance(execution_output, dict):
                code = execution_output.get("exit_code")
                is_zero = (code == 0)
                return VerificationResult(
                    verified=is_zero,
                    check_type=check,
                    status=VerificationStatus.ACTION_SUCCEEDED if is_zero else VerificationStatus.FAILED,
                    details={"exit_code": code},
                    failure_reason=None if is_zero else f"Command exited with non-zero status code: {code}",
                    provenance=prov,
                )
            return VerificationResult(
                verified=False,
                check_type=check,
                status=VerificationStatus.FAILED,
                details={},
                failure_reason="Execution output did not contain an exit code.",
                provenance=prov,
            )

        # 4. State variable equality
        elif check == "state_equals":
            metrics = world_state_metrics or {}
            val = metrics.get(spec.target)
            matches = (val == spec.expected_value)
            return VerificationResult(
                verified=matches,
                check_type=check,
                status=VerificationStatus.VERIFIED_COMPLETE if matches else VerificationStatus.FAILED,
                details={"target": spec.target, "actual": val, "expected": spec.expected_value},
                failure_reason=None if matches else f"State metric '{spec.target}' was {val}, expected {spec.expected_value}",
                provenance=prov,
            )

        # 5. JSON response field equality
        elif check == "json_field_equals":
            if isinstance(execution_output, dict):
                val = execution_output.get(spec.target)
                matches = (val == spec.expected_value)
                return VerificationResult(
                    verified=matches,
                    check_type=check,
                    status=VerificationStatus.VERIFIED_COMPLETE if matches else VerificationStatus.FAILED,
                    details={"target": spec.target, "actual": val, "expected": spec.expected_value},
                    failure_reason=None if matches else f"Field '{spec.target}' in result was {val}, expected {spec.expected_value}",
                    provenance=prov,
                )
            return VerificationResult(
                verified=False,
                check_type=check,
                status=VerificationStatus.FAILED,
                details={},
                failure_reason="Execution output is not a JSON dictionary.",
                provenance=prov,
            )

        return VerificationResult(
            verified=False,
            check_type=check,
            status=VerificationStatus.INCONCLUSIVE,
            details={},
            failure_reason=f"Unsupported verification check type: {spec.check_type}",
            provenance=prov,
        )
