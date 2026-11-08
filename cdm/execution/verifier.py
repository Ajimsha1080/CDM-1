"""Layer 4: Ground-Truth Outcome Verification Subsystem for CDM-1.
After each action, verifies the actual physical or state outcome against task specifications.
Explicitly refuses to accept the LLM's self-reported claim of success.
"""

from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
import os


class VerificationSpec(BaseModel):
    check_type: str = Field(..., description="'file_exists', 'file_contains', 'exit_code_zero', 'state_equals', 'json_field_equals'")
    target: str = Field(..., description="File path, metric name, or command output key")
    expected_value: Optional[Any] = Field(default=None, description="Expected literal value")


class VerificationResult(BaseModel):
    verified: bool
    check_type: str
    details: Dict[str, Any] = Field(default_factory=dict)
    failure_reason: Optional[str] = None


class OutcomeVerifier:
    """Verifies actual execution outcomes independently of LLM assertions."""

    @classmethod
    def verify(
        cls,
        spec: VerificationSpec,
        execution_output: Optional[Any] = None,
        world_state_metrics: Optional[Dict[str, Any]] = None,
        sandbox_root: Optional[str] = None,
    ) -> VerificationResult:
        """Runs concrete ground-truth verification check."""
        root = sandbox_root or os.getcwd()
        check = spec.check_type.lower()

        # 1. Physical file existence
        if check == "file_exists":
            full_path = os.path.join(root, spec.target)
            exists = os.path.exists(full_path)
            return VerificationResult(
                verified=exists,
                check_type=check,
                details={"target": spec.target, "exists": exists},
                failure_reason=None if exists else f"File does not exist on disk: {spec.target}",
            )

        # 2. File content inspection
        elif check == "file_contains":
            full_path = os.path.join(root, spec.target)
            if not os.path.isfile(full_path):
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    details={"target": spec.target, "file_found": False},
                    failure_reason=f"File not found: {spec.target}",
                )
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                contains = str(spec.expected_value) in content
                return VerificationResult(
                    verified=contains,
                    check_type=check,
                    details={"target": spec.target, "expected": str(spec.expected_value), "matched": contains},
                    failure_reason=None if contains else f"Pattern '{spec.expected_value}' not found in {spec.target}",
                )
            except Exception as e:
                return VerificationResult(
                    verified=False,
                    check_type=check,
                    details={"error": str(e)},
                    failure_reason=f"Failed to read file: {e}",
                )

        # 3. Process exit code
        elif check == "exit_code_zero":
            if isinstance(execution_output, dict):
                code = execution_output.get("exit_code")
                is_zero = (code == 0)
                return VerificationResult(
                    verified=is_zero,
                    check_type=check,
                    details={"exit_code": code},
                    failure_reason=None if is_zero else f"Command exited with non-zero status code: {code}",
                )
            return VerificationResult(
                verified=False,
                check_type=check,
                details={},
                failure_reason="Execution output did not contain an exit code.",
            )

        # 4. State variable equality
        elif check == "state_equals":
            metrics = world_state_metrics or {}
            val = metrics.get(spec.target)
            matches = (val == spec.expected_value)
            return VerificationResult(
                verified=matches,
                check_type=check,
                details={"target": spec.target, "actual": val, "expected": spec.expected_value},
                failure_reason=None if matches else f"State metric '{spec.target}' was {val}, expected {spec.expected_value}",
            )

        # 5. JSON response field equality
        elif check == "json_field_equals":
            if isinstance(execution_output, dict):
                val = execution_output.get(spec.target)
                matches = (val == spec.expected_value)
                return VerificationResult(
                    verified=matches,
                    check_type=check,
                    details={"target": spec.target, "actual": val, "expected": spec.expected_value},
                    failure_reason=None if matches else f"Field '{spec.target}' in result was {val}, expected {spec.expected_value}",
                )
            return VerificationResult(
                verified=False,
                check_type=check,
                details={},
                failure_reason="Execution output is not a JSON dictionary.",
            )

        return VerificationResult(
            verified=False,
            check_type=check,
            details={},
            failure_reason=f"Unsupported verification check type: {spec.check_type}",
        )
