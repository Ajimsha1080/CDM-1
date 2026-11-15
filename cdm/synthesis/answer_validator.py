"""Answer Validation Subsystem for CDM-1 Cognitive AI Engine.
Performs rigorous consistency checks between natural-language answers and structured
cognitive results. Ensures safety abstentions are strictly preserved, verifies decision
alignment, detects unsupported claims, and prevents fabricated verification claims.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from cdm.cognitive.state import CognitiveEngineResult


class AnswerValidationResult(BaseModel):
    """Structured report verifying consistency of the natural language answer."""
    valid: bool
    decision_consistent: bool
    safety_preserved: bool
    execution_status_accurate: bool
    verification_accurate: bool
    approval_requirement_accurate: bool = True
    abstention_communicated: bool = True
    unsupported_claims_detected: bool
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class AnswerValidator:
    """Validates synthesized natural-language outputs against structured pipeline state."""

    def __init__(self):
        pass

    def validate(
        self,
        answer_text: str,
        cognitive_result: CognitiveEngineResult,
        unsupported_claims: Optional[List[str]] = None,
    ) -> AnswerValidationResult:
        """Validates that answer_text does not contradict or fabricate cognitive_result fields."""
        violations: List[str] = []
        warnings: List[str] = []

        ans_lower = answer_text.lower()
        proposed = str(cognitive_result.proposed_decision or "").lower()
        abstained = cognitive_result.abstain
        authorized = cognitive_result.execution_authorized
        exec_status = cognitive_result.execution_status
        verified = cognitive_result.verification.get("verified", False)

        # 1. Decision Consistency Check
        decision_consistent = True
        if proposed and proposed not in ("none", "abstain"):
            # If a decision is proposed and not abstained, check that answer doesn't recommend a completely different action
            if not abstained and exec_status == "authorized":
                if proposed not in ans_lower and "recommended" in ans_lower:
                    decision_consistent = False
                    violations.append(
                        f"Decision inconsistency: Structured decision is '{proposed}', but answer does not cite this recommendation."
                    )

        # 2. Safety Abstention Preservation
        safety_preserved = True
        if abstained or exec_status in ("abstained", "rejected"):
            # Check that answer does NOT claim authorized execution
            if "authorized for execution" in ans_lower or "execution is authorized" in ans_lower:
                safety_preserved = False
                violations.append(
                    "Safety violation: Cognitive engine abstained, but answer falsely states execution is authorized."
                )
            if "successfully executed" in ans_lower or "has been executed" in ans_lower:
                safety_preserved = False
                violations.append(
                    "Safety violation: Action was not authorized or dispatched, but answer claims execution succeeded."
                )

        # 3. Execution Status Accuracy
        execution_status_accurate = True
        if not authorized:
            if "authorized" in ans_lower and "not authorized" not in ans_lower and "unauthorized" not in ans_lower and "requires" not in ans_lower and "pending" not in ans_lower:
                execution_status_accurate = False
                violations.append(
                    "Execution status mismatch: execution_authorized=False but answer states authorized."
                )

        # 4. Verification Accuracy Check
        verification_accurate = True
        if not verified and ("verification passed" in ans_lower or "safely verified" in ans_lower):
            verification_accurate = False
            violations.append(
                "Verification claim fabrication: Safety verification was not confirmed, but answer claims verification passed."
            )

        # 5. Human Approval Requirement Check
        approval_requirement_accurate = True
        appr_dict = cognitive_result.approval_requirements if isinstance(cognitive_result.approval_requirements, dict) else {}
        requires_approval = bool(appr_dict.get("requires_human_approval") or appr_dict.get("required"))
        if requires_approval:
            # If human approval is required, answer cannot claim unconditional authorization
            if ("authorized for execution" in ans_lower or "ready for execution" in ans_lower) and "approval" not in ans_lower:
                approval_requirement_accurate = False
                violations.append(
                    "Approval requirement violation: Human approval is mandatory, but answer claims action is ready/authorized without approval."
                )

        # 6. Abstention Clear Communication Check
        abstention_communicated = True
        if abstained or exec_status in ("abstained", "rejected"):
            has_abstention_cue = any(
                cue in ans_lower
                for cue in (
                    "abstain", "abstained", "policy", "cannot", "prohibited",
                    "uncertainty", "not authorized", "rejected", "risk",
                    "escalat", "manual intervention", "pending approval",
                    "unable to execute", "prohibit"
                )
            )
            if not has_abstention_cue:
                abstention_communicated = False
                violations.append(
                    "Abstention communication failure: Engine abstained or rejected candidate, but answer fails to state abstention or policy constraints."
                )

        # 7. Unsupported Claims Check
        unsupported_detected = bool(unsupported_claims and len(unsupported_claims) > 0)
        if unsupported_detected:
            for unc in unsupported_claims:
                warnings.append(f"Potentially unsupported claim without direct citation: '{unc[:80]}...'")

        is_valid = (
            decision_consistent
            and safety_preserved
            and execution_status_accurate
            and verification_accurate
            and approval_requirement_accurate
            and abstention_communicated
            and len(violations) == 0
        )

        return AnswerValidationResult(
            valid=is_valid,
            decision_consistent=decision_consistent,
            safety_preserved=safety_preserved,
            execution_status_accurate=execution_status_accurate,
            verification_accurate=verification_accurate,
            approval_requirement_accurate=approval_requirement_accurate,
            abstention_communicated=abstention_communicated,
            unsupported_claims_detected=unsupported_detected,
            violations=violations,
            warnings=warnings,
        )
