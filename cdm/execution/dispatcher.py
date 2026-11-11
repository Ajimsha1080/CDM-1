"""Plan Dispatcher and Execution Safety Controller for CDM-1.
Guarantees execution safety gates:
1. Verify dispatcher only executes approved plans.
2. Reject proposed, abstained, prohibited, and expired plans.
3. Require explicit authorization for sensitive actions.
4. Detect and block attempts to bypass the planning and approval flow.
"""

from typing import Dict, List, Optional, Any, Callable
import time
import logging

from cdm.cognitive.state import Plan, PlanStep, ExecutedActionStep, WorldState

logger = logging.getLogger("cdm.execution.dispatcher")


class PlanExecutionError(RuntimeError):
    """Raised when an execution safety rule or approval requirement is violated."""
    pass


# Actions with financial, physiological, security, or irreversible infrastructure impact
SENSITIVE_ACTIONS = {
    "issue_refund",
    "refund_customer",
    "refund_transaction",
    "administer_meds",
    "isolate_cluster",
    "terminate_service",
    "delete_database",
    "modify_access_policy",
    "shell_exec",
}

AUTHORIZED_ROLES = {"security_admin", "compliance_officer", "authorized_manager", "lead_physician"}


class PlanDispatcher:
    """Production execution controller enforcing safety invariants on plan dispatch."""

    def __init__(
        self,
        sensitive_actions: Optional[set] = None,
        authorized_roles: Optional[set] = None,
    ):
        self.sensitive_actions = sensitive_actions if sensitive_actions is not None else set(SENSITIVE_ACTIONS)
        self.authorized_roles = authorized_roles if authorized_roles is not None else set(AUTHORIZED_ROLES)

    def dispatch_plan(
        self,
        plan: Plan,
        executor_role: str = "system",
        authorization_token: Optional[str] = None,
        prohibited_actions: Optional[List[str]] = None,
        current_time: Optional[float] = None,
    ) -> Plan:
        """Validates and transitions an approved plan into dispatched stage.
        Strictly rejects proposed, abstained, prohibited, or expired plans.
        """
        if plan is None:
            raise PlanExecutionError("Cannot dispatch null plan.")

        now = current_time if current_time is not None else time.time()

        # 1. Reject proposed, abstained, prohibited, expired, or rejected plans
        current_stage = (plan.plan_stage or "").lower().strip()
        current_status = (getattr(plan, "status", "") or "").lower().strip()

        if current_stage == "proposed" or current_status == "proposed":
            raise PlanExecutionError(
                "Execution Safety Violation: Attempted to dispatch unapproved plan in 'proposed' stage. "
                "Autonomous or manual execution strictly requires explicit plan approval before dispatch."
            )

        if current_stage == "abstained" or current_status == "abstained":
            raise PlanExecutionError(
                "Execution Safety Violation: Attempted to dispatch plan marked 'abstained'. "
                "Abstained plans cannot be executed without re-planning and explicit resolution."
            )

        if current_stage == "prohibited" or current_status == "prohibited":
            raise PlanExecutionError(
                "Execution Safety Violation: Attempted to dispatch plan marked 'prohibited'."
            )

        if current_stage == "expired" or current_status == "expired":
            raise PlanExecutionError(
                "Execution Safety Violation: Attempted to dispatch plan marked 'expired'."
            )

        if current_stage != "approved" and current_status != "approved":
            raise PlanExecutionError(
                f"Execution Safety Violation: Cannot dispatch plan in invalid stage '{current_stage}'. "
                "Only plans in 'approved' stage may be dispatched."
            )

        # 2. Check for plan expiration
        if plan.expires_at is not None and now > plan.expires_at:
            plan.plan_stage = "expired"
            plan.status = "expired"
            raise PlanExecutionError(
                f"Execution Safety Violation: Plan '{plan.plan_id}' expired at {plan.expires_at} (current time: {now})."
            )

        # 3. Check for prohibited actions in planned steps
        prohibited = set(prohibited_actions or [])
        for step in plan.steps:
            if step.action_name in prohibited:
                plan.plan_stage = "prohibited"
                plan.status = "prohibited"
                raise PlanExecutionError(
                    f"Execution Safety Violation: Step {step.step_number} action '{step.action_name}' is explicitly prohibited."
                )

        # 4. Require explicit authorization for sensitive actions
        token = authorization_token or plan.authorization_token
        for step in plan.steps:
            if step.action_name in self.sensitive_actions:
                has_valid_token = bool(token and token.startswith("AUTH-"))
                is_explicitly_authorized = step.action_name in plan.authorized_actions
                is_authorized_role = (plan.approved_by in self.authorized_roles) or (executor_role in self.authorized_roles)

                if not (has_valid_token or is_explicitly_authorized or is_authorized_role):
                    raise PlanExecutionError(
                        f"Execution Safety Violation: Sensitive action '{step.action_name}' at step {step.step_number} "
                        "requires explicit authorization (valid AUTH-* token, explicit authorized_action registration, "
                        f"or approval by one of {sorted(list(self.authorized_roles))})."
                    )

        # Invariants satisfied: advance lifecycle stage
        plan.plan_stage = "dispatched"
        plan.status = "dispatched"
        logger.info(f"Plan '{plan.plan_id}' safely dispatched for execution.")
        return plan

    def execute_dispatched_step(
        self,
        plan: Plan,
        step_number: int,
        action_fn: Callable[[PlanStep], Dict[str, Any]],
    ) -> ExecutedActionStep:
        """Executes a single step of a dispatched plan with ordering and lifecycle enforcement."""
        if plan.plan_stage not in ("dispatched", "in_progress"):
            raise PlanExecutionError(
                f"Execution Safety Violation: Cannot execute step for plan in '{plan.plan_stage}' stage. "
                "Step execution requires a plan that has passed dispatch."
            )

        expected_step = len(plan.executed_steps) + 1
        if step_number != expected_step:
            raise PlanExecutionError(
                f"Execution Safety Violation: Step order violation. Expected step {expected_step}, got {step_number}."
            )

        matching_step = next((s for s in plan.steps if s.step_number == step_number), None)
        if not matching_step:
            raise PlanExecutionError(f"Step {step_number} not found in plan '{plan.plan_id}'.")

        plan.plan_stage = "in_progress"
        plan.status = "in_progress"

        outcome = action_fn(matching_step)

        exec_step = ExecutedActionStep(
            step_number=step_number,
            action_name=matching_step.action_name,
            executed_at=time.time(),
            execution_status=outcome.get("status", "success"),
            observed_outcome=outcome,
        )
        plan.executed_steps.append(exec_step)
        plan.observed_outcomes.append(outcome)

        if len(plan.executed_steps) == len(plan.steps):
            plan.plan_stage = "completed"
            plan.status = "completed"

        return exec_step
