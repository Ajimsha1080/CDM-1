"""Planning and Goal Intelligence Subsystem for CDM-1 Cognitive AI Engine.
Manages goal decomposition, multi-step action sequencing, constraint-aware plan search,
prospective plan evaluation against World Model state transitions, and replanning triggers.
"""

from typing import Dict, List, Optional, Any, Tuple
from cdm.cognitive.state import Plan, PlanStep, GoalNode, WorldState, ReasoningTrace
from cdm.world_model.world_model import WorldModel


class PlanningEngine:
    """Multi-step action sequencing, plan evaluation, and goal accomplishment engine."""

    def __init__(self, world_model: Optional[WorldModel] = None):
        self.world_model = world_model or WorldModel()

    def generate_plan(
        self,
        target_goal: GoalNode,
        available_actions: List[str],
        current_world_state: WorldState,
        reasoning_trace: ReasoningTrace,
        prohibited_actions: Optional[List[str]] = None,
    ) -> Plan:
        """Constructs and evaluates a multi-step plan to achieve the target goal while adhering to constraints."""
        forbidden = set(prohibited_actions or [])
        # Also extract prohibited actions flagged by reasoning
        for v in reasoning_trace.violated_constraints:
            for act in available_actions:
                if act.lower() in v.lower():
                    forbidden.add(act)

        valid_actions = [a for a in available_actions if a not in forbidden]

        plan_steps: List[PlanStep] = []
        cumulative_risk = 0.0

        # Sequence actions towards goal
        for idx, action in enumerate(valid_actions[:4], start=1):
            # Simulate forward transition consequence in World Model
            simulated_next_state = self.world_model.simulate_transition(action)
            step_risk = 0.15 if "escalate" in action.lower() else (0.25 if "refund" in action.lower() else 0.05)
            cumulative_risk += step_risk

            step = PlanStep(
                step_number=idx,
                action_name=action,
                parameters={"target_goal": target_goal.goal_id},
                expected_outcome={
                    "state_delta": simulated_next_state.environment_metrics,
                },
                preconditions=[f"constraint_clear_{action}"],
                postconditions=[f"action_executed_{action}"],
                risk_estimate=step_risk,
                is_reversible="refund" not in action.lower(),
            )
            plan_steps.append(step)

        efficiency = max(0.2, 1.0 - (len(plan_steps) * 0.1))
        constraint_compliance = len(forbidden.intersection(set(valid_actions))) == 0

        # Generate alternative plan option
        alternatives = []
        if len(valid_actions) > 1:
            alt_plan_summary = {
                "name": "Minimal Intervention Plan",
                "first_action": valid_actions[-1],
                "expected_risk": round(cumulative_risk * 0.7, 3),
            }
            alternatives.append(alt_plan_summary)

        return Plan(
            target_goal_id=target_goal.goal_id,
            steps=plan_steps,
            cumulative_risk=round(cumulative_risk, 3),
            plan_efficiency_score=round(efficiency, 2),
            constraint_compliance=constraint_compliance,
            alternative_plans=alternatives,
        )

    def should_replan(self, current_plan: Plan, current_world_state: WorldState) -> Tuple[bool, Optional[str]]:
        """Evaluates whether current execution conditions have diverged, necessitating replanning."""
        if not current_plan.steps:
            return True, "Current plan has no remaining steps."

        # Check if any hard constraint in world state now forbids the upcoming action
        next_action = current_plan.steps[0].action_name
        for constraint in current_world_state.active_constraints:
            if next_action.lower() in constraint.lower() and ("prohibit" in constraint.lower() or "forbidden" in constraint.lower()):
                return True, f"Planned action '{next_action}' is now prohibited by updated world state constraint: {constraint}"

        # Check if environment metrics deteriorated unexpectedly
        if current_world_state.environment_metrics.get("system_exposure", 0.0) > 0.8:
            return True, "Critical security exposure detected in world state; immediate emergency replan required."

        return False, None
