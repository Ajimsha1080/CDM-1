"""Planning and Goal Intelligence Subsystem for CDM-1 Cognitive AI Engine.
Manages goal decomposition, multi-step action sequencing, constraint-aware plan search,
prospective plan evaluation against World Model state transitions, preconditions,
risk estimates, alternative plan generation, and dynamic replanning triggers.
"""

from typing import Dict, List, Optional, Any, Tuple
from cdm.cognitive.state import Plan, PlanStep, GoalNode, WorldState, ReasoningTrace
from cdm.world_model.world_model import WorldModel


class PlanningEngine:
    """Multi-step action sequencing, precondition validation, risk evaluation, and alternative plan generation."""

    def __init__(self, world_model: Optional[WorldModel] = None):
        self.world_model = world_model or WorldModel()

    def _compute_action_risk(
        self,
        action: str,
        current_world_state: WorldState,
        predicted_outcome: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str]:
        """Computes dynamic risk derived from learned outcome predictions vs world state uncertainty heuristics.
        Returns a tuple of (risk_value, risk_source) where risk_source explicitly distinguishes
        'trained_model_prediction' from 'world_state_uncertainty_heuristic'.
        """
        state_unc = float(getattr(current_world_state, "overall_state_uncertainty", 0.05))

        # 1. Extract learned continuous risk prediction if available
        pred_risk = None
        if predicted_outcome:
            pred_risk = (
                predicted_outcome.get("predicted_risk")
                or predicted_outcome.get("risk")
                or predicted_outcome.get("action_risk", {}).get(action)
            )
            if pred_risk is None and "candidate_scores" in predicted_outcome:
                cand_scores = predicted_outcome["candidate_scores"]
                if action in cand_scores:
                    norm_score = max(0.0, min(1.0, cand_scores[action]))
                    pred_risk = round(1.0 - norm_score, 4)

        if pred_risk is None:
            pred_fut = getattr(current_world_state, "predicted_future_state", {})
            if isinstance(pred_fut, dict):
                pred_risk = pred_fut.get("predicted_risk")

        if pred_risk is not None:
            combined = round(float(pred_risk) * 0.7 + state_unc * 0.3, 4)
            return combined, "trained_model_prediction"

        # 2. Dynamic environmental risk calculation derived from observed facts & metrics heuristic
        monetary = float(current_world_state.environment_metrics.get("monetary_value_usd", 50.0))
        sla = float(current_world_state.environment_metrics.get("sla_remaining_hours", 24.0))
        env_pressure = min(1.0, (monetary / 250.0) * 0.5 + max(0.0, 1.0 - sla / 48.0) * 0.5)
        heuristic_risk = round(0.05 + 0.4 * env_pressure + 0.3 * state_unc, 4)
        return heuristic_risk, "world_state_uncertainty_heuristic"

    def _evaluate_preconditions(
        self,
        action: str,
        current_world_state: WorldState,
        forbidden_actions: set,
    ) -> Tuple[List[str], bool]:
        """Evaluates domain and state preconditions, returning defined preconditions and satisfied boolean."""
        preconditions = [f"constraint_clear_{action}"]
        satisfied = action not in forbidden_actions

        monetary = float(current_world_state.environment_metrics.get("monetary_value_usd", 0.0))
        if "refund" in action.lower():
            preconditions.append("policy_refund_limit_verified")
            # If monetary value exceeds refund cap, precondition fails
            if monetary > 1500.0:
                satisfied = False
        elif "escalate" in action.lower():
            preconditions.append("tier1_triage_completed")
        elif "auto_resolve" in action.lower():
            preconditions.append("uncontested_standard_issue")

        return preconditions, satisfied

    def generate_plan(
        self,
        target_goal: GoalNode,
        available_actions: List[str],
        current_world_state: WorldState,
        reasoning_trace: ReasoningTrace,
        prohibited_actions: Optional[List[str]] = None,
        predicted_outcome: Optional[Dict[str, Any]] = None,
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

        # Sequence actions towards goal with preconditions and counterfactual effects
        for idx, action in enumerate(valid_actions[:4], start=1):
            preconditions, preconditions_met = self._evaluate_preconditions(
                action=action,
                current_world_state=current_world_state,
                forbidden_actions=forbidden,
            )

            # If precondition fails, skip or mark blocked
            if not preconditions_met:
                continue

            # Simulate forward transition consequence in World Model
            simulated_next_state = self.world_model.simulate_transition(action)

            # Determine dynamic risk estimate and its provenance source
            step_risk, step_risk_src = self._compute_action_risk(
                action=action,
                current_world_state=current_world_state,
                predicted_outcome=predicted_outcome,
            )
            cumulative_risk += step_risk

            step = PlanStep(
                step_number=len(plan_steps) + 1,
                action_name=action,
                parameters={"target_goal": target_goal.goal_id},
                expected_outcome={
                    "state_delta": simulated_next_state.environment_metrics,
                    "simulated_transition": simulated_next_state.simulated_state.get(f"transition_{action}", {}),
                },
                preconditions=preconditions,
                postconditions=[f"action_executed_{action}", f"goal_advanced_{target_goal.goal_id}"],
                risk_estimate=step_risk,
                risk_source=step_risk_src,
                is_reversible="refund" not in action.lower(),
            )
            plan_steps.append(step)

        efficiency = max(0.2, 1.0 - (len(plan_steps) * 0.1))
        constraint_compliance = len(forbidden.intersection(set(valid_actions))) == 0

        # Generate and evaluate alternative plans using predicted outcomes
        alternatives = []
        if len(valid_actions) > 1:
            for alt_act in valid_actions[1:]:
                alt_score = 0.5
                if predicted_outcome and "candidate_scores" in predicted_outcome:
                    alt_score = predicted_outcome["candidate_scores"].get(alt_act, 0.5)
                alt_risk, alt_risk_src = self._compute_action_risk(
                    action=alt_act,
                    current_world_state=current_world_state,
                    predicted_outcome=predicted_outcome,
                )
                alternatives.append({
                    "name": f"Alternative Branch: {alt_act}",
                    "first_action": alt_act,
                    "predicted_outcome_score": round(float(alt_score), 4),
                    "expected_risk": alt_risk,
                    "risk_source": alt_risk_src,
                    "rationale": f"Contingency plan executing '{alt_act}' evaluated with prospective outcome forecasting.",
                })

        plan_risk_sources = [s.risk_source for s in plan_steps]
        dominant_risk_source = (
            "trained_model_prediction"
            if any(src == "trained_model_prediction" for src in plan_risk_sources)
            else "world_state_uncertainty_heuristic"
        )
        goal_satisfied = (len(plan_steps) > 0) and constraint_compliance

        return Plan(
            target_goal_id=target_goal.goal_id,
            steps=plan_steps,
            cumulative_risk=round(cumulative_risk, 3),
            risk_source=dominant_risk_source,
            plan_efficiency_score=round(efficiency, 2),
            constraint_compliance=constraint_compliance,
            goal_satisfied=goal_satisfied,
            alternative_plans=alternatives,
        )

    def should_replan(self, current_plan: Plan, current_world_state: WorldState) -> Tuple[bool, Optional[str]]:
        """Evaluates whether current execution conditions have diverged, necessitating replanning."""
        if not current_plan.steps:
            return True, "Current plan has no remaining steps."

        # Check if any hard constraint in world state now forbids the upcoming action
        next_action = current_plan.steps[0].action_name
        for constraint in current_world_state.active_constraints:
            if next_action.lower() in constraint.lower() and (
                "prohibit" in constraint.lower() or "forbidden" in constraint.lower()
            ):
                return True, f"Planned action '{next_action}' is now prohibited by updated world state constraint: {constraint}"

        # Check if environment metrics deteriorated unexpectedly
        if current_world_state.environment_metrics.get("system_exposure", 0.0) > 0.8:
            return True, "Critical security exposure detected in world state; immediate emergency replan required."

        return False, None
