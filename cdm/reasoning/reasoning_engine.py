"""Reasoning Intelligence Subsystem for CDM-1 Cognitive AI Engine.
Processes evidence, evaluates deterministic policy constraints, performs multi-step
deductions, handles contradictions, and generates traceable proof graphs.
"""

from typing import Dict, List, Optional, Any, Tuple
from cdm.cognitive.state import ReasoningTrace, ReasoningStep, WorldState, ContextFrame


class ReasoningEngine:
    """Evidence integration, multi-step deduction, and constraint satisfaction engine."""

    def __init__(self):
        pass

    def reason(
        self,
        world_state: WorldState,
        context: ContextFrame,
        semantic_rules: List[Dict[str, Any]],
        candidate_actions: List[str],
        user_constraints: Optional[List[str]] = None,
        procedural_workflows: Optional[List[Any]] = None,
        causal_consequences: Optional[Dict[str, Any]] = None,
    ) -> ReasoningTrace:
        """Executes multi-step deduction, verifies constraints, and builds a traceable reasoning proof graph."""
        steps: List[ReasoningStep] = []
        satisfied_constraints: List[str] = []
        violated_constraints: List[str] = []
        contradictions_found: List[str] = list(context.detected_contradictions)

        active_rules = semantic_rules or []
        user_specified = user_constraints or []

        step_idx = 1

        # Step 1: Ingest world state metrics and entity facts
        metrics_summary = ", ".join(f"{k}={v}" for k, v in world_state.environment_metrics.items())
        steps.append(
            ReasoningStep(
                step_index=step_idx,
                rule_or_premise="Premise: World State Observation",
                inputs=list(world_state.environment_metrics.keys()),
                conclusion=f"Observed environment metrics: {metrics_summary or 'Standard Baseline'}",
                confidence=1.0,
                is_hard_constraint=False,
            )
        )
        step_idx += 1

        # Step 1b: Ingest procedural workflows if available
        if procedural_workflows:
            for wf in procedural_workflows:
                name = getattr(wf, "name", str(wf))
                preconds = getattr(wf, "preconditions", [])
                wf_steps = getattr(wf, "steps", [])
                steps.append(
                    ReasoningStep(
                        step_index=step_idx,
                        rule_or_premise=f"Procedural Runbook: {name}",
                        inputs=[str(p) for p in preconds] if preconds else ["default_preconditions"],
                        conclusion=f"Applicable procedure '{name}' contains {len(wf_steps)} recommended operational steps.",
                        confidence=0.95,
                        is_hard_constraint=False,
                    )
                )
                step_idx += 1

        # Step 1c: Ingest causal consequences & side-effects
        if causal_consequences:
            for cand, consequence in causal_consequences.items():
                if isinstance(consequence, dict):
                    risk_lvl = consequence.get("expected_risk", 0.0)
                    side_effects = consequence.get("side_effects", [])
                    if risk_lvl > 0.6 or side_effects:
                        steps.append(
                            ReasoningStep(
                                step_index=step_idx,
                                rule_or_premise=f"Causal Mechanism Analysis: {cand}",
                                inputs=[cand],
                                conclusion=f"Causal analysis predicts elevated risk ({risk_lvl}) or side-effects: {side_effects}",
                                confidence=0.88,
                                is_hard_constraint=False,
                            )
                        )
                        step_idx += 1

        # Step 2: Evaluate deterministic policy constraints from Semantic Rules
        for rule in active_rules:
            if not rule.get("active", True):
                continue
            rule_id = rule.get("rule_id", "rule_general")
            condition = rule.get("condition", {})
            prohibs = []
            if rule.get("prohibited_action"):
                prohibs.append(rule["prohibited_action"])
            if isinstance(rule.get("prohibited_actions"), list):
                for p in rule["prohibited_actions"]:
                    if p not in prohibs:
                        prohibs.append(p)

            # Check condition matching against world metrics
            condition_met = True
            if condition:
                for cond_k, cond_v in condition.items():
                    actual_v = world_state.environment_metrics.get(cond_k)
                    if actual_v is not None and isinstance(cond_v, (int, float)):
                        if actual_v < cond_v:
                            condition_met = False
                            break

            if condition_met and prohibs:
                for action_restriction in prohibs:
                    violated_constraints.append(f"Policy Violation: {action_restriction} is forbidden by {rule_id}")
                    steps.append(
                        ReasoningStep(
                            step_index=step_idx,
                            rule_or_premise=f"Rule {rule_id}: {rule.get('condition_description') or rule.get('description', 'Policy Gate')}",
                            inputs=[f"{k}={v}" for k, v in condition.items()] if condition else ["domain_policy"],
                            conclusion=f"Action '{action_restriction}' is strictly PROHIBITED under active policy.",
                            confidence=1.0,
                            is_hard_constraint=True,
                        )
                    )
                    step_idx += 1
            else:
                satisfied_constraints.append(f"Policy Passed: {rule_id}")

        # Step 3: Evaluate explicit user constraints
        for user_c in user_specified:
            for cand in candidate_actions:
                if cand.lower() in user_c.lower() and ("prohibit" in user_c.lower() or "forbid" in user_c.lower() or "no" in user_c.lower()):
                    violated_constraints.append(f"User Constraint Violation: {cand} is restricted by instruction '{user_c}'")
                    steps.append(
                        ReasoningStep(
                            step_index=step_idx,
                            rule_or_premise=f"User Constraint: {user_c}",
                            inputs=[cand],
                            conclusion=f"Action '{cand}' is excluded per caller constraint.",
                            confidence=1.0,
                            is_hard_constraint=True,
                        )
                    )
                    step_idx += 1

        # Calculate overall reasoning confidence
        reasoning_conf = 1.0
        if contradictions_found:
            reasoning_conf *= 0.85
        if world_state.overall_state_uncertainty > 0.5:
            reasoning_conf *= 0.90

        return ReasoningTrace(
            steps=steps,
            satisfied_constraints=satisfied_constraints,
            violated_constraints=violated_constraints,
            contradictions_found=contradictions_found,
            reasoning_confidence=round(reasoning_conf, 4),
        )
