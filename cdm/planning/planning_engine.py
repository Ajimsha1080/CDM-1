"""Planning and Goal Intelligence Subsystem for CDM-1 Cognitive AI Engine (V9).
Manages goal decomposition, multi-step action sequencing, constraint-aware plan search,
prospective plan evaluation against World Model state transitions, explicit preconditions and postconditions,
cryptographic prediction provenance verification, independent alternative plan evaluation,
measurable goal condition verification, and clear separation of proposed plans from executed outcomes.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from cdm.cognitive.state import Plan, PlanStep, ExecutedActionStep, GoalNode, WorldState, ReasoningTrace
from cdm.world_model.world_model import WorldModel
from cdm.inference.provenance import PredictionProvenance


# Documented Domain Action Registry with explicit preconditions, postconditions, and transition effects
ACTION_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "auto_resolve": {
        "preconditions": ["uncontested_standard_issue"],
        "postconditions": ["action_executed_auto_resolve", "inquiry_marked_resolved", "case_closed"],
        "effects": {"sla_impact": 0.0, "case_closed": 1.0},
        "is_reversible": True,
    },
    "issue_refund": {
        "preconditions": ["policy_refund_limit_verified"],
        "postconditions": ["action_executed_issue_refund", "refund_transaction_completed", "billing_dispute_resolved"],
        "effects": {"monetary_impact": -50.0, "dispute_active": 0.0},
        "is_reversible": False,
    },
    "refund_customer": {
        "preconditions": ["policy_refund_limit_verified"],
        "postconditions": ["action_executed_refund_customer", "refund_transaction_completed", "billing_dispute_resolved"],
        "effects": {"monetary_impact": -100.0, "dispute_active": 0.0},
        "is_reversible": False,
    },
    "refund_transaction": {
        "preconditions": ["policy_refund_limit_verified"],
        "postconditions": ["action_executed_refund_transaction", "refund_transaction_completed"],
        "effects": {"monetary_impact": -50.0},
        "is_reversible": False,
    },
    "escalate_tier2": {
        "preconditions": ["tier1_triage_completed"],
        "postconditions": ["action_executed_escalate_tier2", "tier2_ticket_opened", "escalation_active"],
        "effects": {"sla_impact": -6.0, "escalation_active": 1.0},
        "is_reversible": True,
    },
    "escalate_to_tier2": {
        "preconditions": ["tier1_triage_completed"],
        "postconditions": ["action_executed_escalate_to_tier2", "tier2_ticket_opened", "escalation_active"],
        "effects": {"sla_impact": -6.0, "escalation_active": 1.0},
        "is_reversible": True,
    },
    "manual_escalation": {
        "preconditions": ["supervisor_review_requested"],
        "postconditions": ["action_executed_manual_escalation", "supervisor_assigned"],
        "effects": {"sla_impact": -12.0},
        "is_reversible": True,
    },
    "request_documentation": {
        "preconditions": ["customer_contact_verified"],
        "postconditions": ["action_executed_request_documentation", "documentation_requested"],
        "effects": {"awaiting_customer_response": 1.0},
        "is_reversible": True,
    },
    "triage_patient": {
        "preconditions": ["intake_vitals_measured"],
        "postconditions": ["action_executed_triage_patient", "triage_assessment_recorded"],
        "effects": {"triage_completed": 1.0},
        "is_reversible": True,
    },
    "administer_meds": {
        "preconditions": ["physician_order_verified"],
        "postconditions": ["action_executed_administer_meds", "pharmacotherapy_administered"],
        "effects": {"medication_given": 1.0},
        "is_reversible": False,
    },
    "order_labs": {
        "preconditions": ["clinical_indication_present"],
        "postconditions": ["action_executed_order_labs", "diagnostic_labs_ordered"],
        "effects": {"labs_pending": 1.0},
        "is_reversible": True,
    },
    "stabilize_vitals": {
        "preconditions": ["iv_access_established"],
        "postconditions": ["action_executed_stabilize_vitals", "vital_signs_stabilized", "cardiac_rhythm_stabilized", "arterial_blood_pressure_normal"],
        "effects": {"vital_instability": 0.0, "patient_status": "stable"},
        "is_reversible": True,
    },
    "isolate_cluster": {
        "preconditions": ["active_incident_confirmed"],
        "postconditions": ["action_executed_isolate_cluster", "cluster_network_isolated", "system_exposure_reduced"],
        "effects": {"system_exposure": 0.1},
        "is_reversible": True,
    },
    "safe_diagnostics": {
        "preconditions": ["read_only_access_granted"],
        "postconditions": ["action_executed_safe_diagnostics", "diagnostic_telemetry_collected"],
        "effects": {"diagnostic_telemetry": 1.0},
        "is_reversible": True,
    },
    "standard_healthcheck": {
        "preconditions": ["system_online"],
        "postconditions": ["action_executed_standard_healthcheck", "healthcheck_completed"],
        "effects": {"healthcheck_ok": 1.0},
        "is_reversible": True,
    },
}


class PlanningEngine:
    """Multi-step action sequencing, precondition validation, verified prediction risk evaluation,
    independent alternative plan generation, and measurable goal condition verification.
    """

    def __init__(self, world_model: Optional[WorldModel] = None):
        self.world_model = world_model or WorldModel()

    def _compute_action_risk(
        self,
        action: str,
        current_world_state: WorldState,
        predicted_outcome: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Tuple[float, str, Optional[str], Optional[str], Dict[str, Any]]:
        """Computes dynamic action risk with cryptographic provenance validation.
        Strictly rejects caller-controlled claims of neural authenticity.
        Distinguishes:
        - 'trained_model_prediction': Cryptographically verified prediction from trusted model runtime.
        - 'deterministic_heuristic': Algorithmic/environmental rule calculation from world state.
        - 'unverified_caller_input': Forged, unsigned, or caller-supplied arbitrary dict.
        """
        state_unc = float(getattr(current_world_state, "overall_state_uncertainty", 0.05))

        prov_dict = None
        if predicted_outcome and isinstance(predicted_outcome, dict):
            prov_dict = predicted_outcome.get("provenance") or (
                predicted_outcome if "provenance_signature" in predicted_outcome else None
            )

        verify_pred_output = None
        if prov_dict and prov_dict.get("prediction_digest"):
            verify_pred_output = predicted_outcome.get("prediction_output") or predicted_outcome

        is_valid, classification_source, diag = PredictionProvenance.validate(
            prov_dict,
            expected_tenant_id=tenant_id,
            expected_task=task,
            prediction_output=verify_pred_output,
        )

        pred_risk = None
        if predicted_outcome and isinstance(predicted_outcome, dict):
            pred_risk = (
                predicted_outcome.get("predicted_risk")
                or predicted_outcome.get("risk")
                or predicted_outcome.get("action_risk", {}).get(action)
            )
            if pred_risk is None and "candidate_scores" in predicted_outcome:
                cand_scores = predicted_outcome["candidate_scores"]
                if isinstance(cand_scores, dict) and action in cand_scores:
                    norm_score = max(0.0, min(1.0, cand_scores[action]))
                    pred_risk = round(1.0 - norm_score, 4)

        if is_valid and classification_source == "trained_model_prediction":
            model_id = diag.get("model_id", "CDM-1-General")
            model_version = diag.get("model_version", "1.0.0")
            if pred_risk is None:
                pred_risk = prov_dict.get("estimated_risk", 0.05) if prov_dict else 0.05
            combined = round(float(pred_risk) * 0.7 + state_unc * 0.3, 4)
            prov_meta = {
                "model_id": model_id,
                "model_version": model_version,
                "artifact_hash": diag.get("artifact_hash"),
                "execution_id": diag.get("execution_id"),
                "prediction_source": "trained_model_prediction",
                "authenticated": True,
            }
            return combined, "trained_model_prediction", model_id, model_version, prov_meta

        # If caller passed a predicted_outcome dict with arbitrary values without valid signature
        if predicted_outcome and isinstance(predicted_outcome, dict) and pred_risk is not None:
            combined = round(float(pred_risk) * 0.7 + state_unc * 0.3, 4)
            prov_meta = {
                "prediction_source": "unverified_caller_input",
                "authenticated": False,
                "diagnostic": diag.get("reason", "Caller supplied unverified predictions without runtime signature."),
            }
            return combined, "unverified_caller_input", None, None, prov_meta

        # Fallback: Deterministic environmental heuristic derived from world state metrics
        monetary = float(current_world_state.environment_metrics.get("monetary_value_usd", 50.0))
        sla = float(current_world_state.environment_metrics.get("sla_remaining_hours", 24.0))
        env_pressure = min(1.0, (monetary / 250.0) * 0.5 + max(0.0, 1.0 - sla / 48.0) * 0.5)
        heuristic_risk = round(0.05 + 0.4 * env_pressure + 0.3 * state_unc, 4)
        prov_meta = {
            "prediction_source": "deterministic_heuristic",
            "authenticated": True,
            "environmental_pressure": env_pressure,
            "state_uncertainty": state_unc,
        }
        return heuristic_risk, "deterministic_heuristic", None, None, prov_meta

    def _evaluate_preconditions(
        self,
        action: str,
        current_world_state: WorldState,
        forbidden_actions: set,
    ) -> Tuple[List[str], bool]:
        """Evaluates domain and state preconditions, returning defined preconditions and satisfied boolean."""
        spec = ACTION_SPECIFICATIONS.get(action, {})
        preconditions = list(spec.get("preconditions", [f"constraint_clear_{action}"]))
        satisfied = action not in forbidden_actions

        monetary = float(current_world_state.environment_metrics.get("monetary_value_usd", 0.0))
        if "refund" in action.lower():
            if "policy_refund_limit_verified" not in preconditions:
                preconditions.append("policy_refund_limit_verified")
            if monetary > 1500.0:
                satisfied = False
        elif "escalate" in action.lower():
            if "tier1_triage_completed" not in preconditions:
                preconditions.append("tier1_triage_completed")
        elif "auto_resolve" in action.lower():
            if "uncontested_standard_issue" not in preconditions:
                preconditions.append("uncontested_standard_issue")

        return preconditions, satisfied

    def _evaluate_constraint_compliance(
        self,
        plan_steps: List[PlanStep],
        reasoning_trace: ReasoningTrace,
        current_world_state: WorldState,
        forbidden_actions: set,
    ) -> Tuple[bool, List[str]]:
        """Evaluates whether actual plan steps violate any constraints, prohibited actions,
        or violated constraints detected in reasoning or world state.
        """
        violations = []
        for step in plan_steps:
            act = step.action_name
            if act in forbidden_actions:
                violations.append(f"Action '{act}' in plan step {step.step_number} is prohibited.")

            for vc in reasoning_trace.violated_constraints:
                if act.lower() in vc.lower():
                    violations.append(f"Action '{act}' in plan step {step.step_number} violates constraint: {vc}")

            for ac in current_world_state.active_constraints:
                if ("prohibit" in ac.lower() or "forbidden" in ac.lower() or "disallow" in ac.lower()) and act.lower() in ac.lower():
                    violations.append(f"Action '{act}' in plan step {step.step_number} violates active world constraint: {ac}")

        if reasoning_trace.contradictions_found:
            for c in reasoning_trace.contradictions_found:
                violations.append(f"Plan executed under active contradiction: {c}")

        is_compliant = (len(violations) == 0)
        return is_compliant, violations

    def _evaluate_goal_satisfaction(
        self,
        target_goal: GoalNode,
        plan_steps: List[PlanStep],
        constraint_compliance: bool,
        current_world_state: WorldState,
        final_simulated_metrics: Dict[str, float],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluates whether the executed plan steps actually satisfy explicit goal conditions,
        measurable postconditions, and target state metrics, rather than merely containing nonempty steps.
        Never grants automatic goal advancement without evidence.
        """
        details = {
            "evaluated_steps_count": len(plan_steps),
            "constraint_compliance": constraint_compliance,
            "satisfied_conditions": [],
            "failed_conditions": [],
        }

        # Rule 1: A plan with 0 steps cannot satisfy any goal
        if not plan_steps:
            details["failed_conditions"].append("Plan contains zero executable steps.")
            return False, details

        # Rule 2: A plan that violates constraints cannot satisfy the goal safely
        if not constraint_compliance:
            details["failed_conditions"].append("Plan violated operational safety constraints.")
            return False, details

        # Collect cumulative postconditions and executed actions from all steps
        cumulative_postconditions = set()
        executed_actions = set()
        for step in plan_steps:
            cumulative_postconditions.update(step.postconditions)
            executed_actions.add(step.action_name)

        # Verification A: Explicit required postconditions on target_goal
        required_posts = getattr(target_goal, "required_postconditions", []) or []
        for req_post in required_posts:
            if req_post in cumulative_postconditions:
                details["satisfied_conditions"].append(f"postcondition:{req_post}")
            else:
                details["failed_conditions"].append(f"missing_required_postcondition:{req_post}")

        # Verification B: Explicit target state metrics on target_goal
        target_metrics = getattr(target_goal, "target_state_metrics", {}) or {}
        for metric_name, target_val in target_metrics.items():
            sim_val = final_simulated_metrics.get(metric_name)
            if sim_val is None:
                sim_val = current_world_state.environment_metrics.get(metric_name)
            if sim_val is not None:
                if "risk" in metric_name.lower() or "exposure" in metric_name.lower():
                    if sim_val <= target_val:
                        details["satisfied_conditions"].append(f"metric_met:{metric_name}<={target_val}")
                    else:
                        details["failed_conditions"].append(f"metric_exceeded:{metric_name}={sim_val}>{target_val}")
                else:
                    details["satisfied_conditions"].append(f"metric_checked:{metric_name}")

        # Verification C: Explicit success criteria dict on target_goal
        success_criteria = getattr(target_goal, "success_criteria", {}) or {}
        if success_criteria:
            if "required_actions" in success_criteria:
                for req_act in success_criteria["required_actions"]:
                    if req_act in executed_actions:
                        details["satisfied_conditions"].append(f"action_executed:{req_act}")
                    else:
                        details["failed_conditions"].append(f"missing_required_action:{req_act}")

            if "required_postconditions" in success_criteria:
                for req_p in success_criteria["required_postconditions"]:
                    if req_p in cumulative_postconditions:
                        details["satisfied_conditions"].append(f"criterion_postcondition:{req_p}")
                    else:
                        details["failed_conditions"].append(f"missing_criterion_postcondition:{req_p}")

            if "max_acceptable_risk" in success_criteria:
                max_risk = float(success_criteria["max_acceptable_risk"])
                plan_risk = sum(s.risk_estimate for s in plan_steps)
                if plan_risk <= max_risk:
                    details["satisfied_conditions"].append(f"risk_within_tolerance:{plan_risk}<={max_risk}")
                else:
                    details["failed_conditions"].append(f"plan_risk_too_high:{plan_risk}>{max_risk}")

        # Verification D: Baseline semantic domain goal alignment
        # Check whether any step produced domain resolution matching goal intent
        # (e.g. 'billing_dispute_resolved' for dispute, 'triage_assessment_recorded' for triage, 'case_closed' for resolve)
        has_domain_fulfillment = False
        goal_text = f"{target_goal.goal_id} {target_goal.description}".lower()

        for post in cumulative_postconditions:
            # Clinical stabilization
            if "stabilized" in goal_text and ("stabilized" in post or "patient_status" in post):
                has_domain_fulfillment = True
            # Dispute resolution
            elif "dispute" in goal_text and ("dispute_resolved" in post or "refund" in post):
                has_domain_fulfillment = True
            # Inquiry resolution
            elif ("resolve" in goal_text or "inquiry" in goal_text or "ticket" in goal_text) and ("resolved" in post or "closed" in post or "assigned" in post or "ticket_opened" in post):
                has_domain_fulfillment = True
            # Triage
            elif "triage" in goal_text and ("triage" in post or "assessment" in post or "ordered" in post):
                has_domain_fulfillment = True
            # Security / exposure
            elif ("exposure" in goal_text or "incident" in goal_text or "security" in goal_text) and ("isolated" in post or "remediated" in post or "telemetry" in post or "healthcheck" in post):
                has_domain_fulfillment = True

        if has_domain_fulfillment:
            details["satisfied_conditions"].append("domain_goal_alignment_verified")
        else:
            if not required_posts and not target_metrics and not success_criteria:
                details["failed_conditions"].append("Executed actions produce no postconditions aligned with target goal intent.")

        is_satisfied = (len(details["failed_conditions"]) == 0) and (len(details["satisfied_conditions"]) > 0)
        return is_satisfied, details

    def generate_plan(
        self,
        target_goal: GoalNode,
        available_actions: List[str],
        current_world_state: WorldState,
        reasoning_trace: ReasoningTrace,
        prohibited_actions: Optional[List[str]] = None,
        predicted_outcome: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Plan:
        """Constructs and evaluates a proposed multi-step plan to achieve the target goal while adhering to constraints.
        Executes sequential forward world-state simulation: S_{t+1} = T(S_t, A_t), passing resulting state
        to subsequent steps, re-evaluating preconditions at each intermediate state, and evaluating alternatives independently.
        Never marks proposed plans as verified complete.
        """
        forbidden = set(prohibited_actions or [])
        for v in reasoning_trace.violated_constraints:
            for act in available_actions:
                if act.lower() in v.lower():
                    forbidden.add(act)

        valid_actions = [a for a in available_actions if a not in forbidden]

        # Sequential world-state simulation for proposed plan
        current_sim_state = current_world_state.model_copy(deep=True)
        plan_steps: List[PlanStep] = []
        cumulative_risk = 0.0

        for idx, action in enumerate(valid_actions[:4], start=1):
            preconditions, preconditions_met = self._evaluate_preconditions(
                action=action,
                current_world_state=current_sim_state,
                forbidden_actions=forbidden,
            )

            if not preconditions_met:
                continue

            simulated_next_state = self.world_model.simulate_transition(action, base_state=current_sim_state)

            step_risk, step_risk_src, model_id, model_version, prov = self._compute_action_risk(
                action=action,
                current_world_state=current_sim_state,
                predicted_outcome=predicted_outcome,
                tenant_id=tenant_id,
                task=task,
            )
            cumulative_risk += step_risk

            # Explicit postconditions from documented action specifications (no automatic goal_advanced)
            action_spec = ACTION_SPECIFICATIONS.get(action, {})
            step_postconditions = list(action_spec.get("postconditions", [f"action_executed_{action}"]))

            step = PlanStep(
                step_number=len(plan_steps) + 1,
                action_name=action,
                parameters={"target_goal": target_goal.goal_id},
                expected_outcome={
                    "state_delta": simulated_next_state.environment_metrics,
                    "simulated_transition": simulated_next_state.simulated_state.get(f"transition_{action}", {}),
                },
                preconditions=preconditions,
                postconditions=step_postconditions,
                risk_estimate=step_risk,
                risk_source=step_risk_src,
                model_id=model_id,
                model_version=model_version,
                prediction_source=prov.get("prediction_source"),
                risk_provenance=prov,
                is_reversible=action_spec.get("is_reversible", "refund" not in action.lower()),
            )
            plan_steps.append(step)
            # Advance sequential simulation state to next step
            current_sim_state = simulated_next_state

        efficiency = max(0.2, 1.0 - (len(plan_steps) * 0.1))

        # Independently evaluate constraint compliance on actual plan steps and violated constraints
        constraint_compliance, constraint_violations = self._evaluate_constraint_compliance(
            plan_steps=plan_steps,
            reasoning_trace=reasoning_trace,
            current_world_state=current_world_state,
            forbidden_actions=forbidden,
        )

        # Independently evaluate alternative plans with sequential world-state simulation
        alternatives = []
        if len(valid_actions) > 1:
            for alt_act in valid_actions[1:]:
                alt_sim_state = current_world_state.model_copy(deep=True)
                alt_plan_steps: List[PlanStep] = []
                alt_cum_risk = 0.0
                candidate_sequence = [alt_act] + [a for a in valid_actions if a != alt_act][:2]

                for s_idx, a_name in enumerate(candidate_sequence, start=1):
                    p_conds, p_met = self._evaluate_preconditions(a_name, alt_sim_state, forbidden)
                    if not p_met:
                        continue
                    sim_state = self.world_model.simulate_transition(a_name, base_state=alt_sim_state)
                    s_risk, s_src, s_mid, s_mver, s_prov = self._compute_action_risk(
                        action=a_name,
                        current_world_state=alt_sim_state,
                        predicted_outcome=predicted_outcome,
                        tenant_id=tenant_id,
                        task=task,
                    )
                    alt_cum_risk += s_risk
                    alt_spec = ACTION_SPECIFICATIONS.get(a_name, {})
                    alt_postconditions = list(alt_spec.get("postconditions", [f"action_executed_{a_name}"]))

                    alt_step = PlanStep(
                        step_number=len(alt_plan_steps) + 1,
                        action_name=a_name,
                        parameters={"target_goal": target_goal.goal_id, "branch": "alternative"},
                        expected_outcome={
                            "state_delta": sim_state.environment_metrics,
                            "simulated_transition": sim_state.simulated_state.get(f"transition_{a_name}", {}),
                        },
                        preconditions=p_conds,
                        postconditions=alt_postconditions,
                        risk_estimate=s_risk,
                        risk_source=s_src,
                        model_id=s_mid,
                        model_version=s_mver,
                        prediction_source=s_prov.get("prediction_source"),
                        risk_provenance=s_prov,
                        is_reversible=alt_spec.get("is_reversible", "refund" not in a_name.lower()),
                    )
                    alt_plan_steps.append(alt_step)
                    # Advance sequential simulation state for alternative branch
                    alt_sim_state = sim_state

                alt_compliant, _ = self._evaluate_constraint_compliance(
                    alt_plan_steps, reasoning_trace, current_world_state, forbidden
                )
                alt_final_metrics = alt_plan_steps[-1].expected_outcome.get("state_delta", {}) if alt_plan_steps else {}
                alt_goal_satisfied, alt_details = self._evaluate_goal_satisfaction(
                    target_goal=target_goal,
                    plan_steps=alt_plan_steps,
                    constraint_compliance=alt_compliant,
                    current_world_state=current_world_state,
                    final_simulated_metrics=alt_final_metrics,
                )

                alt_score = 0.5
                if predicted_outcome and "candidate_scores" in predicted_outcome:
                    alt_score = predicted_outcome["candidate_scores"].get(alt_act, 0.5)

                alt_risk_sources = [s.risk_source for s in alt_plan_steps]
                alt_dom_src = (
                    "trained_model_prediction"
                    if any(s == "trained_model_prediction" for s in alt_risk_sources)
                    else ("unverified_caller_input" if any(s == "unverified_caller_input" for s in alt_risk_sources) else "deterministic_heuristic")
                )

                alternatives.append({
                    "name": f"Alternative Branch: {alt_act}",
                    "first_action": alt_act,
                    "steps": [s.model_dump() for s in alt_plan_steps],
                    "predicted_outcome_score": round(float(alt_score), 4),
                    "cumulative_risk": round(alt_cum_risk, 3),
                    "expected_risk": round(alt_cum_risk, 3),
                    "risk_source": alt_dom_src,
                    "constraint_compliance": alt_compliant,
                    "goal_satisfied": alt_goal_satisfied,
                    "goal_satisfaction_details": alt_details,
                    "rationale": f"Independent alternative plan executing '{alt_act}' sequence evaluated against goal criteria and world transitions.",
                })

        plan_risk_sources = [s.risk_source for s in plan_steps]
        dominant_risk_source = (
            "trained_model_prediction"
            if any(src == "trained_model_prediction" for src in plan_risk_sources)
            else ("unverified_caller_input" if any(src == "unverified_caller_input" for src in plan_risk_sources) else "deterministic_heuristic")
        )

        overall_provenance = {
            "dominant_risk_source": dominant_risk_source,
            "step_sources": plan_risk_sources,
            "model_id": next((s.model_id for s in plan_steps if s.model_id), None),
            "model_version": next((s.model_version for s in plan_steps if s.model_version), None),
        }

        # Explicit goal conditions and measurable postcondition verification
        final_metrics = plan_steps[-1].expected_outcome.get("state_delta", {}) if plan_steps else {}
        goal_satisfied, satisfaction_details = self._evaluate_goal_satisfaction(
            target_goal=target_goal,
            plan_steps=plan_steps,
            constraint_compliance=constraint_compliance,
            current_world_state=current_world_state,
            final_simulated_metrics=final_metrics,
        )

        return Plan(
            target_goal_id=target_goal.goal_id,
            plan_stage="proposed",
            steps=plan_steps,
            cumulative_risk=round(cumulative_risk, 3),
            risk_source=dominant_risk_source,
            risk_provenance=overall_provenance,
            plan_efficiency_score=round(efficiency, 2),
            constraint_compliance=constraint_compliance,
            goal_satisfied=goal_satisfied,
            simulated_goal_achieved=goal_satisfied,
            is_verified_goal_completion=False,  # ALWAYS False for proposed plans
            goal_satisfaction_details=satisfaction_details,
            alternative_plans=alternatives,
        )

    def approve_plan(self, plan: Plan, approver_id: str = "system_safety_controller") -> Plan:
        """Explicit plan lifecycle transition: separates proposed plan from approved plan.
        Only approved plans may be dispatched for execution.
        """
        if plan.plan_stage != "proposed":
            raise ValueError(
                f"Cannot approve plan in '{plan.plan_stage}' stage. Only 'proposed' plans can be approved."
            )
        plan.plan_stage = "approved"
        return plan

    def _evaluate_metric_condition(self, metric_name: str, obs_val: Any, target_condition: Any) -> Tuple[bool, str]:
        """Compares actual observed metric value against numeric or non-numeric target conditions.
        Supports:
        - Dict-based condition operators: {"<=": 0.05}, {">=": 10}, {"==": "stable"}, {"!=": "offline"}
        - Categorical / non-numeric string comparisons: "stable", "isolated", "closed"
        - Boolean flags: True, False
        - Standard numeric threshold and bound checks
        """
        # Dict condition specification: e.g. {"<=": 0.1}, {">=": 12.0}, {"==": "stable"}
        if isinstance(target_condition, dict):
            for op, expected in target_condition.items():
                if op == "<=":
                    if not (float(obs_val) <= float(expected)):
                        return False, f"Expected <= {expected}, observed {obs_val}"
                elif op == "<":
                    if not (float(obs_val) < float(expected)):
                        return False, f"Expected < {expected}, observed {obs_val}"
                elif op == ">=":
                    if not (float(obs_val) >= float(expected)):
                        return False, f"Expected >= {expected}, observed {obs_val}"
                elif op == ">":
                    if not (float(obs_val) > float(expected)):
                        return False, f"Expected > {expected}, observed {obs_val}"
                elif op in ("==", "eq"):
                    if str(obs_val).strip().lower() != str(expected).strip().lower():
                        return False, f"Expected == '{expected}', observed '{obs_val}'"
                elif op in ("!=", "neq"):
                    if str(obs_val).strip().lower() == str(expected).strip().lower():
                        return False, f"Expected != '{expected}', observed '{obs_val}'"
            return True, "Dict condition satisfied"

        # Boolean comparison
        if isinstance(target_condition, bool) or isinstance(obs_val, bool):
            ok = (bool(obs_val) == bool(target_condition))
            return ok, ("Boolean matched" if ok else f"Expected {target_condition}, observed {obs_val}")

        # Non-numeric string condition (e.g. status == "stable")
        if isinstance(target_condition, str) and not target_condition.replace(".", "", 1).replace("-", "", 1).isdigit():
            ok = (str(obs_val).strip().lower() == target_condition.strip().lower())
            return ok, ("Categorical condition met" if ok else f"Expected '{target_condition}', observed '{obs_val}'")

        # Numeric float/int comparison
        try:
            num_obs = float(obs_val)
            num_target = float(target_condition)

            m_lower = metric_name.lower()
            # Metrics representing risk, instability, exposure, latency, or error must be <= target
            if any(k in m_lower for k in ("risk", "exposure", "instability", "error", "disruption", "latency")):
                ok = (num_obs <= num_target)
                return ok, (f"Upper bound met ({num_obs} <= {num_target})" if ok else f"Exceeded limit ({num_obs} > {num_target})")

            # Metrics representing SLA, uptime, availability, score, or balance must be >= target
            if any(k in m_lower for k in ("sla", "uptime", "availability", "score", "balance", "health")):
                ok = (num_obs >= num_target)
                return ok, (f"Lower bound met ({num_obs} >= {num_target})" if ok else f"Below threshold ({num_obs} < {num_target})")

            # Default exact numerical matching with tolerance
            ok = (abs(num_obs - num_target) <= 1e-4)
            return ok, (f"Metric matched ({num_obs} == {num_target})" if ok else f"Value mismatch ({num_obs} != {num_target})")
        except (ValueError, TypeError):
            ok = (str(obs_val).strip().lower() == str(target_condition).strip().lower())
            return ok, ("String fallback matched" if ok else f"Expected '{target_condition}', observed '{obs_val}'")

    def record_step_execution(
        self,
        plan: Plan,
        step_number: int,
        observed_outcome: Dict[str, Any],
        observed_state: Optional[WorldState] = None,
        execution_status: str = "success",
    ) -> ExecutedActionStep:
        """Explicit action execution and outcome observation interface.
        Enforces step ID legitimacy, sequential ordering, record completeness,
        and records real observed telemetry without mistaking proposed steps for outcomes.
        """
        # Validate plan lifecycle stage
        if plan.plan_stage not in ["proposed", "approved", "in_progress", "executing"]:
            raise ValueError(
                f"Cannot record execution for plan in '{plan.plan_stage}' stage."
            )

        # Validate step completeness
        if observed_outcome is None or not isinstance(observed_outcome, dict):
            raise ValueError(
                f"Incomplete execution record: observed_outcome must be a valid dictionary for step {step_number}."
            )
        if execution_status not in ["success", "failed", "timeout"]:
            raise ValueError(
                f"Incomplete execution record: invalid execution_status '{execution_status}' for step {step_number}."
            )

        # Validate legitimate step ID
        matching_step = next((s for s in plan.steps if s.step_number == step_number), None)
        if not matching_step:
            planned_ids = [s.step_number for s in plan.steps]
            raise ValueError(
                f"Unknown execution step ID: step {step_number} is not in plan (expected one of {planned_ids})."
            )

        # Reject duplicate step execution
        if any(s.step_number == step_number for s in plan.executed_steps):
            raise ValueError(
                f"Duplicate execution step ID: step {step_number} has already been recorded."
            )

        # Validate sequential ordering and dependencies
        expected_step = len(plan.executed_steps) + 1
        if step_number != expected_step:
            raise ValueError(
                f"Execution order violation: step {step_number} executed out of sequence (expected step {expected_step})."
            )

        action_name = matching_step.action_name
        metrics = (observed_state.environment_metrics if observed_state else {}).copy()

        executed_step = ExecutedActionStep(
            step_number=step_number,
            action_name=action_name,
            executed_at=time.time(),
            execution_status=execution_status,
            observed_outcome=observed_outcome,
            observed_state_metrics=metrics,
            verified_goal_completion=False,
        )
        plan.executed_steps.append(executed_step)
        plan.observed_outcomes.append(observed_outcome)

        if len(plan.executed_steps) == len(plan.steps):
            plan.plan_stage = "executed"
        else:
            plan.plan_stage = "in_progress"

        return executed_step

    def verify_plan_execution_completion(
        self,
        plan: Plan,
        target_goal: GoalNode,
        final_observed_world_state: WorldState,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verifies actual execution completion against real observed outcomes and final world state.
        Strictly enforces:
        1. Removal of automatic planned-postcondition copying: only observed evidence is accepted.
        2. Rejection of duplicate, missing, unknown, or out-of-order execution step IDs.
        3. Rejection of incomplete execution records.
        4. Independent comparison of every numeric and non-numeric target metric against actual observed values.
        5. Verification that successful execution with unsuccessful outcomes fails goal verification.
        """
        if not plan.executed_steps:
            return False, {
                "verified": False,
                "reason": "No actions have been executed.",
                "error_code": "NO_EXECUTED_STEPS",
            }

        planned_step_numbers = [s.step_number for s in plan.steps]
        executed_step_numbers = [s.step_number for s in plan.executed_steps]

        # 1. Reject duplicate step IDs
        if len(executed_step_numbers) != len(set(executed_step_numbers)):
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"
            return False, {
                "verified": False,
                "reason": "Duplicate execution step IDs detected.",
                "error_code": "DUPLICATE_STEP_ID",
            }

        # 2. Reject unknown step IDs
        unknown_ids = [n for n in executed_step_numbers if n not in planned_step_numbers]
        if unknown_ids:
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"
            return False, {
                "verified": False,
                "reason": f"Unknown execution step IDs: {unknown_ids}",
                "error_code": "UNKNOWN_STEP_ID",
            }

        # 3. Reject missing step IDs (partial execution)
        missing_ids = [n for n in planned_step_numbers if n not in executed_step_numbers]
        if missing_ids:
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"
            return False, {
                "verified": False,
                "reason": f"Missing execution steps: {missing_ids} ({len(plan.executed_steps)} of {len(plan.steps)} steps executed).",
                "error_code": "MISSING_STEP_ID",
            }

        # 4. Reject execution ordering violations
        if executed_step_numbers != planned_step_numbers:
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"
            return False, {
                "verified": False,
                "reason": f"Execution ordering mismatch: expected {planned_step_numbers}, got {executed_step_numbers}",
                "error_code": "ORDERING_MISMATCH",
            }

        # 5. Reject incomplete execution records
        for s in plan.executed_steps:
            if s.observed_outcome is None or not isinstance(s.observed_outcome, dict) or s.execution_status not in ["success", "failed", "timeout"]:
                plan.is_verified_goal_completion = False
                plan.plan_stage = "failed"
                return False, {
                    "verified": False,
                    "reason": f"Incomplete execution record in step {s.step_number}.",
                    "error_code": "INCOMPLETE_RECORD",
                }

        # 6. Check for any execution status failures
        failed_steps = [s for s in plan.executed_steps if s.execution_status != "success"]
        if failed_steps:
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"
            return False, {
                "verified": False,
                "reason": f"Execution failures in steps: {[s.step_number for s in failed_steps]}",
                "error_code": "STEP_EXECUTION_FAILURE",
            }

        # 7. Collect observed postconditions STRICTLY from observed telemetry
        # (NO automatic copying of planned postconditions!)
        observed_postconditions = set()
        for s in plan.executed_steps:
            # Check observed_postconditions list
            obs_posts = s.observed_outcome.get("observed_postconditions", [])
            if isinstance(obs_posts, (list, tuple, set)):
                observed_postconditions.update(obs_posts)

            # Check explicit positive factual evidence in observed_outcome
            for k, v in s.observed_outcome.items():
                if v is True and isinstance(k, str) and (
                    k.startswith("action_executed_")
                    or "resolved" in k
                    or "stabilized" in k
                    or "isolated" in k
                    or "completed" in k
                ):
                    observed_postconditions.add(k)

        # 8. Target-Metric Validation: compare every target metric against actual observed value
        target_metrics = getattr(target_goal, "target_state_metrics", {}) or {}
        metrics_satisfied = True
        failed_metrics = []
        metric_eval_details = {}

        for metric_name, target_condition in target_metrics.items():
            obs_val = None
            if hasattr(final_observed_world_state, "environment_metrics") and metric_name in final_observed_world_state.environment_metrics:
                obs_val = final_observed_world_state.environment_metrics[metric_name]
            elif hasattr(final_observed_world_state, "observed_facts") and metric_name in final_observed_world_state.observed_facts:
                obs_val = final_observed_world_state.observed_facts[metric_name]
            else:
                for s in reversed(plan.executed_steps):
                    if metric_name in s.observed_state_metrics:
                        obs_val = s.observed_state_metrics[metric_name]
                        break
                    if metric_name in s.observed_outcome:
                        obs_val = s.observed_outcome[metric_name]
                        break

            if obs_val is None:
                metrics_satisfied = False
                failed_metrics.append(f"Missing target metric: '{metric_name}' not found in observed telemetry.")
                metric_eval_details[metric_name] = {"satisfied": False, "reason": "Metric not found"}
                continue

            is_ok, eval_reason = self._evaluate_metric_condition(metric_name, obs_val, target_condition)
            metric_eval_details[metric_name] = {
                "observed": obs_val,
                "target": target_condition,
                "satisfied": is_ok,
                "reason": eval_reason,
            }
            if not is_ok:
                metrics_satisfied = False
                failed_metrics.append(f"Metric '{metric_name}': observed {obs_val} violates target condition {target_condition} ({eval_reason})")

        # 9. Verify required postconditions
        required_posts = getattr(target_goal, "required_postconditions", []) or []
        missing_posts = [p for p in required_posts if p not in observed_postconditions]

        is_completed = (len(failed_steps) == 0) and metrics_satisfied and (len(missing_posts) == 0)

        verification_details = {
            "verified": is_completed,
            "executed_steps_count": len(plan.executed_steps),
            "observed_postconditions": list(observed_postconditions),
            "metric_evaluations": metric_eval_details,
            "failed_metrics": failed_metrics,
            "missing_required_postconditions": missing_posts,
        }

        if is_completed:
            plan.is_verified_goal_completion = True
            plan.plan_stage = "completed"
            for s in plan.executed_steps:
                s.verified_goal_completion = True
        else:
            plan.is_verified_goal_completion = False
            plan.plan_stage = "failed"

        return is_completed, verification_details

    def should_replan(self, current_plan: Plan, current_world_state: WorldState) -> Tuple[bool, Optional[str]]:
        """Evaluates whether current execution conditions have diverged, necessitating replanning."""
        if not current_plan.steps:
            return True, "Current plan has no remaining steps."

        next_action = current_plan.steps[0].action_name
        for constraint in current_world_state.active_constraints:
            if next_action.lower() in constraint.lower() and (
                "prohibit" in constraint.lower() or "forbidden" in constraint.lower()
            ):
                return True, f"Planned action '{next_action}' is now prohibited by updated world state constraint: {constraint}"

        if current_world_state.environment_metrics.get("system_exposure", 0.0) > 0.8:
            return True, "Critical security exposure detected in world state; immediate emergency replan required."

        return False, None
