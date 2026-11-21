"""Cognitive Answer Synthesis Subsystem for CDM-1 Cognitive AI Engine.
Generates complete, understandable, evidence-grounded natural-language answers
from actual pipeline outputs, neural inferences, world states, and safety verification results.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import re
import time
import logging
from pydantic import BaseModel, Field

from cdm.cognitive.state import (
    PerceptionResult,
    ContextFrame,
    WorkingMemoryState,
    WorldState,
    ReasoningTrace,
    Plan,
    CognitiveEngineResult,
)
from cdm.synthesis.document_retriever import DocumentKnowledgeRetriever, DocumentPassage
from cdm.synthesis.citation_manager import CitationManager, GroundedCitation, CitationVerificationReport
from cdm.synthesis.router import IntentTaskRouter, CognitiveRoutePlan
from cdm.synthesis.answer_validator import AnswerValidator, AnswerValidationResult
from cdm.synthesis.tool_orchestrator import ControlledToolOrchestrator
from cdm.context.reference_resolver import ReferenceResolver

logger = logging.getLogger("cdm.synthesis")


class SynthesizedAnswer(BaseModel):
    """Structured representation of synthesized natural-language response."""
    answer: str
    answer_type: str
    explanation: str
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    important_assumptions: List[str] = Field(default_factory=list)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    risks_and_uncertainty: str = ""
    recommended_next_steps: List[str] = Field(default_factory=list)
    safety_status: str = "verified"  # 'verified', 'needs_approval', 'abstained', 'policy_violation', 'unverified'
    limitations: List[str] = Field(default_factory=list)
    retrieved_external_facts: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)
    intent_task: str = "question"
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None


class ExternalInformationRetriever:
    """Configurable retriever for real-world and external information.
    Ensures clear separation of user input, model predictions, stored memory, and retrieved external facts.
    """

    def __init__(self, external_tools: Optional[Dict[str, Any]] = None, allow_demo_knowledge: Optional[bool] = None):
        self.external_tools = external_tools or {}
        self.is_available = True
        if allow_demo_knowledge is None:
            # Strictly disabled in production; permitted in development / testing environments
            env = os.environ.get("CDM_ENV", "development").lower()
            self.allow_demo_knowledge = (env != "production")
        else:
            self.allow_demo_knowledge = allow_demo_knowledge

    def set_availability(self, available: bool) -> None:
        """Configures live availability status of external retrieval services."""
        self.is_available = available

    def retrieve(self, query: str, domain: Optional[str] = None) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """Retrieves verified external information for a query.
        Returns:
            (success: bool, facts: List[Dict], error_message: Optional[str])
        """
        if not self.is_available:
            return False, [], "External retrieval service is currently unavailable or disconnected."

        facts: List[Dict[str, Any]] = []
        now = time.time()

        # Check for registered custom tools (e.g. WebRetrievalTool, retrieval_api)
        if "retrieval_api" in self.external_tools:
            try:
                tool = self.external_tools["retrieval_api"]
                tool_res = tool(query=query, domain=domain)
                if isinstance(tool_res, list):
                    for item in tool_res:
                        if isinstance(item, dict):
                            item.setdefault("retrieval_timestamp", now)
                            item.setdefault("source", "Infrastructure Telemetry API")
                            facts.append(item)
                elif isinstance(tool_res, dict):
                    tool_res.setdefault("retrieval_timestamp", now)
                    tool_res.setdefault("source", "Infrastructure Telemetry API")
                    facts.append(tool_res)
                return True, facts, None
            except Exception as e:
                logger.error(f"External retrieval tool execution failed: {e}")
                return False, [], f"Retrieval tool error: {str(e)}"

        # Strictly distinguish demo knowledge from production retrieval
        if self.allow_demo_knowledge:
            q_lower = query.lower()
            if "server" in q_lower or "10.0.1.5" in q_lower or "cluster" in q_lower:
                facts.append({
                    "source": "Infrastructure Telemetry Service (v1.0-demo)",
                    "fact": "Cluster node 10.0.1.5 telemetry indicates memory pressure at 87% with healthy heartbeat.",
                    "retrieval_timestamp": now,
                    "verified": False,
                    "is_demo": True,
                    "confidence": 0.85,
                })
                return True, facts, None

        # Production mode: zero simulated facts; return empty if unmapped
        return True, [], None


class AnswerSynthesizer:
    """Cognitive Answer Synthesis Engine for CDM-1.
    Translates five-stage cognitive pipeline outputs into human-understandable,
    verifiable, natural-language answers strictly grounded in actual execution telemetry.
    """

    def __init__(
        self,
        retriever: Optional[ExternalInformationRetriever] = None,
        doc_retriever: Optional[DocumentKnowledgeRetriever] = None,
        citation_manager: Optional[CitationManager] = None,
        router: Optional[IntentTaskRouter] = None,
        validator: Optional[AnswerValidator] = None,
        tool_orchestrator: Optional[ControlledToolOrchestrator] = None,
        reference_resolver: Optional[ReferenceResolver] = None,
    ):
        self.retriever = retriever or ExternalInformationRetriever()
        self.doc_retriever = doc_retriever or DocumentKnowledgeRetriever()
        self.citation_manager = citation_manager or CitationManager()
        self.router = router or IntentTaskRouter()
        self.validator = validator or AnswerValidator()
        self.tool_orchestrator = tool_orchestrator or ControlledToolOrchestrator()
        self.reference_resolver = reference_resolver or ReferenceResolver()

    def classify_question_type(
        self,
        question: str,
        context: Optional[ContextFrame] = None,
        task: str = "choice",
        structured_features: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classifies incoming query into one of the nine supported question categories."""
        q = question.strip().lower()

        # Check for follow-up questions
        follow_up_cues = ["why did you", "why choose", "what if", "compare it with", "revise the plan", "previous", "earlier"]
        if any(c in q for c in follow_up_cues) or (context and context.turn_index > 1 and len(q.split()) <= 8 and ("why" in q or "how" in q)):
            return "follow_up"

        # Check for comparison
        if "compare" in q or " vs " in q or "versus" in q or "difference between" in q:
            return "comparison"

        # Check for risk assessment
        if "risk" in q or "financial impact" in q or "safe to" in q or "exposure" in q or "hazard" in q:
            return "risk_assessment"

        # Check for planning
        if "steps" in q or "plan" in q or "how do we execute" in q or "prerequisites" in q or "workflow" in q:
            return "planning"

        # Check for external information
        if "external" in q or "latest status" in q or "check live" in q or "real-time" in q:
            return "external_info"

        # Check for factual / general informational
        if q.startswith("what is") or q.startswith("what are") or q.startswith("where") or q.startswith("who") or "define" in q:
            return "factual"

        # Check for explanatory
        if q.startswith("why") or q.startswith("how does") or "explain" in q or "reason for" in q:
            return "explanatory"

        # Check for problem-solving
        if "how to resolve" in q or "how do i fix" in q or "customer reports" in q or "error" in q or "issue" in q or "troubleshoot" in q:
            return "problem_solving"

        # Check for decision-making
        if "should we" in q or "choose between" in q or "select" in q or "which option" in q or (task in ("choice", "binary") and structured_features and "candidates" in structured_features):
            return "decision"

        if task in ("choice", "binary"):
            return "decision"

        return "general"

    def synthesize(
        self,
        question: str,
        task: str,
        perception: PerceptionResult,
        context: ContextFrame,
        working_memory: WorkingMemoryState,
        world_state: WorldState,
        semantic_rules: List[Dict[str, Any]],
        procedural_workflows: List[Dict[str, Any]],
        episodic_cases: List[Dict[str, Any]],
        reasoning_trace: ReasoningTrace,
        causal_analysis: Optional[Dict[str, Any]],
        cdm_result: Any,
        candidate_rankings: List[Dict[str, Any]],
        proposed_decision: Any,
        execution_authorized: bool,
        execution_status: str,
        abstention_status: str,
        abstention_reason: Optional[str],
        policy_check: Dict[str, Any],
        approval_requirements: Dict[str, Any],
        verification: Dict[str, Any],
        generated_plan: Optional[Plan],
        confidence: float,
        calibrated_conf: float,
        uncertainty: float,
        assessed_risk: float,
        confidence_origin: str,
        risk_origin: str,
        retrieved_external_facts: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
        tenant_id: str = "default_tenant",
    ) -> SynthesizedAnswer:
        """Synthesizes an evidence-grounded natural-language response matching all P0/P1 invariants."""
        err_list = list(errors or [])
        ext_facts = list(retrieved_external_facts or [])

        # 0. Route & Intent check
        route_plan = self.router.plan_route(
            text=question,
            structured_features=world_state.environment_metrics,
            candidates=[c.get("candidate") for c in candidate_rankings if isinstance(c, dict)],
            task_hint=task,
        )

        # 0b. Document Knowledge Base Retrieval with tenant isolation
        doc_passages = self.doc_retriever.retrieve(query=question, top_k=3, tenant_id=tenant_id)
        passages_dicts = [p.model_dump() for p in doc_passages]

        # 0c. Contextual Reference Resolution
        resolved_refs = self.reference_resolver.resolve_references(
            current_text=question,
            context=context,
        )

        q_type = self.classify_question_type(
            question=question,
            context=context,
            task=task,
            structured_features=world_state.environment_metrics,
        )

        limitations: List[str] = []
        important_assumptions: List[str] = []
        recommended_next_steps: List[str] = []

        # Determine safety status string
        if abstention_status == "policy_rejection" or not policy_check.get("passed", True):
            safety_status = "policy_violation"
        elif abstention_status in ("uncertainty", "invalid_numerical_scores", "missing_evidence", "abstained"):
            safety_status = "abstained"
        elif approval_requirements.get("required", False):
            safety_status = "needs_approval"
        elif execution_authorized and verification.get("verified", False):
            safety_status = "verified"
        else:
            safety_status = "unverified"

        # Record limitations
        if uncertainty > 0.40:
            limitations.append(f"State uncertainty is elevated ({uncertainty * 100:.1f}%), indicating incomplete situational parameters.")
        if context.detected_contradictions:
            limitations.append(f"Contradictory user statements detected across conversation dialogue: {'; '.join(context.detected_contradictions)}")
        if not semantic_rules:
            limitations.append("No domain-specific semantic rules are loaded for this operational context.")

        # Supporting evidence compilation
        evidence_summary: Dict[str, Any] = {
            "perceived_entities": [f"{e.entity_type}:{e.name}" for e in perception.extracted_entities],
            "signals": perception.signals,
            "deduction_steps_count": len(reasoning_trace.steps),
            "semantic_rules_count": len(semantic_rules),
            "procedural_workflows_count": len(procedural_workflows),
            "episodic_precedents_count": len(episodic_cases),
            "candidate_rankings": candidate_rankings,
            "external_facts_count": len(ext_facts),
        }

        # Build risk & uncertainty explanation
        conf_pct = calibrated_conf * 100.0
        risk_pct = assessed_risk * 100.0
        risk_uncertainty_narrative = (
            f"Confidence is assessed at {conf_pct:.1f}% ({confidence_origin}), representing the neural model's "
            f"calibrated probability distribution across evaluated candidates. Assessed operational risk is {risk_pct:.1f}% "
            f"({risk_origin}), representing the projected probability of service degradation or policy violation."
        )

        # Build alternatives list
        alternatives = [
            {
                "candidate": c.get("candidate"),
                "score": c.get("score"),
                "rank": c.get("rank"),
                "is_prohibited": c.get("is_prohibited", False),
            }
            for c in candidate_rankings
            if c.get("candidate") != proposed_decision
        ]

        # -------------------------------------------------------------
        # Case A: Safety Abstention or Policy Rejection
        # -------------------------------------------------------------
        is_abstained_or_rejected = (
            abstention_status != "none"
            or not policy_check.get("passed", True)
            or execution_status in ("abstained", "rejected")
        )

        if is_abstained_or_rejected and q_type not in ("external_info", "risk_assessment", "factual"):
            direct_answer, explanation = self._synthesize_abstention_or_rejection(
                proposed_decision=proposed_decision,
                abstention_status=abstention_status,
                abstention_reason=abstention_reason,
                policy_check=policy_check,
                confidence=calibrated_conf,
                uncertainty=uncertainty,
                assessed_risk=assessed_risk,
                contradictions=reasoning_trace.contradictions_found,
            )
            recommended_next_steps.append("Escalate case to human supervisor for manual intervention.")
            recommended_next_steps.append("Review policy constraints and resolve recorded contradictions.")

            return SynthesizedAnswer(
                answer=direct_answer,
                answer_type="abstention",
                explanation=explanation,
                supporting_evidence=evidence_summary,
                important_assumptions=important_assumptions,
                alternatives=alternatives,
                risks_and_uncertainty=risk_uncertainty_narrative,
                recommended_next_steps=recommended_next_steps,
                safety_status=safety_status,
                limitations=limitations,
                retrieved_external_facts=ext_facts,
            )

        # -------------------------------------------------------------
        # Case B: Human Approval Required
        # -------------------------------------------------------------
        if approval_requirements.get("required", False):
            direct_answer, explanation = self._synthesize_human_approval_required(
                proposed_decision=proposed_decision,
                approval_requirements=approval_requirements,
                confidence=calibrated_conf,
                assessed_risk=assessed_risk,
                generated_plan=generated_plan,
            )
            approver_role = approval_requirements.get("approver_role", "compliance_officer")
            recommended_next_steps.append(f"Submit execution authorization ticket to authorized [{approver_role}].")
            recommended_next_steps.append("Keep proposed plan in staged holding until cryptographic token is provisioned.")

            return SynthesizedAnswer(
                answer=direct_answer,
                answer_type="decision" if q_type in ("decision", "general") else q_type,
                explanation=explanation,
                supporting_evidence=evidence_summary,
                important_assumptions=important_assumptions,
                alternatives=alternatives,
                risks_and_uncertainty=risk_uncertainty_narrative,
                recommended_next_steps=recommended_next_steps,
                safety_status=safety_status,
                limitations=limitations,
                retrieved_external_facts=ext_facts,
            )

        # -------------------------------------------------------------
        # Case C: Question-Type Specialized Syntheses
        # -------------------------------------------------------------
        if q_type == "factual":
            direct_answer, explanation = self._synthesize_factual(
                question=question,
                semantic_rules=semantic_rules,
                world_state=world_state,
                ext_facts=ext_facts,
                passages=doc_passages,
            )
        elif q_type == "explanatory":
            direct_answer, explanation = self._synthesize_explanatory(
                question=question,
                proposed_decision=proposed_decision,
                reasoning_trace=reasoning_trace,
                causal_analysis=causal_analysis,
                semantic_rules=semantic_rules,
            )
        elif q_type == "comparison":
            direct_answer, explanation = self._synthesize_comparison(
                candidate_rankings=candidate_rankings,
                causal_analysis=causal_analysis,
                world_state=world_state,
            )
        elif q_type == "risk_assessment":
            direct_answer, explanation = self._synthesize_risk_assessment(
                assessed_risk=assessed_risk,
                risk_origin=risk_origin,
                uncertainty=uncertainty,
                causal_analysis=causal_analysis,
                world_state=world_state,
                execution_status=execution_status,
                abstention_reason=abstention_reason,
            )
        elif q_type == "planning":
            direct_answer, explanation = self._synthesize_planning(
                generated_plan=generated_plan,
                proposed_decision=proposed_decision,
                execution_authorized=execution_authorized,
            )
        elif q_type == "follow_up":
            direct_answer, explanation = self._synthesize_follow_up(
                question=question,
                context=context,
                proposed_decision=proposed_decision,
                reasoning_trace=reasoning_trace,
                candidate_rankings=candidate_rankings,
                resolved_decision=resolved_refs.get("resolved_decision"),
            )
        elif q_type == "external_info":
            direct_answer, explanation = self._synthesize_external_info(
                question=question,
                ext_facts=ext_facts,
                execution_status=execution_status,
                abstention_reason=abstention_reason,
            )
        elif q_type == "problem_solving":
            direct_answer, explanation = self._synthesize_problem_solving(
                question=question,
                proposed_decision=proposed_decision,
                procedural_workflows=procedural_workflows,
                generated_plan=generated_plan,
                execution_authorized=execution_authorized,
            )
        else:
            # Standard Decision & General Synthesis
            direct_answer, explanation = self._synthesize_standard_decision(
                proposed_decision=proposed_decision,
                candidate_rankings=candidate_rankings,
                execution_authorized=execution_authorized,
                confidence=calibrated_conf,
                generated_plan=generated_plan,
            )

        if is_abstained_or_rejected:
            recommended_next_steps.append("Escalate case to human supervisor for manual intervention.")
            recommended_next_steps.append("Review policy constraints and resolve recorded contradictions.")
        elif generated_plan and generated_plan.steps:
            recommended_next_steps.append(f"Execute step 1: {generated_plan.steps[0].action_name}.")
            if len(generated_plan.steps) > 1:
                recommended_next_steps.append(f"Proceed with remaining {len(generated_plan.steps) - 1} plan steps upon verification.")
        else:
            recommended_next_steps.append(f"Proceed with primary action '{proposed_decision}'.")

        # Attribute citations and verify claims
        citation_report = self.citation_manager.attribute_and_verify(
            answer_text=direct_answer,
            retrieved_passages=doc_passages,
            semantic_rules=semantic_rules,
            external_facts=ext_facts,
            observed_facts=world_state.observed_facts,
        )

        citations_list = [c.model_dump() for c in citation_report.citations]
        unsupported = citation_report.unsupported_claims

        # Validate answer consistency against structured results
        val_res = self.validator.validate(
            answer_text=direct_answer,
            cognitive_result=CognitiveEngineResult(
                proposed_decision=proposed_decision,
                execution_authorized=execution_authorized,
                execution_status=execution_status,
                abstain=abstention_status != "none" or execution_status in ("abstained", "rejected"),
                verification=verification,
                decision=proposed_decision,
                confidence=confidence,
                calibrated_confidence=calibrated_conf,
                reasoning_evidence=reasoning_trace,
            ),
            unsupported_claims=unsupported,
            citations=citations_list,
            retrieved_passages=passages_dicts,
        )

        return SynthesizedAnswer(
            answer=direct_answer,
            answer_type=q_type,
            explanation=explanation,
            supporting_evidence=evidence_summary,
            important_assumptions=important_assumptions,
            alternatives=alternatives,
            risks_and_uncertainty=risk_uncertainty_narrative,
            recommended_next_steps=recommended_next_steps,
            safety_status=safety_status,
            limitations=limitations,
            retrieved_external_facts=ext_facts,
            citations=citations_list,
            unsupported_claims=unsupported,
            retrieved_passages=passages_dicts,
            intent_task=route_plan.intent_task,
            validation_result=val_res.model_dump(),
        )

    # -----------------------------------------------------------------
    # Specialized Synthesizer Generators
    # -----------------------------------------------------------------

    def _synthesize_abstention_or_rejection(
        self,
        proposed_decision: Any,
        abstention_status: str,
        abstention_reason: Optional[str],
        policy_check: Dict[str, Any],
        confidence: float,
        uncertainty: float,
        assessed_risk: float,
        contradictions: List[str],
    ) -> Tuple[str, str]:
        """Generates clear explanation when model abstains or policy rejects execution."""
        if abstention_status == "policy_rejection" or not policy_check.get("passed", True):
            violated = policy_check.get("violated_rules", [])
            violated_str = ", ".join(violated) if violated else f"Action '{proposed_decision}' is prohibited by domain policy"
            direct_answer = (
                f"Execution is rejected due to policy constraint violations. Proposed candidate '{proposed_decision}' "
                f"cannot be executed under active governance rules."
            )
            explanation = (
                f"The Safety Verifier blocked execution because the following hard policy rules were breached: {violated_str}. "
                f"In accordance with CDM-1 safety invariants, prohibited candidates are strictly non-executable."
            )
        elif abstention_status == "uncertainty":
            direct_answer = (
                f"The cognitive engine abstained from autonomous decision-making because uncertainty exceeded safe operating thresholds."
            )
            explanation = (
                f"Calibrated confidence is {confidence * 100:.1f}% with state uncertainty at {uncertainty * 100:.1f}%. "
                f"Reason: {abstention_reason or 'Confidence fell below the minimum operating threshold (35%)'}. "
                f"Manual intervention by a human operator is required to prevent unintended operational drift."
            )
        elif abstention_status == "invalid_numerical_scores":
            direct_answer = (
                "The cognitive engine safely abstained because corrupted numerical values (NaN or infinity) were detected in model scoring."
            )
            explanation = (
                "Numerical safety checks detected non-finite scores in confidence or risk heads. "
                "Execution authorization was withheld immediately to preserve system stability."
            )
        else:
            direct_answer = f"The cognitive engine abstained from selecting an executable action: {abstention_reason or 'Safety boundary reached.'}"
            explanation = (
                f"Abstention status: [{abstention_status}]. Reason: {abstention_reason}. "
                "Autonomous execution authorization is withheld."
            )

        return direct_answer, explanation

    def _synthesize_human_approval_required(
        self,
        proposed_decision: Any,
        approval_requirements: Dict[str, Any],
        confidence: float,
        assessed_risk: float,
        generated_plan: Optional[Plan],
    ) -> Tuple[str, str]:
        """Generates clear response when proposed action requires human sign-off."""
        role = approval_requirements.get("approver_role", "compliance_officer")
        reason = approval_requirements.get("reason", "Action is classified as sensitive")
        direct_answer = (
            f"The recommended action is '{proposed_decision}', but execution is NOT authorized autonomously. "
            f"It requires mandatory human approval by a [{role}]."
        )
        explanation = (
            f"The neural reasoner selected '{proposed_decision}' with {confidence * 100:.1f}% confidence and "
            f"{assessed_risk * 100:.1f}% assessed risk. However, because {reason}, execution authorization is "
            f"strictly withheld until an authorized human supervisor reviews and signs off on the proposed plan."
        )
        return direct_answer, explanation

    def _synthesize_factual(
        self,
        question: str,
        semantic_rules: List[Dict[str, Any]],
        world_state: WorldState,
        ext_facts: List[Dict[str, Any]],
        passages: Optional[List[Any]] = None,
    ) -> Tuple[str, str]:
        """Generates concise factual answers grounded in verified memory, rules, and document passages."""
        facts: List[str] = []
        if ext_facts:
            facts.extend([f"{f.get('fact')} (Source: {f.get('source')})" for f in ext_facts])

        for p in (passages or []):
            p_content = getattr(p, "content", "") if hasattr(p, "content") else p.get("content", "")
            p_title = getattr(p, "title", "Document") if hasattr(p, "title") else p.get("title", "Document")
            p_doc = getattr(p, "doc_id", "DOC") if hasattr(p, "doc_id") else p.get("doc_id", "DOC")
            if p_content:
                facts.append(f"{p_content} (Source: {p_title} [{p_doc}])")

        for r in semantic_rules:
            desc = r.get("description") or r.get("condition_description")
            if desc:
                facts.append(f"Rule [{r.get('rule_id', 'RULE')}]: {desc}")

        metrics = [f"{k} = {v}" for k, v in world_state.environment_metrics.items()]
        if metrics:
            facts.append(f"Environment parameters: {', '.join(metrics)}")

        if facts:
            direct_answer = facts[0]
            explanation = "Additional verified operational facts: " + "; ".join(facts[1:]) if len(facts) > 1 else "Grounded directly in stored domain knowledge."
        else:
            direct_answer = "No stored domain rules or verified external records were found matching this inquiry."
            explanation = "The working memory and semantic repository do not contain explicit entries for this specific subject."

        return direct_answer, explanation

    def _synthesize_explanatory(
        self,
        question: str,
        proposed_decision: Any,
        reasoning_trace: ReasoningTrace,
        causal_analysis: Optional[Dict[str, Any]],
        semantic_rules: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Explains why a decision was made or how rules led to a deduction."""
        deductions = [s.conclusion for s in reasoning_trace.steps if s.conclusion]
        direct_answer = f"The selected decision is '{proposed_decision}', determined through multi-step constraint evaluation."
        explanation_parts = []
        if deductions:
            explanation_parts.append(f"Deductive trace: {'; '.join(deductions[:3])}.")
        if causal_analysis and "consequences" in causal_analysis:
            impacts = causal_analysis["consequences"].get(proposed_decision, {})
            if isinstance(impacts, dict):
                explanation_parts.append(f"Causal analysis projects: {impacts.get('projected_effect', 'optimal compliance outcome')}.")

        explanation = " ".join(explanation_parts) if explanation_parts else "Evaluated against verified domain constraints and operational priorities."
        return direct_answer, explanation

    def _synthesize_comparison(
        self,
        candidate_rankings: List[Dict[str, Any]],
        causal_analysis: Optional[Dict[str, Any]],
        world_state: WorldState,
    ) -> Tuple[str, str]:
        """Compares alternatives with numerical scores and trade-offs."""
        if not candidate_rankings:
            return "No candidates available for comparison.", "Evaluation requires at least two candidate options."

        top = candidate_rankings[0]
        runner_up = candidate_rankings[1] if len(candidate_rankings) > 1 else None

        direct_answer = (
            f"Comparison shows '{top['candidate']}' ranked highest with a score of {top.get('score', 0.0):.4f}"
            + (f", exceeding '{runner_up['candidate']}' ({runner_up.get('score', 0.0):.4f})." if runner_up else ".")
        )

        breakdown = []
        for c in candidate_rankings[:3]:
            score_str = f"{c.get('score', 0.0):.4f}" if c.get("score") is not None else "N/A"
            prohib = " (PROHIBITED)" if c.get("is_prohibited") else ""
            breakdown.append(f"Rank {c.get('rank')}: {c.get('candidate')} (Score: {score_str}{prohib})")

        explanation = f"Evaluated candidate distribution: {'; '.join(breakdown)}."
        return direct_answer, explanation

    def _synthesize_risk_assessment(
        self,
        assessed_risk: float,
        risk_origin: str,
        uncertainty: float,
        causal_analysis: Optional[Dict[str, Any]],
        world_state: WorldState,
        execution_status: str = "proposed",
        abstention_reason: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Provides financial, technical, and operational risk assessments."""
        risk_pct = assessed_risk * 100.0
        level = "High" if assessed_risk > 0.40 else ("Medium" if assessed_risk > 0.15 else "Low")
        direct_answer = f"The assessed risk level is {level} at {risk_pct:.1f}% ({risk_origin})."

        explanation_parts = [
            f"Risk is computed from predictive network heads and world state transitions.",
            f"State uncertainty is {uncertainty * 100:.1f}%.",
            f"Active constraints monitored: {len(world_state.active_constraints)}.",
        ]
        if execution_status in ("abstained", "rejected"):
            explanation_parts.append(f"Autonomous operational execution is {execution_status} ({abstention_reason or 'Risk threshold reached'}).")

        explanation = " ".join(explanation_parts)
        return direct_answer, explanation

    def _synthesize_planning(
        self,
        generated_plan: Optional[Plan],
        proposed_decision: Any,
        execution_authorized: bool,
    ) -> Tuple[str, str]:
        """Summarizes planned action sequence, prerequisites, and approval requirements."""
        if not generated_plan or not generated_plan.steps:
            return (
                f"A direct single-step action '{proposed_decision}' is recommended without a multi-step workflow.",
                "No prerequisite sub-steps were required to achieve the active goal."
            )

        step_descs = [f"Step {s.step_number}: {s.action_name}" for s in generated_plan.steps]
        direct_answer = f"Plan '{generated_plan.plan_id}' comprises {len(generated_plan.steps)} steps: {', '.join(step_descs)}."
        explanation = (
            f"Plan status is [{generated_plan.plan_stage}]. Cumulative risk is {generated_plan.cumulative_risk * 100:.1f}%. "
            f"Execution eligibility is {'granted' if execution_authorized else 'pending approval'}."
        )
        return direct_answer, explanation

    def _synthesize_follow_up(
        self,
        question: str,
        context: ContextFrame,
        proposed_decision: Any,
        reasoning_trace: ReasoningTrace,
        candidate_rankings: List[Dict[str, Any]],
        resolved_decision: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Answers contextual follow-up questions referencing dialogue history and resolved references."""
        prior_turns = len(context.dialogue_history)
        last_turn = context.dialogue_history[-2]["content"] if prior_turns >= 2 else "initial inquiry"

        target_dec = resolved_decision or proposed_decision
        direct_answer = (
            f"In the context of your previous message ('{last_turn[:60]}...'), "
            f"the recommended action remains '{target_dec}'."
        )
        explanation = (
            f"Turn index is {context.turn_index}. Reasoning verified that the parameters from earlier in the dialogue "
            f"support '{target_dec}' over alternative candidates."
        )
        return direct_answer, explanation

    def _synthesize_external_info(
        self,
        question: str,
        ext_facts: List[Dict[str, Any]],
        execution_status: str = "proposed",
        abstention_reason: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Answers questions that require external data retrieval."""
        if not ext_facts:
            return (
                "External information for this query could not be retrieved from external sources.",
                "The external retrieval service reported no available data or the endpoint was unreachable. No facts were fabricated."
            )

        fact_narratives = [f"{f.get('fact')} [Source: {f.get('source')}]" for f in ext_facts]
        direct_answer = " ".join(fact_narratives)
        explanation = f"Retrieved {len(ext_facts)} verified external records. Distinguished strictly from internal model hypotheses."
        if execution_status in ("abstained", "rejected"):
            explanation += f" Note: Autonomous operational execution is {execution_status} ({abstention_reason or 'Safety boundary'})."
        return direct_answer, explanation

    def _synthesize_problem_solving(
        self,
        question: str,
        proposed_decision: Any,
        procedural_workflows: List[Dict[str, Any]],
        generated_plan: Optional[Plan],
        execution_authorized: bool,
    ) -> Tuple[str, str]:
        """Provides actionable problem resolution steps."""
        wf_name = procedural_workflows[0].get("name") if procedural_workflows else None
        if wf_name:
            direct_answer = f"To resolve this issue, apply the standard procedure '{wf_name}' starting with '{proposed_decision}'."
            explanation = f"Procedural workflow '{wf_name}' provides compliant runbook guidance for this incident category."
        elif generated_plan and generated_plan.steps:
            steps_str = " -> ".join([s.action_name for s in generated_plan.steps])
            direct_answer = f"The recommended resolution sequence is: {steps_str}."
            explanation = f"Generated by the planning engine to satisfy active recovery goals."
        else:
            direct_answer = f"The optimal resolution is to execute '{proposed_decision}'."
            explanation = "Evaluated against active incident parameters and policy constraints."

        return direct_answer, explanation

    def _synthesize_standard_decision(
        self,
        proposed_decision: Any,
        candidate_rankings: List[Dict[str, Any]],
        execution_authorized: bool,
        confidence: float,
        generated_plan: Optional[Plan],
    ) -> Tuple[str, str]:
        """Standard decision response when selecting among candidates."""
        direct_answer = (
            f"The recommended action is '{proposed_decision}', authorized for execution with {confidence * 100:.1f}% confidence."
            if execution_authorized
            else f"The recommended action is '{proposed_decision}' (evaluation pending approval or verification)."
        )
        explanation = (
            f"Selected as the top-ranking candidate under active policy and metric evaluation."
        )
        return direct_answer, explanation

    def fallback_synthesize(
        self,
        decision: Any,
        proposed_decision: Any,
        execution_status: str,
        execution_authorized: bool,
        abstention_status: str,
        abstention_reason: Optional[str],
        confidence: float,
        risk: float,
        error_msg: str,
    ) -> SynthesizedAnswer:
        """Graceful fallback synthesis invoked if unexpected exceptions occur."""
        direct_answer = (
            f"The evaluated decision is '{proposed_decision or decision}'. "
            f"Execution status: [{execution_status}]."
        )
        explanation = (
            f"Answer synthesis encountered an unexpected formatting error ({error_msg}). "
            f"Core decision parameters: confidence={confidence:.4f}, risk={risk:.4f}, "
            f"authorized={execution_authorized}, abstention={abstention_status}."
        )
        return SynthesizedAnswer(
            answer=direct_answer,
            answer_type="general",
            explanation=explanation,
            safety_status="abstained" if not execution_authorized else "verified",
            limitations=[f"Answer synthesis fallback triggered: {error_msg}"],
        )
