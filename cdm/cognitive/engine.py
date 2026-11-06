"""CDM-1 Unified Integrated Cognitive AI Engine.
Orchestrates Perception, Context, Working & Long-Term Memory, World Model,
Reasoning, Planning, and the CDM-1 Decision Core into a unified cognitive architecture.
"""

from typing import Dict, List, Optional, Any, Union
import time
import uuid

from cdm.cognitive.state import (
    SharedCognitiveState,
    PerceptionResult,
    ContextFrame,
    WorkingMemoryState,
    GoalNode,
    WorldState,
    ReasoningTrace,
    Plan,
    CognitiveEngineResult,
)
from cdm.perception.perception_engine import PerceptionEngine
from cdm.context.context_engine import ContextEngine
from cdm.memory.working_memory import WorkingMemory
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory
from cdm.memory.semantic import SemanticMemory
from cdm.memory.procedural import ProceduralMemory
from cdm.world_model.world_model import WorldModel
from cdm.reasoning.reasoning_engine import ReasoningEngine
from cdm.planning.planning_engine import PlanningEngine
from cdm.learning.feedback import FeedbackCollector
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult


class CognitiveEngine:
    """Unified Cognitive AI Engine with CDM-1 as its Central Decision-Making Core."""

    def __init__(
        self,
        decision_model: Optional[CognitiveDecisionModel] = None,
        db_path: str = "experiments/cognitive_memory.db",
    ):
        # 1. Perception Subsystem
        self.perception_engine = PerceptionEngine()

        # 2. Contextual Subsystem
        self.context_engine = ContextEngine()

        # 3. Memory Subsystems
        self.storage = SQLiteMemoryStorage(db_path=db_path)
        self.episodic_memory = EpisodicMemory(storage=self.storage)
        self.semantic_memory = SemanticMemory(storage=self.storage)
        self.procedural_memory = ProceduralMemory(storage=self.storage)
        self.working_memory = WorkingMemory()

        # 4. World Model
        self.world_model = WorldModel()

        # 5. Reasoning Engine
        self.reasoning_engine = ReasoningEngine()

        # 6. Planning Engine
        self.planning_engine = PlanningEngine(world_model=self.world_model)

        # 7. Learning & Metacognition
        self.feedback_collector = FeedbackCollector()

        # 8. CDM-1 Central Cognitive Decision Core
        self.decision_core = decision_model

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "experiments/release/cdm-1-general",
        db_path: str = "experiments/cognitive_memory.db",
        device: Optional[str] = None,
        local_files_only: bool = True,
    ) -> "CognitiveEngine":
        """Factory method: loads the trained CDM-1 model into the Cognitive Engine."""
        decision_model = CognitiveDecisionModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            device=device,
            local_files_only=local_files_only,
        )
        return cls(decision_model=decision_model, db_path=db_path)

    def process(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
        task: str = "choice",
        goals: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        session_id: str = "default_session",
        tenant_id: str = "default_tenant",
        execute_planning: bool = True,
    ) -> CognitiveEngineResult:
        """Executes the complete coordinated cognitive loop:

        Input -> Perception -> Context -> Memory Retrieval -> World State Update
        -> Reasoning -> Planning -> CDM-1 Central Decision -> Metacognition -> Result.
        """
        start_time = time.perf_counter()
        struct_feats = structured_features or {}
        candidate_list = candidates or ["auto_resolve", "issue_refund", "escalate_to_tier2"]
        prohibitions = list(prohibited_actions or [])
        user_constraints = list(constraints or [])

        # Step A: Perception Intelligence
        perception: PerceptionResult = self.perception_engine.process(
            text=text,
            structured_features=struct_feats,
        )

        # Step B: Contextual Intelligence
        context: ContextFrame = self.context_engine.update_context(
            session_id=session_id,
            tenant_id=tenant_id,
            perception=perception,
            user_message=text,
        )

        # Step C: Memory Retrieval (Episodic, Semantic Rules, Procedural Workflows)
        domain = struct_feats.get("domain", "customer_support")
        rules_objs = self.semantic_memory.query_rules_for_domain(domain=domain, tenant_id=tenant_id)
        semantic_rules = [r.model_dump() for r in rules_objs]
        procedural_workflows = self.procedural_memory.get_workflows_for_domain(domain=domain, tenant_id=tenant_id)

        # Retrieve relevant episodic experiences if embeddings/features available
        episodic_cases = []
        try:
            # Query similar past cases
            sentiment = struct_feats.get("sentiment_score", 0.0)
            urgency_val = 1.0 if struct_feats.get("urgency_level") == "critical" else 0.5
            monetary = min(1.0, float(struct_feats.get("monetary_value_usd", 50.0)) / 500.0)
            mock_vec = [sentiment, urgency_val, monetary] + [0.0] * 13
            episodic_cases = self.episodic_memory.search_similar(
                tenant_id=tenant_id,
                query_vector=mock_vec,
                top_k=3,
                min_sim=0.1,
            )
        except Exception:
            episodic_cases = []

        # Step D: Update World Model
        # Ingest environment metrics
        for k, v in struct_feats.items():
            if isinstance(v, (int, float)):
                self.world_model.set_environment_metric(k, float(v))

        # Ingest extracted entities
        for ent in perception.extracted_entities:
            self.world_model.update_entity(
                entity_id=ent.entity_id,
                entity_type=ent.entity_type,
                properties={"name": ent.name, **ent.attributes},
                uncertainty=1.0 - ent.confidence,
            )

        # Ingest active constraints
        for p in prohibitions:
            self.world_model.register_constraint(f"ProhibitedAction: {p}")
        for c in user_constraints:
            self.world_model.register_constraint(f"UserConstraint: {c}")

        world_state: WorldState = self.world_model.get_state()

        # Step E: Working Memory Setup
        if goals:
            for g_idx, g_desc in enumerate(goals, start=1):
                self.working_memory.push_goal(
                    goal_id=f"goal_{g_idx}",
                    description=g_desc,
                    priority=g_idx,
                )
        else:
            self.working_memory.push_goal(
                goal_id="primary_goal",
                description=f"Resolve request with optimal action under policy compliance for {domain}",
                priority=1,
            )

        self.working_memory.write_scratchpad("session_id", session_id)
        self.working_memory.write_scratchpad("domain", domain)
        working_state: WorkingMemoryState = self.working_memory.snapshot()

        # Step F: Reasoning Intelligence
        reasoning_trace: ReasoningTrace = self.reasoning_engine.reason(
            world_state=world_state,
            context=context,
            semantic_rules=semantic_rules,
            candidate_actions=candidate_list,
            user_constraints=user_constraints,
        )

        # Update prohibitions based on violated hard constraints from reasoning
        effective_prohibitions = list(prohibitions)
        for v in reasoning_trace.violated_constraints:
            for cand in candidate_list:
                if cand.lower() in v.lower() and cand not in effective_prohibitions:
                    effective_prohibitions.append(cand)

        # Step G: Planning Intelligence (when enabled)
        generated_plan: Optional[Plan] = None
        if execute_planning and working_state.active_goals:
            primary_goal = working_state.active_goals[0]
            generated_plan = self.planning_engine.generate_plan(
                target_goal=primary_goal,
                available_actions=candidate_list,
                current_world_state=world_state,
                reasoning_trace=reasoning_trace,
                prohibited_actions=effective_prohibitions,
            )

        # Step H: CDM-1 Central Cognitive Decision Core Execution
        decision = "auto_resolve"
        confidence = 0.90
        calibrated_conf = 0.88
        priority = "MEDIUM"
        risk = 0.05
        operational_score = 85.0
        predicted_outcomes: Dict[str, Any] = {}
        abstain = False
        abstention_reason = None
        alternatives: List[Dict[str, Any]] = []

        if self.decision_core is not None:
            # Execute actual trained CDM-1 model inference
            cdm_result: DecisionResult = self.decision_core.decide(
                text=text,
                structured_features=struct_feats,
                task=task,
                candidates=candidate_list,
                prohibited_actions=effective_prohibitions,
            )
            decision = cdm_result.decision
            confidence = cdm_result.confidence
            calibrated_conf = cdm_result.confidence if cdm_result.calibrated else cdm_result.confidence * 0.9
            priority = cdm_result.priority or "MEDIUM"
            risk = cdm_result.risk or 0.05
            operational_score = cdm_result.score
            predicted_outcomes = {
                "raw_decision": cdm_result.decision,
                "candidate_scores": cdm_result.candidate_scores,
                "predicted_outcome": cdm_result.predicted_outcome,
            }
            abstain = cdm_result.abstain
            abstention_reason = cdm_result.abstention_reason

            if cdm_result.candidate_scores:
                alternatives = [
                    {"action": act, "score": score}
                    for act, score in cdm_result.candidate_scores.items()
                    if act != decision
                ]
        else:
            # Fallback when decision model is unattached
            valid_candidates = [c for c in candidate_list if c not in effective_prohibitions]
            decision = valid_candidates[0] if valid_candidates else "ABSTAIN"
            abstain = decision == "ABSTAIN"
            if abstain:
                abstention_reason = "All prospective candidate actions are prohibited under active constraints."

        # Step I: Metacognitive Review & Safety Gate
        # Check if reasoning discovered severe contradictions or high world uncertainty
        if reasoning_trace.contradictions_found and not abstain:
            if calibrated_conf < 0.75:
                abstain = True
                abstention_reason = f"Metacognitive Abstention: Discovered contradictions ({len(reasoning_trace.contradictions_found)}) with marginal confidence ({calibrated_conf:.3f})."

        # Formulate execution recommendation
        recommendation = "Execute primary recommended action"
        if abstain:
            recommendation = "Escalate to human supervisor for manual intervention due to active abstention policy."
        elif generated_plan and generated_plan.steps:
            recommendation = f"Proceed with plan '{generated_plan.plan_id}' starting with step: {generated_plan.steps[0].action_name}"

        # Step J: Shared Cognitive State Synthesis
        shared_state = SharedCognitiveState(
            perception=perception,
            context=context,
            working_memory=working_state,
            world_state=world_state,
            reasoning=reasoning_trace,
            plan=generated_plan,
            episodic_context=episodic_cases,
            semantic_rules=semantic_rules,
            procedural_workflows=procedural_workflows,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return CognitiveEngineResult(
            decision=decision,
            confidence=round(confidence, 4),
            calibrated_confidence=round(calibrated_conf, 4),
            priority=priority,
            risk=round(risk, 4),
            operational_score=operational_score,
            predicted_outcomes=predicted_outcomes,
            selected_plan=generated_plan,
            reasoning_evidence=reasoning_trace,
            world_state_summary={
                "entity_count": len(world_state.entities),
                "active_constraints": world_state.active_constraints,
                "environment_metrics": world_state.environment_metrics,
                "state_uncertainty": round(world_state.overall_state_uncertainty, 4),
            },
            abstain=abstain,
            abstention_reason=abstention_reason,
            alternatives=alternatives,
            execution_recommendation=recommendation,
            latency_ms=round(latency_ms, 2),
        )

    def record_feedback(
        self,
        decision_id: str,
        predicted_result: CognitiveEngineResult,
        actual_outcome: Dict[str, Any],
        is_success: bool,
    ) -> None:
        """Records telemetry feedback and audits discrepancies for offline dataset enhancement."""
        self.feedback_collector.record_feedback(
            decision_id=decision_id,
            predicted_decision={"decision": predicted_result.decision, "risk": predicted_result.risk},
            predicted_outcome=predicted_result.predicted_outcomes,
            actual_outcome=actual_outcome,
            is_success=is_success,
        )
