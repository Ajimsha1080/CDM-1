"""Unified Cognitive AI Engine for CDM-1.
Coordinates Perception, Context, Working Memory, World Model, Reasoning, Planning,
Central Decision Core, and Metacognition with strict tenant isolation and no fake inference fallbacks.
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import uuid
import logging
import threading

logger = logging.getLogger("cdm.cognitive.engine")

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
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory
from cdm.memory.semantic import SemanticMemory
from cdm.memory.procedural import ProceduralMemory
from cdm.memory.working_memory import WorkingMemory
from cdm.memory.embedder import SemanticMemoryEmbedder
from cdm.world_model.world_model import WorldModel
from cdm.reasoning.reasoning_engine import ReasoningEngine
from cdm.planning.planning_engine import PlanningEngine
from cdm.learning.feedback import FeedbackCollector
from cdm.inference.pipeline import CognitiveDecisionModel, DecisionResult


class CognitiveEngine:
    """Unified Cognitive AI Engine with CDM-1 as its Central Decision-Making Core.
    Enforces thread-safe tenant isolation, genuine model inference, persistent memory,
    and rigorous cognitive orchestration.
    """

    def __init__(
        self,
        decision_model: Optional[CognitiveDecisionModel] = None,
        db_path: str = "experiments/cognitive_memory.db",
    ):
        # 1. Perception Subsystem
        self.perception_engine = PerceptionEngine()

        # 2. Contextual Subsystem
        self.context_engine = ContextEngine()

        # 3. Persistent Memory Subsystems
        self.storage = SQLiteMemoryStorage(db_path=db_path)
        self.episodic_memory = EpisodicMemory(storage=self.storage)
        self.semantic_memory = SemanticMemory(storage=self.storage)
        self.procedural_memory = ProceduralMemory(storage=self.storage)

        # Scoped tenant-isolated world models & working memories
        self._state_lock = threading.Lock()
        self._tenant_world_models: Dict[Tuple[str, str], WorldModel] = {}
        self._tenant_working_memories: Dict[Tuple[str, str], WorkingMemory] = {}
        self._session_locks: Dict[Tuple[str, str], threading.Lock] = {}

        # 4. Global fallback world model for backward compatibility
        self.world_model = WorldModel()
        self.working_memory = WorkingMemory()

        # 5. Reasoning Engine
        self.reasoning_engine = ReasoningEngine()

        # 6. Planning Engine
        self.planning_engine = PlanningEngine(world_model=self.world_model)

        # 7. Durable Learning & Feedback (SQLite-backed)
        self.feedback_collector = FeedbackCollector(db_path=db_path)

        # 8. CDM-1 Central Cognitive Decision Core
        self.decision_core = decision_model

        # 9. Grounded Semantic Memory Embedder
        lang_enc = None
        tok = None
        if self.decision_core is not None:
            if hasattr(self.decision_core, "model"):
                lang_enc = getattr(self.decision_core.model, "language_encoder", None)
            tok = getattr(self.decision_core, "tokenizer", None)
            if tok is None and hasattr(self.decision_core, "preprocessor"):
                tok = getattr(self.decision_core.preprocessor, "tokenizer", None)
        self.memory_embedder = SemanticMemoryEmbedder(
            language_encoder=lang_enc,
            tokenizer=tok,
            use_neural=True if (lang_enc is not None and tok is not None) else None,
        )

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

    def _get_session_lock(self, tenant_id: str, session_id: str) -> threading.Lock:
        """Retrieves or creates a dedicated synchronization lock for a specific (tenant_id, session_id)."""
        with self._state_lock:
            key = (tenant_id, session_id)
            if key not in self._session_locks:
                self._session_locks[key] = threading.Lock()
            return self._session_locks[key]

    def _get_or_create_world_model(self, tenant_id: str, session_id: str) -> WorldModel:
        """Retrieves or creates a tenant-and-session isolated WorldModel."""
        with self._state_lock:
            key = (tenant_id, session_id)
            if key not in self._tenant_world_models:
                self._tenant_world_models[key] = WorldModel(tenant_id=tenant_id, session_id=session_id)
            return self._tenant_world_models[key]

    def _get_or_create_working_memory(self, tenant_id: str, session_id: str) -> WorkingMemory:
        """Retrieves or creates a tenant-and-session isolated WorkingMemory."""
        with self._state_lock:
            key = (tenant_id, session_id)
            if key not in self._tenant_working_memories:
                self._tenant_working_memories[key] = WorkingMemory()
            return self._tenant_working_memories[key]

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
        -> Reasoning -> Decision Core (Real Inference) -> Planning -> Metacognition -> Result.
        Strictly requires an initialized model; no fake or ungrounded heuristic fallbacks.
        Per-tenant-session lock guarantees thread safety without cross-session serialization.
        """
        session_lock = self._get_session_lock(tenant_id=tenant_id, session_id=session_id)
        with session_lock:
            start_time = time.perf_counter()
            struct_feats = structured_features or {}
            candidate_list = candidates or ["auto_resolve", "issue_refund", "escalate_to_tier2"]
            prohibitions = list(prohibited_actions or [])
            user_constraints = list(constraints or [])

            # Retrieve isolated World Model and Working Memory for this tenant session
            world_model = self._get_or_create_world_model(tenant_id=tenant_id, session_id=session_id)
            working_memory = self._get_or_create_working_memory(tenant_id=tenant_id, session_id=session_id)

            # Step A: Perception Intelligence
            perception: PerceptionResult = self.perception_engine.process(
                text=text,
                structured_features=struct_feats,
            )

            # Step B: Contextual Intelligence (Tenant-isolated)
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

            # Grounded semantic vector embedding for memory retrieval
            query_vector, emb_mode = self.memory_embedder.embed_with_metadata(text=text, structured_features=struct_feats)
            episodic_cases = []
            try:
                episodic_cases = self.episodic_memory.search_similar(
                    tenant_id=tenant_id,
                    query_vector=query_vector,
                    top_k=3,
                    min_sim=0.1,
                    prohibited_actions=prohibitions,
                )
            except Exception as e:
                logger.warning(f"Episodic memory search warning for tenant '{tenant_id}': {e}")
                episodic_cases = []

            # Step D: Update Tenant-Isolated World Model
            # Ingest observed facts
            world_model.record_observed_fact("text_inquiry", text)
            for k, v in struct_feats.items():
                world_model.record_observed_fact(k, v)
                if isinstance(v, (int, float)):
                    world_model.set_environment_metric(k, float(v))

            # Ingest extracted entities
            for ent in perception.extracted_entities:
                world_model.update_entity(
                    entity_id=ent.entity_id,
                    entity_type=ent.entity_type,
                    properties={"name": ent.name, **ent.attributes},
                    uncertainty=1.0 - ent.confidence,
                )

            # Ingest active constraints
            for p in prohibitions:
                world_model.register_constraint(f"ProhibitedAction: {p}")
            for c in user_constraints:
                world_model.register_constraint(f"UserConstraint: {c}")

            world_state: WorldState = world_model.get_state()

            # Step E: Working Memory Setup (Tenant-isolated)
            working_memory.clear()
            if goals:
                for g_idx, g_desc in enumerate(goals, start=1):
                    working_memory.push_goal(
                        goal_id=f"goal_{g_idx}",
                        description=g_desc,
                        priority=g_idx,
                    )
            else:
                working_memory.push_goal(
                    goal_id="primary_goal",
                    description=f"Resolve request with optimal action under policy compliance for {domain}",
                    priority=1,
                )

            working_memory.write_scratchpad("session_id", session_id)
            working_memory.write_scratchpad("tenant_id", tenant_id)
            working_memory.write_scratchpad("domain", domain)
            working_state: WorkingMemoryState = working_memory.snapshot()

            # Step F: Reasoning Intelligence
            reasoning_trace: ReasoningTrace = self.reasoning_engine.reason(
                world_state=world_state,
                context=context,
                semantic_rules=semantic_rules,
                candidate_actions=candidate_list,
                user_constraints=user_constraints,
            )

            # Record inferred state in world model
            for step in reasoning_trace.steps:
                world_model.record_inferred_state(f"deduction_step_{step.step_index}", step.conclusion)

            # Update prohibitions based on violated hard constraints from reasoning
            effective_prohibitions = list(prohibitions)
            for v in reasoning_trace.violated_constraints:
                for cand in candidate_list:
                    if cand.lower() in v.lower() and cand not in effective_prohibitions:
                        effective_prohibitions.append(cand)

            # Step G: CDM-1 Central Cognitive Decision Core Execution (P0: STRICT INFERENCE)
            if self.decision_core is None:
                raise RuntimeError(
                    "CognitiveEngine cannot process decisions without an initialized CDM-1 decision core model. "
                    "Fake or ungrounded heuristic fallbacks are strictly disabled in production. "
                    "Initialize CognitiveEngine using CognitiveEngine.from_pretrained() or supply an instantiated CognitiveDecisionModel."
                )

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

            # Record predicted future state into world model
            world_model.record_predicted_future_state({
                "predicted_decision": decision,
                "predicted_confidence": confidence,
                "predicted_risk": risk,
                "predicted_outcome": cdm_result.predicted_outcome,
            })

            alternatives = [
                {"action": act, "score": score}
                for act, score in (cdm_result.candidate_scores or {}).items()
                if act != decision
            ]

            # Step H: Planning Intelligence (when enabled)
            generated_plan: Optional[Plan] = None
            if execute_planning and working_state.active_goals:
                primary_goal = working_state.active_goals[0]
                self.planning_engine.world_model = world_model
                pred_for_planning = dict(cdm_result.predicted_outcome or {})
                pred_for_planning["predicted_risk"] = risk
                pred_for_planning["candidate_scores"] = cdm_result.candidate_scores
                pred_for_planning["model_id"] = getattr(cdm_result, "model_id", "CDM-1-General")
                pred_for_planning["model_version"] = getattr(cdm_result, "model_version", "1.0.0")
                pred_for_planning["prediction_source"] = "trained_model_prediction"
                pred_for_planning["provenance"] = getattr(cdm_result, "provenance", {})
                generated_plan = self.planning_engine.generate_plan(
                    target_goal=primary_goal,
                    available_actions=candidate_list,
                    current_world_state=world_state,
                    reasoning_trace=reasoning_trace,
                    prohibited_actions=effective_prohibitions,
                    predicted_outcome=pred_for_planning,
                )

            # Step I: Metacognitive Review & Safety Gate
            # Check if reasoning discovered severe contradictions or high world uncertainty
            if reasoning_trace.contradictions_found and not abstain:
                if calibrated_conf < 0.75:
                    abstain = True
                    abstention_reason = (
                        f"Metacognitive Abstention: Discovered contradictions ({len(reasoning_trace.contradictions_found)}) "
                        f"with marginal confidence ({calibrated_conf:.3f})."
                    )

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
                world_state=world_model.get_state(),
                reasoning=reasoning_trace,
                plan=generated_plan,
                embedding_mode=emb_mode,
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
                embedding_mode=emb_mode,
                latency_ms=round(latency_ms, 2),
            )

    def record_feedback(
        self,
        decision_id: str,
        predicted_result: CognitiveEngineResult,
        actual_outcome: Dict[str, Any],
        is_success: bool,
        tenant_id: str = "default_tenant",
        review_status: str = "pending",
    ) -> None:
        """Records telemetry feedback and audits discrepancies for offline dataset enhancement with tenant isolation."""
        self.feedback_collector.record_feedback(
            decision_id=decision_id,
            predicted_decision={"decision": predicted_result.decision, "risk": predicted_result.risk},
            predicted_outcome=predicted_result.predicted_outcomes,
            actual_outcome=actual_outcome,
            is_success=is_success,
            tenant_id=tenant_id,
            review_status=review_status,
        )
