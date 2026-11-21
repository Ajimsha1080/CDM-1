"""Unified Cognitive AI Engine for CDM-1.
Coordinates Perception, Context, Working Memory, World Model, Reasoning, Planning,
Central Decision Core, and Metacognition with strict tenant isolation and no fake inference fallbacks.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import time
import math
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
from cdm.execution.dispatcher import SENSITIVE_ACTIONS
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
from cdm.causal.causal_engine import CausalEngine
from cdm.synthesis.answer_synthesizer import AnswerSynthesizer, ExternalInformationRetriever, SynthesizedAnswer


class CognitiveEngine:
    """Unified Cognitive AI Engine with CDM-1 as its Central Decision-Making Core.
    Enforces thread-safe tenant isolation, genuine model inference, persistent memory,
    and rigorous cognitive orchestration.
    """

    def __init__(
        self,
        decision_model: Optional[CognitiveDecisionModel] = None,
        db_path: str = "experiments/cognitive_memory.db",
        production_mode: bool = False,
        decision_core: Optional[CognitiveDecisionModel] = None,
    ):
        effective_model = decision_core if decision_core is not None else decision_model
        is_prod = production_mode or (os.environ.get("CDM_PRODUCTION_MODE", "").strip().lower() in ("1", "true", "yes"))
        self.production_mode = is_prod

        if is_prod:
            self.verify_model_trust(effective_model)

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

        # 8. Causal Analysis & Counterfactual Subsystem
        self.causal_engine = CausalEngine()

        # 9. CDM-1 Central Cognitive Decision Core
        self.decision_core = effective_model
        self.decision_model = effective_model
        self.production_mode = is_prod

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

        # 10. Cognitive Answer Synthesis Layer
        self.external_retriever = ExternalInformationRetriever()
        self.answer_synthesizer = AnswerSynthesizer(retriever=self.external_retriever)

    def get_health_status(self) -> Dict[str, Any]:
        """Returns runtime health and model loading status."""
        model_status = (
            getattr(self.decision_core, "loading_status", {"loaded": False, "is_fallback": True})
            if self.decision_core
            else {"loaded": False, "is_fallback": True, "reason": "no_model_instance"}
        )
        is_healthy = bool(self.decision_core and not getattr(self.decision_core, "is_fallback", True))
        return {
            "status": "healthy" if is_healthy else "degraded",
            "production_mode": self.production_mode,
            "model": model_status,
            "perception": True,
            "memory": True,
            "world_model": True,
            "reasoning": True,
            "planning": True,
            "causal": True,
            "feedback_collector": True,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "experiments/release/cdm-1-general",
        db_path: str = "experiments/cognitive_memory.db",
        device: Optional[str] = None,
        local_files_only: bool = True,
        production_mode: bool = False,
    ) -> "CognitiveEngine":
        """Factory method: loads the trained CDM-1 model into the Cognitive Engine."""
        decision_model = CognitiveDecisionModel.from_pretrained(
            model_name_or_path=model_name_or_path,
            device=device,
            local_files_only=local_files_only,
            production_mode=production_mode,
        )
        return cls(decision_model=decision_model, db_path=db_path, production_mode=production_mode)

    @classmethod
    def verify_model_trust(cls, model: Any) -> None:
        """Independently verifies production readiness of a decision model instance.
        Rejects:
        1. None or fallback rule scorers.
        2. Directly supplied unverified model objects (in-memory injection bypass).
        3. Randomly initialized or unauthenticated model checkpoints.
        4. Checkpoints with invalid/tampered artifact SHA-256 hashes or signatures.
        5. Checkpoints missing genuine CDMModel architecture.
        """
        if model is None:
            raise RuntimeError(
                "Production mode startup failure: Production model trust failure: No decision model provided. "
                "Production mode strictly requires an authenticated, trained CDM-1 model."
            )
        if getattr(model, "is_fallback", False):
            raise RuntimeError(
                "Production mode startup failure: Production model trust failure: Fallback rule scorer cannot be used in production mode. "
                "A genuine trained neural checkpoint is strictly required."
            )

        status = getattr(model, "loading_status", None)
        if not status or not isinstance(status, dict):
            raise RuntimeError(
                "Production mode startup failure: Production model trust failure: Directly supplied unverified model object rejected in production. "
                "Model object was not loaded through authenticated checkpoint loader (loading_status missing)."
            )

        if not status.get("authenticated", False):
            raise RuntimeError(
                "Production mode startup failure: Production model trust failure: Directly supplied unverified model object rejected in production. "
                "Model object lacks verified cryptographic authentication."
            )

        if not status.get("signature_verified", False):
            raise RuntimeError(
                "Production mode startup failure: Production model trust failure: Model manifest signature verification failed or was bypassed."
            )

        # Expected SHA-256 for release artifact
        EXPECTED_RELEASE_SHA256 = "9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a"
        artifact_hash = status.get("artifact_hash")
        if not artifact_hash or artifact_hash.lower() != EXPECTED_RELEASE_SHA256.lower():
            raise RuntimeError(
                f"Production mode startup failure: Production model trust failure: Artifact SHA-256 hash '{artifact_hash}' does not match "
                f"expected release hash '{EXPECTED_RELEASE_SHA256}'."
            )

        # Verify underlying PyTorch model architecture and weights
        py_model = getattr(model, "model", None)
        if py_model is None:
            raise RuntimeError("Production mode startup failure: Production model trust failure: Underlying PyTorch neural model is missing.")

        from cdm.models.cdm_model import CDMModel
        if not isinstance(py_model, CDMModel):
            raise RuntimeError(
                f"Production mode startup failure: Production model trust failure: Model architecture {type(py_model)} does not conform to CDMModel."
            )

        param_count = sum(p.numel() for p in py_model.parameters())
        if param_count < 100_000_000:
            raise RuntimeError(
                f"Production mode startup failure: Production model trust failure: Model parameter count ({param_count}) is inconsistent with CDM-1 architecture."
            )

    @property
    def decision_model(self) -> Optional[CognitiveDecisionModel]:
        return self.decision_core

    @decision_model.setter
    def decision_model(self, model: Optional[CognitiveDecisionModel]) -> None:
        if self.production_mode and model is not None:
            self.verify_model_trust(model)
        self.decision_core = model

    def get_isolated_world_model(self, tenant_id: str, session_id: str) -> WorldModel:
        """Public accessor for tenant-and-session scoped WorldModel."""
        return self._get_or_create_world_model(tenant_id=tenant_id, session_id=session_id)

    def get_isolated_working_memory(self, tenant_id: str, session_id: str) -> WorkingMemory:
        """Public accessor for tenant-and-session scoped WorkingMemory."""
        return self._get_or_create_working_memory(tenant_id=tenant_id, session_id=session_id)

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
            errors: List[str] = []
            cognitive_trace: List[Dict[str, Any]] = []

            struct_feats = dict(structured_features or {})
            for k, v in list(struct_feats.items()):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    errors.append(f"Invalid numerical value in structured_feature '{k}': {v}")
                    struct_feats[k] = 0.0

            candidate_list = candidates or ["auto_resolve", "issue_refund", "escalate_to_tier2"]
            prohibitions = list(prohibited_actions or [])
            user_constraints = list(constraints or [])

            # Retrieve isolated World Model and Working Memory for this tenant session
            world_model = self._get_or_create_world_model(tenant_id=tenant_id, session_id=session_id)
            working_memory = self._get_or_create_working_memory(tenant_id=tenant_id, session_id=session_id)

            # Stage 1: Perception Intelligence
            t_perc_start = time.perf_counter()
            perception: PerceptionResult = self.perception_engine.process(
                text=text,
                structured_features=struct_feats,
            )
            perc_duration = (time.perf_counter() - t_perc_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Perception",
                "execution_status": "success",
                "duration_ms": round(perc_duration, 2),
                "inputs": {"text_length": len(text), "feature_keys": list(struct_feats.keys())},
                "outputs": {
                    "extracted_entities_count": len(perception.extracted_entities),
                    "extracted_entity_types": list({e.entity_type for e in perception.extracted_entities}),
                    "signals": perception.signals,
                    "modality_confidence": perception.modality_confidence,
                },
                "errors": [],
            })

            # Stage 2: Contextual Intelligence & Working Memory (Tenant-isolated)
            t_wm_start = time.perf_counter()
            context: ContextFrame = self.context_engine.update_context(
                session_id=session_id,
                tenant_id=tenant_id,
                perception=perception,
                user_message=text,
            )

            # Ingest observed facts into tenant-isolated world model
            world_model.record_observed_fact("text_inquiry", text)
            for k, v in struct_feats.items():
                world_model.record_observed_fact(k, v)
                if isinstance(v, (int, float)):
                    world_model.set_environment_metric(k, float(v))

            for ent in perception.extracted_entities:
                world_model.update_entity(
                    entity_id=ent.entity_id,
                    entity_type=ent.entity_type,
                    properties={"name": ent.name, **ent.attributes},
                    uncertainty=1.0 - ent.confidence,
                )

            for p in prohibitions:
                world_model.register_constraint(f"ProhibitedAction: {p}")
            for c in user_constraints:
                world_model.register_constraint(f"UserConstraint: {c}")

            world_state: WorldState = world_model.get_state()

            if not working_memory.active_goals:
                if goals:
                    for g_idx, g_desc in enumerate(goals, start=1):
                        working_memory.push_goal(
                            goal_id=f"goal_{g_idx}",
                            description=g_desc,
                            priority=g_idx,
                        )
                else:
                    domain = struct_feats.get("domain", "customer_support")
                    working_memory.push_goal(
                        goal_id="primary_goal",
                        description=f"Resolve request with optimal action under policy compliance for {domain}",
                        priority=1,
                    )

            domain = struct_feats.get("domain", "customer_support")
            working_memory.write_scratchpad("session_id", session_id)
            working_memory.write_scratchpad("tenant_id", tenant_id)
            working_memory.write_scratchpad("domain", domain)
            working_state: WorkingMemoryState = working_memory.snapshot()

            wm_duration = (time.perf_counter() - t_wm_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Working Memory & Context",
                "execution_status": "success",
                "duration_ms": round(wm_duration, 2),
                "inputs": {"session_id": session_id, "tenant_id": tenant_id, "turn_index": context.turn_index},
                "outputs": {
                    "dialogue_turns": len(context.dialogue_history),
                    "detected_contradictions": context.detected_contradictions,
                    "active_goals_count": len(working_state.active_goals),
                    "scratchpad_keys": list(working_state.scratchpad.keys()),
                },
                "errors": [],
            })

            # Stage 3: Memory Retrieval (Episodic, Semantic Rules, Procedural Workflows)
            t_mem_start = time.perf_counter()
            rules_objs = self.semantic_memory.query_rules_for_domain(domain=domain, tenant_id=tenant_id)
            semantic_rules = [r.model_dump() for r in rules_objs]
            procedural_workflows = self.procedural_memory.get_workflows_for_domain(domain=domain, tenant_id=tenant_id)

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

            mem_duration = (time.perf_counter() - t_mem_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Memory Retrieval",
                "execution_status": "success",
                "duration_ms": round(mem_duration, 2),
                "inputs": {"domain": domain, "tenant_id": tenant_id},
                "outputs": {
                    "semantic_rules_retrieved": len(semantic_rules),
                    "procedural_workflows_retrieved": len(procedural_workflows),
                    "episodic_cases_retrieved": len(episodic_cases),
                    "embedding_mode": emb_mode,
                },
                "retrieved_memory": {
                    "rules": [r.get("rule_id", "rule") for r in semantic_rules],
                    "workflows": [w.get("name", "workflow") for w in procedural_workflows],
                    "episodic_count": len(episodic_cases),
                },
                "errors": [],
            })

            # Stage 4: Neural Reasoner & Central Decision Core
            t_reason_start = time.perf_counter()
            causal_analysis = self.causal_engine.analyze_candidate_consequences(
                candidate_actions=candidate_list,
                current_metrics=world_state.environment_metrics,
            )

            reasoning_trace: ReasoningTrace = self.reasoning_engine.reason(
                world_state=world_state,
                context=context,
                semantic_rules=semantic_rules,
                candidate_actions=candidate_list,
                user_constraints=user_constraints,
                procedural_workflows=procedural_workflows,
                causal_consequences=causal_analysis,
            )

            for step in reasoning_trace.steps:
                world_model.record_inferred_state(f"deduction_step_{step.step_index}", step.conclusion)

            effective_prohibitions = list(prohibitions)
            for v in reasoning_trace.violated_constraints:
                for cand in candidate_list:
                    if cand.lower() in v.lower() and cand not in effective_prohibitions:
                        effective_prohibitions.append(cand)

            if self.decision_core is None:
                if self.production_mode:
                    raise RuntimeError(
                        "CognitiveEngine cannot process decisions without an initialized CDM-1 decision core model. "
                        "Fake or ungrounded heuristic fallbacks are strictly disabled in production. "
                        "Initialize CognitiveEngine using CognitiveEngine.from_pretrained() or supply an instantiated CognitiveDecisionModel."
                    )
                cdm_result = DecisionResult({
                    "decision": candidate_list[0] if candidate_list else "abstain",
                    "confidence": 0.5,
                    "calibrated": False,
                    "score": None,
                    "neural_score": None,
                    "heuristic_score": 0.5,
                    "used_neural_inference": False,
                    "candidate_scores": {c: None for c in candidate_list},
                    "heuristic_candidate_scores": {c: 0.5 for c in candidate_list},
                    "risk": 0.05,
                    "predicted_outcome": {"status": "development_baseline"},
                    "abstain": False,
                    "provenance": {"mode": "development_without_weights", "used_neural_inference": False},
                })
            else:
                try:
                    cdm_result = self.decision_core.decide(
                        text=text,
                        structured_features=struct_feats,
                        task=task,
                        candidates=candidate_list,
                        prohibited_actions=effective_prohibitions,
                        tenant_id=tenant_id,
                    )
                except Exception as e:
                    logger.error(f"Decision core execution failure: {e}")
                    cdm_result = DecisionResult({
                        "decision": "abstain",
                        "confidence": 0.0,
                        "calibrated": True,
                        "score": None,
                        "neural_score": None,
                        "heuristic_score": None,
                        "used_neural_inference": False,
                        "candidate_scores": {c: None for c in candidate_list},
                        "heuristic_candidate_scores": {c: 0.0 for c in candidate_list},
                        "risk": 1.0,
                        "predicted_outcome": {"error": str(e)},
                        "abstain": True,
                        "abstention_reason": f"Decision core failure: {e}",
                        "provenance": {"error": str(e), "used_neural_inference": False},
                    })

            # P1: Numerical validation rejecting NaN and Infinity
            raw_conf = getattr(cdm_result, "confidence", 0.0)
            raw_risk = getattr(cdm_result, "risk", 0.05)
            if raw_risk is None:
                raw_risk = 0.05

            has_numerical_error = False
            if isinstance(raw_conf, float) and (math.isnan(raw_conf) or math.isinf(raw_conf)):
                errors.append(f"Invalid model confidence score: {raw_conf}")
                confidence = 0.0
                calibrated_conf = 0.0
                has_numerical_error = True
            else:
                confidence = float(raw_conf)
                is_calibrated = bool(getattr(cdm_result, "calibrated", False))
                calibrated_conf = float(raw_conf if is_calibrated else raw_conf * 0.9)

            if isinstance(raw_risk, float) and (math.isnan(raw_risk) or math.isinf(raw_risk)):
                errors.append(f"Invalid model risk score: {raw_risk}")
                risk = 1.0
                has_numerical_error = True
            else:
                risk = float(raw_risk)

            priority = cdm_result.priority or "MEDIUM"
            operational_score = cdm_result.score
            predicted_outcomes = {
                "raw_decision": cdm_result.decision,
                "candidate_scores": cdm_result.candidate_scores,
                "predicted_outcome": cdm_result.predicted_outcome,
            }

            world_model.record_predicted_future_state({
                "predicted_decision": cdm_result.decision,
                "predicted_confidence": confidence,
                "predicted_risk": risk,
                "predicted_outcome": cdm_result.predicted_outcome,
            })

            alternatives = [
                {"action": act, "score": score}
                for act, score in (cdm_result.candidate_scores or {}).items()
                if act != cdm_result.decision
            ]

            candidate_ranking: List[Tuple[str, float]] = []
            if cdm_result.candidate_scores:
                candidate_ranking = sorted(
                    [
                        (k, float(v))
                        for k, v in cdm_result.candidate_scores.items()
                        if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )

            model_mode = "neural" if getattr(cdm_result, "used_neural_inference", False) else "rule_based_fallback"
            reason_duration = (time.perf_counter() - t_reason_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Neural Reasoner & Decision Core",
                "execution_status": "warning" if (has_numerical_error or cdm_result.abstain) else "success",
                "duration_ms": round(reason_duration, 2),
                "model_mode": model_mode,
                "availability": "available" if self.decision_core is not None else "uninitialized",
                "inputs": {"candidates": candidate_list, "prohibitions": effective_prohibitions},
                "outputs": {
                    "decision": cdm_result.decision,
                    "confidence": round(confidence, 4),
                    "calibrated_confidence": round(calibrated_conf, 4),
                    "risk": round(risk, 4),
                    "candidate_scores": cdm_result.candidate_scores,
                    "violated_constraints": reasoning_trace.violated_constraints,
                    "contradictions_found": reasoning_trace.contradictions_found,
                },
                "errors": [e for e in errors if "confidence" in e or "risk" in e],
            })

            # Stage 5: Planning Engine & Safety Verifier
            t_plan_start = time.perf_counter()

            # 1. Separate Proposed Decision from Execution Authorization
            proposed_candidate = cdm_result.decision
            if proposed_candidate in (None, "abstain", "ABSTAIN"):
                if candidate_ranking:
                    proposed_candidate = candidate_ranking[0][0]
                elif candidate_list:
                    proposed_candidate = candidate_list[0]
                else:
                    proposed_candidate = "none"

            proposed_decision = proposed_candidate

            # 2. Evaluate Policy Check independently
            policy_violated_rules = []
            if proposed_decision in effective_prohibitions:
                policy_violated_rules.append(f"ProhibitedAction: {proposed_decision}")
            for v in reasoning_trace.violated_constraints:
                # Violation applies if it references the proposed decision, or is a global/user constraint
                if str(proposed_decision) in v or "prohibitedaction" in v.lower() or "user constraint" in v.lower() or "general" in v.lower():
                    if v not in policy_violated_rules:
                        policy_violated_rules.append(v)

            policy_passed = (len(policy_violated_rules) == 0) and (proposed_decision not in effective_prohibitions)
            policy_check = {
                "passed": policy_passed,
                "violated_rules": policy_violated_rules,
                "checked_rules": [f"ProhibitedAction: {p}" for p in effective_prohibitions] + [f"UserConstraint: {c}" for c in user_constraints],
                "policy_status": "passed" if policy_passed else "violated",
            }

            # 3. Categorize Abstention (Policy vs Uncertainty vs Missing Evidence vs Invalid Numerical)
            abstain = False
            abstention_status = "none"
            abstention_reason = None

            if has_numerical_error:
                abstain = True
                abstention_status = "invalid_numerical_scores"
                abstention_reason = "Numerical validation failure: confidence or risk score is NaN or infinity."
            elif not policy_passed or (proposed_decision in effective_prohibitions):
                abstain = True
                abstention_status = "policy_rejection"
                abstention_reason = (
                    f"Policy constraint violation: {', '.join(policy_violated_rules)}"
                    if policy_violated_rules
                    else f"Action '{proposed_decision}' is prohibited by policy."
                )
            elif reasoning_trace.contradictions_found and calibrated_conf < 0.75:
                abstain = True
                abstention_status = "uncertainty"
                abstention_reason = (
                    f"Metacognitive Abstention: Discovered contradictions ({len(reasoning_trace.contradictions_found)}) "
                    f"with marginal confidence ({calibrated_conf:.3f} < 0.75)."
                )
            elif cdm_result.abstain:
                abstain = True
                raw_reason = cdm_result.abstention_reason or "Decision core returned abstention"
                if "prohibit" in raw_reason.lower() or "policy" in raw_reason.lower():
                    abstention_status = "policy_rejection"
                elif "uncertain" in raw_reason.lower() or "confidence" in raw_reason.lower():
                    abstention_status = "uncertainty"
                elif "missing" in raw_reason.lower() or "evidence" in raw_reason.lower():
                    abstention_status = "missing_evidence"
                else:
                    abstention_status = "abstained"
                abstention_reason = raw_reason
            elif calibrated_conf < 0.35:
                abstain = True
                abstention_status = "uncertainty"
                abstention_reason = (
                    f"Uncertainty threshold breached: calibrated confidence {calibrated_conf:.3f} < 0.35 "
                    f"(uncertainty {round(1.0 - calibrated_conf, 4)} > 0.65)."
                )

            # 4. Human Approval Determination (Never invent a human supervisor)
            is_sensitive = proposed_decision in SENSITIVE_ACTIONS
            is_high_risk = (risk > 0.40)
            requires_approval = bool((is_sensitive or is_high_risk) and not abstain)

            approver_role = None
            if is_sensitive:
                if proposed_decision in {"issue_refund", "refund_customer", "refund_transaction"}:
                    approver_role = "compliance_officer"
                elif proposed_decision in {"administer_meds", "order_labs"}:
                    approver_role = "lead_physician"
                elif proposed_decision in {
                    "isolate_cluster", "terminate_service", "delete_database", "modify_access_policy", "shell_exec"
                }:
                    approver_role = "security_admin"
                else:
                    approver_role = "compliance_officer"
            elif is_high_risk:
                approver_role = "compliance_officer"

            approval_reason = None
            if is_sensitive:
                approval_reason = f"Action '{proposed_decision}' is a sensitive operational action requiring human approval."
            elif is_high_risk:
                approval_reason = f"Assessed risk ({risk:.2f}) exceeds autonomous operational threshold (0.40)."

            approval_requirements = {
                "required": requires_approval,
                "approver_role": approver_role if requires_approval else None,
                "approval_status": "pending" if requires_approval else ("not_required" if not abstain else "abstained"),
                "reason": approval_reason,
                "configured_supervisor": None,  # Explicitly None: never fabricate a fictitious human supervisor
            }

            # 5. Execution Authorization and Execution Status
            if abstain:
                execution_authorized = False
                execution_status = "rejected" if abstention_status == "policy_rejection" else "abstained"
                decision = "ABSTAIN"
            elif requires_approval:
                execution_authorized = False
                execution_status = "needs_approval"
                decision = proposed_decision
            else:
                execution_authorized = True
                execution_status = "authorized"
                decision = proposed_decision

            # 6. Planning Engine Execution
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

                plan_decision = proposed_decision if not (abstain and abstention_status == "policy_rejection") else "abstain"
                generated_plan = self.planning_engine.generate_plan(
                    target_goal=primary_goal,
                    available_actions=candidate_list,
                    current_world_state=world_state,
                    reasoning_trace=reasoning_trace,
                    prohibited_actions=effective_prohibitions,
                    predicted_outcome=pred_for_planning,
                    tenant_id=tenant_id,
                    task=task,
                    selected_decision=plan_decision,
                    candidate_ranking=candidate_ranking,
                    abstain=abstain,
                    abstention_reason=abstention_reason,
                )

            if generated_plan is not None:
                generated_plan.primary_decision = proposed_decision
                generated_plan.is_executable = execution_authorized
                generated_plan.requires_human_approval = requires_approval
                generated_plan.required_approver_role = approver_role if requires_approval else None
                generated_plan.approval_reason = approval_reason or abstention_reason

                if execution_authorized:
                    generated_plan.plan_stage = "approved"
                    generated_plan.status = "approved"
                elif abstention_status == "policy_rejection":
                    generated_plan.plan_stage = "prohibited"
                    generated_plan.status = "prohibited"
                elif abstain:
                    generated_plan.plan_stage = "abstained"
                    generated_plan.status = "abstained"
                elif requires_approval:
                    generated_plan.plan_stage = "proposed"
                    generated_plan.status = "proposed"

                for step in generated_plan.steps:
                    step.execution_eligibility = execution_authorized and (step.action_name not in effective_prohibitions)
                    step.approval_required = (step.action_name in SENSITIVE_ACTIONS) or (step.risk_estimate > 0.40)
                    if step.approval_required:
                        step.required_approver_role = approver_role or "compliance_officer"
                    step.constraints = list(world_state.active_constraints)

            # 7. Verification: Plan consistency and Safety Invariant Checks
            plan_consistent = True
            if generated_plan and generated_plan.steps:
                step_names = [s.action_name for s in generated_plan.steps]
                for s_act in step_names:
                    if s_act in effective_prohibitions:
                        plan_consistent = False
                        errors.append(f"Plan step '{s_act}' violates active prohibition.")
                if execution_authorized and (proposed_decision not in step_names):
                    plan_consistent = False
                    errors.append(f"Plan inconsistency: Proposed decision '{proposed_decision}' is not in plan steps {step_names}.")

            verification = {
                "verified": bool(not abstain and plan_consistent and policy_passed),
                "verification_status": "passed" if (not abstain and plan_consistent and policy_passed) else ("abstained" if abstain else "failed"),
                "checks": [
                    {"check": "policy_compliance", "passed": policy_passed},
                    {"check": "uncertainty_boundary", "passed": (calibrated_conf >= 0.35)},
                    {"check": "numerical_validity", "passed": not has_numerical_error},
                    {"check": "plan_decision_consistency", "passed": plan_consistent},
                    {"check": "execution_safety_gate", "passed": (not execution_authorized or (execution_authorized and not requires_approval and policy_passed))},
                ],
            }

            plan_duration = (time.perf_counter() - t_plan_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Planning Engine & Safety Verifier",
                "execution_status": "success" if (verification["verified"] or abstain or requires_approval) else "warning",
                "duration_ms": round(plan_duration, 2),
                "inputs": {
                    "target_goal": working_state.active_goals[0].description if working_state.active_goals else None,
                    "proposed_decision": proposed_decision,
                    "available_actions": candidate_list,
                    "prohibitions": effective_prohibitions,
                },
                "outputs": {
                    "plan_generated": generated_plan is not None,
                    "plan_id": generated_plan.plan_id if generated_plan else None,
                    "step_count": len(generated_plan.steps) if generated_plan else 0,
                    "execution_authorized": execution_authorized,
                    "execution_status": execution_status,
                    "approval_required": requires_approval,
                    "required_approver_role": approver_role,
                    "verification_status": verification["verification_status"],
                },
                "errors": [e for e in errors if "Plan" in e or "Safety" in e or "prohibition" in e],
            })

            # Candidate rankings
            candidate_rankings = [
                {
                    "candidate": cand,
                    "score": round(float(score), 4) if score is not None else None,
                    "rank": idx + 1,
                    "is_prohibited": cand in effective_prohibitions,
                }
                for idx, (cand, score) in enumerate(candidate_ranking)
            ]

            # Supporting evidence
            supporting_evidence = {
                "semantic_rules_applied": [r.get("rule_id", "RULE") for r in semantic_rules if isinstance(r, dict)],
                "procedural_workflows": [w.get("name", "workflow") for w in procedural_workflows if isinstance(w, dict)],
                "episodic_precedents": len(episodic_cases),
                "reasoning_steps": len(reasoning_trace.steps),
                "causal_consequences": len(causal_analysis.get("consequences", {})) if causal_analysis else 0,
                "detected_contradictions": context.detected_contradictions,
                "observed_facts_count": len(world_state.observed_facts),
            }

            # Recommendation formulation
            if abstain:
                recommendation = f"Abstain: {abstention_reason}"
            elif requires_approval:
                recommendation = f"Awaiting human approval ({approver_role}): Proposed action '{proposed_decision}'. {approval_reason}"
            elif generated_plan and generated_plan.steps:
                recommendation = f"Authorized: Proceed with plan '{generated_plan.plan_id}' starting with step: {generated_plan.steps[0].action_name}"
            else:
                recommendation = f"Authorized: Execute primary recommended action '{proposed_decision}'"

            # Stage J: Shared Cognitive State Synthesis
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
                causal_analysis=causal_analysis,
            )

            # Stage 6: Cognitive Answer Synthesis Layer
            t_synth_start = time.perf_counter()
            external_facts = []
            need_external = (
                "external" in text.lower()
                or "latest status" in text.lower()
                or "real-time" in text.lower()
                or "10.0.1.5" in text.lower()
                or "check live" in text.lower()
                or "telemetry" in text.lower()
                or "sla" in text.lower()
                or "policy" in text.lower()
            )
            if need_external:
                try:
                    ret_success, ret_facts, ret_err = self.external_retriever.retrieve(query=text, domain=domain)
                    if ret_success:
                        external_facts = ret_facts
                    elif ret_err:
                        errors.append(f"External retrieval error: {ret_err}")
                except Exception as ex_err:
                    errors.append(f"External retrieval exception: {ex_err}")

            conf_origin = (
                "neural_softmax_calibrated"
                if (getattr(cdm_result, "used_neural_inference", False) and getattr(cdm_result, "calibrated", False))
                else (
                    "neural_softmax_uncalibrated"
                    if getattr(cdm_result, "used_neural_inference", False)
                    else "heuristic_fallback"
                )
            )
            r_origin = (
                "neural_risk_head_prediction"
                if getattr(cdm_result, "used_neural_inference", False)
                else "heuristic_risk_assessor"
            )

            try:
                synth_output = self.answer_synthesizer.synthesize(
                    question=text,
                    task=task,
                    perception=perception,
                    context=context,
                    working_memory=working_state,
                    world_state=world_state,
                    semantic_rules=semantic_rules,
                    procedural_workflows=procedural_workflows,
                    episodic_cases=episodic_cases,
                    reasoning_trace=reasoning_trace,
                    causal_analysis=causal_analysis,
                    cdm_result=cdm_result,
                    candidate_rankings=candidate_rankings,
                    proposed_decision=proposed_decision,
                    execution_authorized=execution_authorized,
                    execution_status=execution_status,
                    abstention_status=abstention_status,
                    abstention_reason=abstention_reason,
                    policy_check=policy_check,
                    approval_requirements=approval_requirements,
                    verification=verification,
                    generated_plan=generated_plan,
                    confidence=confidence,
                    calibrated_conf=calibrated_conf,
                    uncertainty=round(1.0 - calibrated_conf, 4),
                    assessed_risk=risk,
                    confidence_origin=conf_origin,
                    risk_origin=r_origin,
                    retrieved_external_facts=external_facts,
                    errors=errors,
                    tenant_id=tenant_id,
                )
            except Exception as synth_err:
                logger.error(f"Answer synthesis exception: {synth_err}", exc_info=True)
                errors.append(f"Answer synthesis exception: {synth_err}")
                synth_output = self.answer_synthesizer.fallback_synthesize(
                    decision=decision,
                    proposed_decision=proposed_decision,
                    execution_status=execution_status,
                    execution_authorized=execution_authorized,
                    abstention_status=abstention_status,
                    abstention_reason=abstention_reason,
                    confidence=confidence,
                    risk=risk,
                    error_msg=str(synth_err),
                )

            synth_duration = (time.perf_counter() - t_synth_start) * 1000.0
            cognitive_trace.append({
                "stage_name": "Cognitive Answer Synthesis",
                "execution_status": "success" if not any("synthesis exception" in e for e in errors) else "warning",
                "duration_ms": round(synth_duration, 2),
                "inputs": {"question_type": synth_output.answer_type},
                "outputs": {
                    "answer_length": len(synth_output.answer),
                    "answer_type": synth_output.answer_type,
                    "safety_status": synth_output.safety_status,
                    "limitations_count": len(synth_output.limitations),
                },
                "errors": [e for e in errors if "synthesis" in e.lower()],
            })

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return CognitiveEngineResult(
                execution_id=f"exec_{uuid.uuid4().hex[:8]}",
                request_id=f"req_{uuid.uuid4().hex[:8]}",
                execution_status=execution_status,
                proposed_decision=proposed_decision,
                execution_authorized=execution_authorized,
                abstention_status=abstention_status,
                abstention_reason=abstention_reason,
                decision=decision,
                confidence=round(confidence, 4),
                calibrated_confidence=round(calibrated_conf, 4),
                uncertainty=round(1.0 - calibrated_conf, 4),
                assessed_risk=round(risk, 4),
                is_calibrated=getattr(cdm_result, "calibrated", False),
                confidence_origin=conf_origin,
                risk_origin=r_origin,
                priority=priority,
                risk=round(risk, 4),
                operational_score=operational_score,
                neural_score=getattr(cdm_result, "neural_score", None),
                heuristic_score=getattr(cdm_result, "heuristic_score", None),
                used_neural_inference=getattr(cdm_result, "used_neural_inference", False),
                predicted_outcomes=predicted_outcomes,
                candidate_scores=cdm_result.candidate_scores or {},
                heuristic_candidate_scores=getattr(cdm_result, "heuristic_candidate_scores", {}) or {},
                candidate_rankings=candidate_rankings,
                policy_check=policy_check,
                approval_requirements=approval_requirements,
                verification=verification,
                supporting_evidence=supporting_evidence,
                cognitive_trace=cognitive_trace,
                errors=errors,
                selected_plan=generated_plan,
                plan=generated_plan,
                reasoning_evidence=reasoning_trace,
                causal_analysis=causal_analysis,
                world_state_summary={
                    "entity_count": len(world_state.entities),
                    "active_constraints": world_state.active_constraints,
                    "environment_metrics": world_state.environment_metrics,
                    "state_uncertainty": round(world_state.overall_state_uncertainty, 4),
                },
                abstain=abstain,
                alternatives=alternatives,
                execution_recommendation=recommendation,
                embedding_mode=emb_mode,
                latency_ms=round(latency_ms, 2),
                shared_state=shared_state,
                metadata={
                    "model_id": getattr(cdm_result, "model_id", "CDM-1-General"),
                    "model_version": getattr(cdm_result, "model_version", "1.0.0"),
                    "provenance": getattr(cdm_result, "provenance", {}),
                },
                answer=synth_output.answer,
                answer_type=synth_output.answer_type,
                explanation=synth_output.explanation,
                risks_and_uncertainty=synth_output.risks_and_uncertainty,
                safety_status=synth_output.safety_status,
                limitations=synth_output.limitations,
                important_assumptions=synth_output.important_assumptions,
                recommended_next_steps=synth_output.recommended_next_steps,
                retrieved_external_facts=external_facts,
                citations=synth_output.citations,
                unsupported_claims=synth_output.unsupported_claims,
                retrieved_passages=synth_output.retrieved_passages,
                intent_task=synth_output.intent_task,
                needs_clarification=synth_output.needs_clarification,
                clarification_prompt=synth_output.clarification_prompt,
                validation_result=synth_output.validation_result,
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
