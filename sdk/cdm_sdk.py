"""CDM-1 Python SDK (cdm_sdk).
Provides an explicit client for either Local or Hosted Commercial API inference
with strict mode enforcement (no silent mode fallback), typed responses,
retries with backoff, and robust error handling.
"""

from typing import Dict, List, Optional, Any, Union
import time
import requests
from pydantic import BaseModel, Field


class CDMResponse(BaseModel):
    decision: Any
    confidence: float
    task: str
    calibrated: bool = True
    score: Optional[float] = None
    priority: Optional[str] = None
    risk: Optional[float] = None
    abstain: bool = False
    selected_candidate: Optional[str] = None
    candidate_scores: Optional[Dict[str, float]] = None
    predicted_outcome: Optional[Dict[str, Any]] = None


class CognitiveEngineResponse(BaseModel):
    # Core Contract v1.2.0
    schema_version: str = "v1.2.0"
    answer: str = ""
    decision: Any = None
    reasoning: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    safety_status: str = "verified"
    abstention_reason: Optional[str] = None
    required_approval: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_status: str = "proposed"
    cognitive_trace: List[Dict[str, Any]] = Field(default_factory=list)
    latency: Dict[str, Any] = Field(default_factory=dict)

    # Extended & Legacy Fields for Full Backward Compatibility
    execution_id: str = ""
    request_id: str = ""
    proposed_decision: Any = None
    execution_authorized: bool = False
    calibrated_confidence: float = 0.0
    priority: str = "MEDIUM"
    risk: float = 0.0
    operational_score: Optional[float] = None
    predicted_outcomes: Dict[str, Any] = Field(default_factory=dict)
    selected_plan: Optional[Dict[str, Any]] = None
    reasoning_evidence: Dict[str, Any] = Field(default_factory=dict)
    world_state_summary: Dict[str, Any] = Field(default_factory=dict)
    abstain: bool = False
    execution_recommendation: str = ""
    latency_ms: float = 0.0
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)
    answer_type: str = "general"
    explanation: str = ""
    risks_and_uncertainty: str = ""
    candidate_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    candidate_rankings: List[Dict[str, Any]] = Field(default_factory=list)


class CDMError(Exception):
    """Base exception for CDM SDK operations."""
    pass


class CDMAuthenticationError(CDMError):
    """Raised on 401 Unauthorized API responses."""
    pass


class CDMRateLimitError(CDMError):
    """Raised on 429 Rate Limit Exceeded API responses."""
    pass


class CDM:
    """Client for CDM-1 decision models with explicit mode enforcement."""

    def __init__(
        self,
        mode: str = "auto",
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        local_model_path: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        local_files_only: bool = True,
    ):
        """Args:

            mode: 'local', 'hosted', or 'auto' (resolves to local if path provided, else hosted).
            api_key: API token for hosted endpoint.
            base_url: Base URL for hosted REST API.
            local_model_path: Checkpoint directory for local inference.
            timeout: Network request timeout in seconds.
            max_retries: Retry attempts on transient network errors.
            local_files_only: Enforce strictly offline loading for local execution.
        """
        if mode not in {"local", "hosted", "auto"}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'local', 'hosted', or 'auto'.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

        # Determine concrete execution mode
        if mode == "local":
            self.mode = "local"
        elif mode == "hosted":
            self.mode = "hosted"
        else:
            self.mode = "local" if local_model_path is not None else "hosted"

        self._local_model = None
        self._local_cognitive_engine = None
        if self.mode == "local":
            path = local_model_path or "experiments/release/cdm-1-general"
            from cdm.inference.pipeline import CognitiveDecisionModel
            from cdm.cognitive.engine import CognitiveEngine
            self._local_model = CognitiveDecisionModel.from_pretrained(
                path,
                local_files_only=local_files_only,
            )
            self._local_cognitive_engine = CognitiveEngine(decision_model=self._local_model)

    def decide(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        task: str = "binary",
        candidates: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
    ) -> CDMResponse:
        """Submit a decision request according to configured execution mode."""
        if self.mode == "local":
            if self._local_model is None:
                raise CDMError("Local model runtime is not initialized.")
            res = self._local_model.decide(
                text=text,
                structured_features=structured_features,
                task=task,
                candidates=candidates,
                prohibited_actions=prohibited_actions,
            )
            return CDMResponse.model_validate(res.to_dict())

        # Hosted API execution
        payload = {
            "text": text,
            "structured_features": structured_features or {},
            "task": task,
            "candidates": candidates,
            "prohibited_actions": prohibited_actions,
        }
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/decisions"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return CDMResponse.model_validate(data)
                elif resp.status_code == 401:
                    raise CDMAuthenticationError("API Key authentication rejected (HTTP 401).")
                elif resp.status_code == 429:
                    if attempt == self.max_retries - 1:
                        raise CDMRateLimitError("Rate limit exceeded (HTTP 429).")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    raise CDMError(f"API Error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise CDMError(f"Connection failure to CDM API at {url}: {e}") from e
                time.sleep(0.5 * (attempt + 1))

        raise CDMError("Exhausted maximum retry attempts.")

    def batch_decide(self, items: List[Dict[str, Any]]) -> List[CDMResponse]:
        """Process multiple decision requests."""
        results = []
        for item in items:
            res = self.decide(**item)
            results.append(res)
        return results

    def process(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        task: str = "choice",
        candidates: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
        goals: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        session_id: str = "default_session",
    ) -> CognitiveEngineResponse:
        """Executes the complete coordinated Cognitive AI Engine pipeline."""
        if self.mode == "local":
            if self._local_cognitive_engine is None:
                raise CDMError("Local Cognitive Engine runtime is not initialized.")
            res = self._local_cognitive_engine.process(
                text=text,
                structured_features=structured_features,
                task=task,
                candidates=candidates,
                prohibited_actions=prohibited_actions,
                goals=goals,
                constraints=constraints,
                session_id=session_id,
            )
            plan_dict = res.selected_plan.model_dump() if res.selected_plan else None
            reasoning_data = {
                "summary": getattr(res, "explanation", "") or getattr(res.reasoning_evidence, "summary", ""),
                "deduction_steps": getattr(res.reasoning_evidence, "deduction_steps", []),
                "candidate_rankings": getattr(res, "candidate_rankings", []),
                "important_assumptions": getattr(res, "important_assumptions", []),
            }
            risk_data = {
                "assessed_risk": getattr(res, "assessed_risk", res.risk),
                "risk_origin": getattr(res, "risk_origin", "unspecified"),
                "priority": res.priority,
                "uncertainty": getattr(res, "uncertainty", round(1.0 - res.calibrated_confidence, 4)),
            }
            evidence_data = {
                "citations": getattr(res, "citations", []),
                "retrieved_passages": getattr(res, "retrieved_passages", []),
                "supporting_evidence": getattr(res, "supporting_evidence", {}),
                "unsupported_claims": getattr(res, "unsupported_claims", []),
            }
            required_approval_data = {
                "requires_human_approval": (
                    getattr(res, "approval_requirements", {}).get("requires_approval", False)
                    or getattr(res, "execution_status", "") == "needs_approval"
                ),
                "required_role": getattr(res, "approval_requirements", {}).get("required_role") or "supervisor",
                "approval_rule": getattr(res, "approval_requirements", {}).get("approval_rule") or "standard_policy",
                "status": "pending" if getattr(res, "execution_status", "") == "needs_approval" else "none",
            }
            latency_data = {
                "latency_ms": getattr(res, "latency_ms", 0.0),
                "total_seconds": round(getattr(res, "latency_ms", 0.0) / 1000.0, 4),
            }
            return CognitiveEngineResponse(
                schema_version="v1.2.0",
                answer=getattr(res, "answer", ""),
                decision=res.decision,
                reasoning=reasoning_data,
                confidence=res.confidence,
                risk_assessment=risk_data,
                safety_status=getattr(res, "safety_status", "verified"),
                abstention_reason=res.abstention_reason,
                required_approval=required_approval_data,
                evidence=evidence_data,
                execution_status=getattr(res, "execution_status", "proposed"),
                cognitive_trace=getattr(res, "cognitive_trace", []),
                latency=latency_data,
                execution_id=res.execution_id,
                request_id=getattr(res, "request_id", ""),
                proposed_decision=getattr(res, "proposed_decision", res.decision),
                execution_authorized=getattr(res, "execution_authorized", False),
                calibrated_confidence=res.calibrated_confidence,
                priority=res.priority,
                risk=res.risk,
                operational_score=res.operational_score,
                predicted_outcomes=res.predicted_outcomes,
                selected_plan=plan_dict,
                reasoning_evidence=res.reasoning_evidence.model_dump(),
                world_state_summary=res.world_state_summary,
                abstain=res.abstain,
                execution_recommendation=res.execution_recommendation,
                latency_ms=res.latency_ms,
                citations=getattr(res, "citations", []),
                unsupported_claims=getattr(res, "unsupported_claims", []),
                retrieved_passages=getattr(res, "retrieved_passages", []),
                answer_type=getattr(res, "answer_type", "general"),
                explanation=getattr(res, "explanation", ""),
                risks_and_uncertainty=getattr(res, "risks_and_uncertainty", ""),
                candidate_scores=getattr(res, "candidate_scores", {}) or {},
                candidate_rankings=getattr(res, "candidate_rankings", []) or [],
            )

        # Hosted API execution
        payload = {
            "text": text,
            "structured_features": structured_features or {},
            "task": task,
            "candidates": candidates,
            "prohibited_actions": prohibited_actions,
            "goals": goals,
            "constraints": constraints,
            "session_id": session_id,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/v1/cognitive/process"
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    return CognitiveEngineResponse.model_validate(resp.json())
                elif resp.status_code == 401:
                    raise CDMAuthenticationError("API Key authentication rejected (HTTP 401).")
                elif resp.status_code == 429:
                    if attempt == self.max_retries - 1:
                        raise CDMRateLimitError("Rate limit exceeded (HTTP 429).")
                    time.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    raise CDMError(f"API Error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise CDMError(f"Connection failure to CDM Cognitive API at {url}: {e}") from e
                time.sleep(0.5 * (attempt + 1))

        raise CDMError("Exhausted maximum retry attempts.")
