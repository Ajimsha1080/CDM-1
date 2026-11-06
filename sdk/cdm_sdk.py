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
    execution_id: str
    decision: Any
    confidence: float
    calibrated_confidence: float
    priority: str
    risk: float
    operational_score: Optional[float] = None
    predicted_outcomes: Dict[str, Any] = Field(default_factory=dict)
    selected_plan: Optional[Dict[str, Any]] = None
    reasoning_evidence: Dict[str, Any] = Field(default_factory=dict)
    world_state_summary: Dict[str, Any] = Field(default_factory=dict)
    abstain: bool = False
    abstention_reason: Optional[str] = None
    execution_recommendation: str
    latency_ms: float


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
            return CognitiveEngineResponse(
                execution_id=res.execution_id,
                decision=res.decision,
                confidence=res.confidence,
                calibrated_confidence=res.calibrated_confidence,
                priority=res.priority,
                risk=res.risk,
                operational_score=res.operational_score,
                predicted_outcomes=res.predicted_outcomes,
                selected_plan=plan_dict,
                reasoning_evidence=res.reasoning_evidence.model_dump(),
                world_state_summary=res.world_state_summary,
                abstain=res.abstain,
                abstention_reason=res.abstention_reason,
                execution_recommendation=res.execution_recommendation,
                latency_ms=res.latency_ms,
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
