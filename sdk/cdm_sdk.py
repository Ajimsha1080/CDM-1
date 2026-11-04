"""CDM-1 Python SDK (cdm_sdk).
Provides a unified client for either Local or Hosted Commercial API inference
with typed responses, retries, and error handling.
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


class CDMError(Exception):
    """Base exception for CDM SDK."""
    pass


class CDM:
    """Client for CDM-1 decision models (supports Local execution or Hosted REST API)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        local_model_path: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.local_model_path = local_model_path
        self.timeout = timeout
        self.max_retries = max_retries

        self._local_model = None
        if local_model_path is not None:
            from cdm.inference.pipeline import CognitiveDecisionModel
            self._local_model = CognitiveDecisionModel.from_pretrained(local_model_path)

    def decide(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        task: str = "binary",
        candidates: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
    ) -> CDMResponse:
        """Submit a decision request to either local engine or remote API."""
        # 1. Local execution
        if self._local_model is not None:
            res = self._local_model.decide(
                text=text,
                structured_features=structured_features,
                task=task,
                candidates=candidates,
                prohibited_actions=prohibited_actions,
            )
            return CDMResponse.model_validate(res.to_dict())

        # 2. Remote API execution
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
                elif resp.status_code == 429:
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
        """Process multiple decision requests sequentially or via batch endpoint."""
        results = []
        for item in items:
            res = self.decide(**item)
            results.append(res)
        return results
