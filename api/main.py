"""CDM-1 Commercial Decision REST API.
Built with FastAPI, providing authenticated endpoints for decisions, classification,
scoring, candidate choice, prediction, batch inference, usage metering, and health checks.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends, HTTPException, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from cdm.inference.pipeline import CognitiveDecisionModel

# API Authentication & Metering State
VALID_API_KEYS = {
    "sk_live_cdm_demo_key": {"tenant_id": "tenant_enterprise_01", "tier": "enterprise"},
    "sk_test_developer_key": {"tenant_id": "tenant_dev_default", "tier": "developer"},
}

USAGE_METER = {
    "total_requests": 0,
    "decisions_by_tenant": {},
    "last_request_time": 0.0,
}

RATE_LIMIT_BUCKET: Dict[str, List[float]] = {}
RATE_LIMIT_MAX_PER_MINUTE = 120

security = HTTPBearer(auto_error=False)


# Pydantic Schemas
class DecisionRequest(BaseModel):
    text: str = Field(..., description="Contextual narrative or inquiry text.")
    structured_features: Dict[str, Any] = Field(default_factory=dict, description="Tabular attributes.")
    task: str = Field(default="binary", description="'binary', 'choice', 'score', or 'predict'")
    candidates: Optional[List[str]] = Field(default=None, description="Candidate action list.")
    prohibited_actions: Optional[List[str]] = Field(default=None, description="Prohibited action constraints.")


class BatchDecisionRequest(BaseModel):
    requests: List[DecisionRequest]


class PredictedOutcomeSchema(BaseModel):
    binary_outcome_probability: float
    predicted_resolution_time: float


class DecisionResponse(BaseModel):
    decision_id: str
    model: str = "CDM-1-general"
    task: str
    decision: Any
    confidence: float
    calibrated: bool = True
    score: float
    priority: str
    risk: float
    abstain: bool
    selected_candidate: Optional[str] = None
    candidate_scores: Optional[Dict[str, float]] = None
    predicted_outcome: PredictedOutcomeSchema
    tenant_id: str
    latency_ms: float


# Global model container
runtime_model: Optional[CognitiveDecisionModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime_model
    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"
    print(f"Loading CDM-1 Model into memory from {model_path}...")
    runtime_model = CognitiveDecisionModel.from_pretrained(model_path)
    print("CDM-1 Model successfully loaded and ready for production inference.")
    yield


app = FastAPI(
    title="CDM-1 Cognitive Decision Model API",
    description="Commercial REST API for neural decision models, predictive estimation, and calibrated actions.",
    version="1.0.0",
    lifespan=lifespan,
)


def authenticate_and_rate_limit(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, str]:
    """Validates API token, enforces rate limits, and returns tenant metadata."""
    if not credentials:
        # Allow open access in local dev if no key passed, defaulting to dev tenant
        return {"tenant_id": "tenant_local_dev", "tier": "developer"}

    token = credentials.credentials
    if token not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired CDM API Key.",
        )

    tenant_info = VALID_API_KEYS[token]
    tenant_id = tenant_info["tenant_id"]

    # Rate limiting: 120 req/minute
    now = time.time()
    timestamps = RATE_LIMIT_BUCKET.setdefault(tenant_id, [])
    # purge stamps older than 60 seconds
    timestamps = [t for t in timestamps if now - t < 60.0]
    RATE_LIMIT_BUCKET[tenant_id] = timestamps

    if len(timestamps) >= RATE_LIMIT_MAX_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 120 requests per minute.",
        )

    RATE_LIMIT_BUCKET[tenant_id].append(now)

    # Metering
    USAGE_METER["total_requests"] += 1
    USAGE_METER["decisions_by_tenant"][tenant_id] = (
        USAGE_METER["decisions_by_tenant"].get(tenant_id, 0) + 1
    )
    USAGE_METER["last_request_time"] = now

    return tenant_info


@app.get("/v1/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": runtime_model is not None,
        "timestamp": time.time(),
    }


@app.get("/v1/version")
def version_info():
    return {
        "service": "CDM-1 Commercial Decision API",
        "version": "1.0.0",
        "architecture": "Hybrid Neural Cognitive Decision Model",
        "framework": "PyTorch + Transformers",
    }


@app.get("/v1/models")
def list_models(tenant: Dict[str, str] = Depends(authenticate_and_rate_limit)):
    return {
        "models": [
            {
                "id": "cdm-1-general",
                "name": "CDM-1 General Operational Decision Model",
                "parameters": "158M",
                "tasks": ["binary", "choice", "score", "priority", "risk", "predict"],
                "status": "active",
            }
        ]
    }


@app.post("/v1/decisions", response_model=DecisionResponse)
def create_decision(
    req: DecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Primary operational decision endpoint supporting all task modalities."""
    if runtime_model is None:
        raise HTTPException(status_code=503, detail="Model runtime not initialized.")

    start_t = time.time()
    res = runtime_model.decide(
        text=req.text,
        structured_features=req.structured_features,
        task=req.task,
        candidates=req.candidates,
        prohibited_actions=req.prohibited_actions,
    )
    latency = round((time.time() - start_t) * 1000, 2)

    data = res.to_dict()
    return DecisionResponse(
        decision_id=f"dec_{int(time.time()*1000)}",
        model="CDM-1-general",
        task=req.task,
        decision=data["decision"],
        confidence=data["confidence"],
        calibrated=data.get("calibrated", True),
        score=data["score"],
        priority=data["priority"],
        risk=data["risk"],
        abstain=data["abstain"],
        selected_candidate=data.get("selected_candidate"),
        candidate_scores=data.get("candidate_scores"),
        predicted_outcome=data["predicted_outcome"],
        tenant_id=tenant["tenant_id"],
        latency_ms=latency,
    )


@app.post("/v1/classify")
def classify(
    req: DecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    req.task = "binary"
    return create_decision(req, tenant)


@app.post("/v1/score")
def score(
    req: DecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    req.task = "score"
    return create_decision(req, tenant)


@app.post("/v1/choice")
def choice(
    req: DecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    req.task = "choice"
    return create_decision(req, tenant)


@app.post("/v1/predict")
def predict_outcome(
    req: DecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    req.task = "predict"
    return create_decision(req, tenant)


@app.post("/v1/batch")
def batch_decisions(
    batch_req: BatchDecisionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    results = []
    for req in batch_req.requests:
        res = create_decision(req, tenant)
        results.append(res)
    return {"results": results, "count": len(results)}


@app.get("/v1/metering")
def get_metering(tenant: Dict[str, str] = Depends(authenticate_and_rate_limit)):
    return {
        "tenant_id": tenant["tenant_id"],
        "tier": tenant["tier"],
        "usage": USAGE_METER,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
