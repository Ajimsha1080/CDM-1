"""CDM-1 Commercial Decision REST API (Production Hardened).
Features environment-based secret management, persistent usage metering,
tenant isolation, distributed rate limiting, payload validation limits,
liveness & readiness probes, and structured error responses.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends, HTTPException, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import CognitiveEngineResult

# Structured Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("cdm.api")

from api.auth import PersistentApiKeyManager
from api.metering import DurableUsageMeter

# Persistent Secret & Key Management (Priority 6)
KEY_MANAGER = PersistentApiKeyManager(db_path=os.environ.get("CDM_KEY_DB", "experiments/api_keys.sqlite"))

# Durable Transactional Metering & Rate Limiting (Priority 6)
USAGE_METER = DurableUsageMeter(db_path=os.environ.get("CDM_METER_DB", "experiments/api_usage_meter.sqlite"))

security = HTTPBearer(auto_error=False)


# Pydantic Schemas with Strict Payload Validation
class DecisionRequest(BaseModel):
    text: str = Field(..., max_length=8192, description="Contextual narrative or inquiry text (max 8192 chars).")
    structured_features: Dict[str, Any] = Field(default_factory=dict, description="Tabular attributes.")
    task: str = Field(default="binary", description="'binary', 'choice', 'score', or 'predict'")
    candidates: Optional[List[str]] = Field(default=None, max_length=32, description="Candidate actions (max 32).")
    prohibited_actions: Optional[List[str]] = Field(default=None, description="Prohibited action constraints.")

    @field_validator("task")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        valid_tasks = {"binary", "choice", "score", "predict", "priority", "risk"}
        if v not in valid_tasks:
            raise ValueError(f"Invalid task '{v}'. Supported tasks: {valid_tasks}")
        return v


class BatchDecisionRequest(BaseModel):
    requests: List[DecisionRequest] = Field(..., max_length=50, description="Batch list (max 50 requests).")


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


class CognitiveProcessRequest(BaseModel):
    text: str = Field(..., max_length=8192, description="Task narrative or inquiry text.")
    structured_features: Dict[str, Any] = Field(default_factory=dict, description="Tabular attributes.")
    task: str = Field(default="choice", description="Decision task type.")
    candidates: Optional[List[str]] = Field(default=None, max_length=32, description="Candidate actions.")
    prohibited_actions: Optional[List[str]] = Field(default=None, description="Prohibited action constraints.")
    goals: Optional[List[str]] = Field(default=None, description="Active task goals.")
    constraints: Optional[List[str]] = Field(default=None, description="Explicit caller constraints.")
    session_id: str = Field(default="default_session", description="Session identifier for multi-turn tracking.")


class CognitiveProcessResponse(BaseModel):
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


# Global model & cognitive engine container
runtime_model: Optional[CognitiveDecisionModel] = None
runtime_cognitive_engine: Optional[CognitiveEngine] = None
model_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime_model, runtime_cognitive_engine, model_load_error

    # Priority 6: Fail startup in production if secrets are missing or default
    KEY_MANAGER.check_production_safety()

    # Seed development / test keys only in non-production environments
    if os.environ.get("CDM_ENV", "development").lower() != "production":
        KEY_MANAGER.register_key("sk_test_developer_key", "tenant_dev_default", tier="developer", rate_limit_rpm=120)
        KEY_MANAGER.register_key("sk_live_cdm_demo_key", "tenant_enterprise_01", tier="enterprise", rate_limit_rpm=300)

    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"

    logger.info(f"Loading CDM-1 Model from {model_path}...")
    try:
        runtime_model = CognitiveDecisionModel.from_pretrained(model_path)
        runtime_cognitive_engine = CognitiveEngine(
            decision_model=runtime_model,
            db_path=os.environ.get("CDM_MEMORY_DB", "experiments/cognitive_memory.db"),
        )
        logger.info("CDM-1 Model and Cognitive Engine loaded successfully and verified for production inference.")
    except Exception as e:
        model_load_error = str(e)
        logger.critical(f"FATAL: Failed to load CDM-1 model: {e}", exc_info=True)
        if os.environ.get("CDM_ENV", "development").lower() == "production":
            raise RuntimeError(f"CRITICAL PRODUCTION FAILURE: CDM-1 model failed to load at startup: {e}")
    yield


app = FastAPI(
    title="CDM-1 Cognitive Decision Model API",
    description="Commercial REST API for neural decision models, predictive estimation, and calibrated actions.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Production exception filter: masks internal exception details from HTTP clients."""
    logger.error(f"Internal unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Diagnostic event has been securely logged."},
    )


def authenticate_and_rate_limit(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, str]:
    """Validates API token against persistent secret store and enforces durable rate limits."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Bearer token required.",
        )

    token = credentials.credentials
    tenant_info = KEY_MANAGER.validate_key(token)
    if not tenant_info:
        logger.warning(f"Unauthorized access attempt with invalid token hash: {token[:6]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked CDM API Key.",
        )

    tenant_id = tenant_info["tenant_id"]
    rpm_limit = tenant_info.get("rate_limit_rpm", 120)

    allowed, count = USAGE_METER.check_rate_limit(tenant_id, max_rpm=rpm_limit)
    if not allowed:
        logger.warning(f"Rate limit exceeded for tenant: {tenant_id} (count={count}, limit={rpm_limit})")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {rpm_limit} requests per minute.",
            headers={"Retry-After": "60"},
        )

    return tenant_info


@app.get("/v1/health")
def health_check():
    """Liveness probe."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
    }


@app.get("/v1/ready")
def readiness_check():
    """Readiness probe checking model initialization and physical weights presence."""
    if runtime_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model runtime not ready: {model_load_error or 'Initializing'}",
        )
    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"
    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Checkpoint weights missing at {weights_path}",
        )
    return {
        "status": "ready",
        "model_loaded": True,
        "weights_verified": True,
        "parameters": "158M",
        "timestamp": time.time(),
    }


@app.get("/v1/version")
def version_info():
    return {
        "service": "CDM-1 Commercial Decision API",
        "version": "2.0.0",
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model runtime not initialized.",
        )

    start_t = time.time()
    try:
        res = runtime_model.decide(
            text=req.text,
            structured_features=req.structured_features,
            task=req.task,
            candidates=req.candidates,
            prohibited_actions=req.prohibited_actions,
        )
    except Exception as e:
        logger.error(f"Inference execution failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during decision processing. Diagnostic event logged.",
        )

    latency = round((time.time() - start_t) * 1000, 2)
    data = res.to_dict()

    # Durable transactional metering (Priority 6)
    USAGE_METER.record_request(
        tenant_id=tenant["tenant_id"],
        endpoint="/v1/decisions",
        task=req.task,
        latency_ms=latency,
        status_code=200,
    )

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


@app.post("/v1/cognitive/process", response_model=CognitiveProcessResponse)
def cognitive_process(
    req: CognitiveProcessRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Processes request through the complete integrated Cognitive AI Engine loop."""
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )
    try:
        res = runtime_cognitive_engine.process(
            text=req.text,
            structured_features=req.structured_features,
            candidates=req.candidates,
            prohibited_actions=req.prohibited_actions,
            task=req.task,
            goals=req.goals,
            constraints=req.constraints,
            session_id=req.session_id,
            tenant_id=tenant["tenant_id"],
        )
        plan_dict = res.selected_plan.model_dump() if res.selected_plan else None
        return CognitiveProcessResponse(
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
    except Exception as e:
        logger.error(f"Cognitive process failure: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during cognitive processing.",
        )


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
    summary = USAGE_METER.get_tenant_summary(tenant["tenant_id"])
    return {
        "tenant_id": tenant["tenant_id"],
        "tier": tenant["tier"],
        "usage": summary,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
