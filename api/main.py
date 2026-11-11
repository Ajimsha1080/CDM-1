"""CDM-1 Commercial Decision REST API (Production Hardened).
Features environment-based secret management, persistent usage metering,
tenant isolation, distributed rate limiting, payload validation limits,
liveness & readiness probes, and structured error responses.
"""

import os
import sys
import time
import json
import uuid
import logging
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends, HTTPException, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator
import uvicorn

from cdm import __version__
from cdm.inference.pipeline import CognitiveDecisionModel
from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import CognitiveEngineResult
from cdm.agent.loop import AgentControlLoop, AgentRunState, AgentStore

# Structured Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("cdm.api")

from api.auth import PersistentApiKeyManager
from api.metering import DurableUsageMeter

# Configurable Writable Data Paths (Item 6)
DATA_DIR = os.environ.get("CDM_DATA_DIR", "experiments")
KEY_DB_PATH = os.environ.get("CDM_KEY_DB", os.path.join(DATA_DIR, "api_keys.sqlite"))
METER_DB_PATH = os.environ.get("CDM_METER_DB", os.path.join(DATA_DIR, "api_usage_meter.sqlite"))
MEMORY_DB_PATH = os.environ.get("CDM_MEMORY_DB", os.path.join(DATA_DIR, "cognitive_memory.db"))
AGENT_DB_PATH = os.environ.get("CDM_AGENT_DB", os.path.join(DATA_DIR, "agent_runs.sqlite"))

KEY_MANAGER = PersistentApiKeyManager(db_path=KEY_DB_PATH)
USAGE_METER = DurableUsageMeter(db_path=METER_DB_PATH)

security = HTTPBearer(auto_error=False)


def _validate_dict_bounds(d: Any, max_entries: int = 100, max_key_len: int = 128, max_val_len: int = 1024) -> Dict[str, Any]:
    if not isinstance(d, dict):
        raise ValueError("Must be a dictionary")
    if len(d) > max_entries:
        raise ValueError(f"Dictionary exceeds maximum of {max_entries} entries (got {len(d)})")
    for k, v in d.items():
        if not isinstance(k, str) or len(k) > max_key_len:
            raise ValueError(f"Dictionary keys must be strings <= {max_key_len} characters")
        if isinstance(v, str) and len(v) > max_val_len:
            raise ValueError(f"Dictionary string value for key '{k}' exceeds {max_val_len} characters")
    return d


def _validate_list_bounds(lst: Any, max_items: int = 50, max_str_len: int = 1024) -> List[Any]:
    if not isinstance(lst, list):
        raise ValueError("Must be a list")
    if len(lst) > max_items:
        raise ValueError(f"List exceeds maximum of {max_items} items (got {len(lst)})")
    for item in lst:
        if isinstance(item, str) and len(item) > max_str_len:
            raise ValueError(f"List string item exceeds {max_str_len} characters")
    return lst


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

    @field_validator("structured_features")
    @classmethod
    def validate_structured_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_dict_bounds(v, max_entries=100, max_key_len=128, max_val_len=1024)

    @field_validator("candidates")
    @classmethod
    def validate_candidates(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=32, max_str_len=1024)
        return v

    @field_validator("prohibited_actions")
    @classmethod
    def validate_prohibited_actions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=50, max_str_len=256)
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
    score: Optional[float] = None
    neural_score: Optional[float] = None
    heuristic_score: Optional[float] = None
    used_neural_inference: bool = True
    priority: str
    risk: float
    abstain: bool
    selected_candidate: Optional[str] = None
    candidate_scores: Optional[Dict[str, Optional[float]]] = None
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
    session_id: str = Field(default="default_session", max_length=128, description="Session identifier for multi-turn tracking.")

    @field_validator("structured_features")
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        return _validate_dict_bounds(v, max_entries=100, max_key_len=128, max_val_len=1024)

    @field_validator("candidates")
    @classmethod
    def validate_cand(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=32, max_str_len=1024)
        return v

    @field_validator("prohibited_actions")
    @classmethod
    def validate_proh(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=50, max_str_len=256)
        return v

    @field_validator("goals")
    @classmethod
    def validate_goals(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=32, max_str_len=512)
        return v

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=32, max_str_len=512)
        return v


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
    embedding_mode: Optional[str] = None
    latency_ms: float


class CreateAgentRunRequest(BaseModel):
    goal: str = Field(..., max_length=2048, description="Primary goal or inquiry for autonomous agent")
    task_description: str = Field(default="", max_length=4096, description="Operational instructions and details")
    step_budget: int = Field(default=10, ge=1, le=50, description="Maximum execution step budget")
    cost_budget_usd: float = Field(default=1.0, ge=0.01, le=50.0, description="Cost budget cap in USD")
    tenant_policies: Optional[List[str]] = Field(default=None, max_length=20, description="Specific tenant policy rules")
    prohibited_actions: Optional[List[str]] = Field(default=None, max_length=50, description="Hard prohibited actions")
    verification_spec: Optional[Dict[str, Any]] = Field(default=None, description="Ground truth verification spec")
    auto_run: bool = Field(default=True, description="Whether to execute until completion or required pause")

    @field_validator("tenant_policies")
    @classmethod
    def validate_policies(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=20, max_str_len=512)
        return v

    @field_validator("prohibited_actions")
    @classmethod
    def validate_proh_actions(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            return _validate_list_bounds(v, max_items=50, max_str_len=256)
        return v

    @field_validator("verification_spec")
    @classmethod
    def validate_verif(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is not None:
            return _validate_dict_bounds(v, max_entries=50, max_key_len=128, max_val_len=1024)
        return v


class AgentRejectRequest(BaseModel):
    reason: str = Field(default="Rejected by human operator", max_length=1024)


class AgentRunResponse(BaseModel):
    run_id: str
    tenant_id: str
    session_id: str
    goal: str
    task_description: str
    status: str
    step_count: int
    step_budget: int
    cost_budget_usd: float
    accumulated_cost_usd: float
    accumulated_tokens: int
    tool_call_count: int
    history: List[Dict[str, Any]]
    pending_action: Optional[Dict[str, Any]] = None
    pending_reason: Optional[str] = None
    created_at: float
    updated_at: float


# Global model & cognitive engine container
runtime_model: Optional[CognitiveDecisionModel] = None
runtime_cognitive_engine: Optional[CognitiveEngine] = None
runtime_agent_loop: Optional[AgentControlLoop] = None
model_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global runtime_model, runtime_cognitive_engine, runtime_agent_loop, model_load_error

    # Fail startup in production if secrets are missing or default
    KEY_MANAGER.check_production_safety()

    is_prod = (os.environ.get("CDM_ENV", "production").lower() in ("production", "prod")) or (
        os.environ.get("CDM_PRODUCTION_MODE", "").strip().lower() in ("1", "true", "yes")
    )
    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"

    logger.info(f"Loading CDM-1 Model from {model_path} (production_mode={is_prod})...")
    try:
        runtime_model = CognitiveDecisionModel.from_pretrained(
            model_path,
            allow_fallback=not is_prod,
            production_mode=is_prod,
        )
        if is_prod:
            if getattr(runtime_model, "is_fallback", False) or runtime_model is None:
                raise RuntimeError("CRITICAL PRODUCTION FAILURE: Model loaded in fallback mode during production startup.")
            auth_check = verify_checkpoint_authenticity(model_path)
            if auth_check.get("status") != "authenticated":
                raise RuntimeError(f"CRITICAL PRODUCTION FAILURE: Checkpoint authenticity check failed: {auth_check}")
            smoke_res = runtime_model.decide(
                text="Production startup smoke test.",
                structured_features={"domain": 0, "urgency_level": 1},
                candidates=["act_pass", "act_verify"],
                task="choice",
            )
            if not getattr(smoke_res, "used_neural_inference", False):
                raise RuntimeError("CRITICAL PRODUCTION FAILURE: Production startup smoke test did not execute neural weights.")

        runtime_cognitive_engine = CognitiveEngine(
            decision_model=runtime_model,
            db_path=MEMORY_DB_PATH,
            production_mode=is_prod,
        )
        runtime_agent_loop = AgentControlLoop(
            cognitive_engine=runtime_cognitive_engine,
            store=AgentStore(AGENT_DB_PATH),
        )
        if getattr(runtime_model, "is_fallback", False):
            logger.warning("CDM-1 Model initialized in deterministic rule-based fallback mode (weights absent).")
        else:
            logger.info("CDM-1 Model and Cognitive Engine loaded successfully and verified for production inference.")
    except Exception as e:
        model_load_error = str(e)
        logger.critical(f"FATAL: Failed to load CDM-1 model: {e}", exc_info=True)
        if is_prod:
            raise RuntimeError(f"CRITICAL PRODUCTION FAILURE: CDM-1 model failed to load at startup: {e}")
    yield


app = FastAPI(
    title="CDM-1 Cognitive Decision Model API",
    description="Commercial REST API for neural decision models, predictive estimation, and calibrated actions.",
    version=__version__,
    lifespan=lifespan,
)

MAX_REQUEST_BODY_BYTES = 1048576  # 1 MB


@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    """Enforces global maximum request body size (1 MB)."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "PayloadTooLarge",
                        "detail": f"Request payload exceeds maximum allowed size of {MAX_REQUEST_BODY_BYTES} bytes (1 MB).",
                    },
                )
        except ValueError:
            pass
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Structured response for request model validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": jsonable_encoder(exc.errors()),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Structured response for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Production exception filter: masks internal exception details from HTTP clients."""
    logger.error(f"Internal unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An internal server error occurred. Diagnostic event has been securely logged.",
        },
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


from cdm.security.manifest_verification import (
    verify_checkpoint_authenticity,
    SecurityVerificationError,
)

VERIFIED_CHECKPOINT_HASH: Optional[str] = None


def _verify_checkpoint_integrity(model_path: str, weights_path: str) -> str:
    """Verifies that model.safetensors matches the trusted SHA-256 and signed release_manifest.json."""
    global VERIFIED_CHECKPOINT_HASH
    if VERIFIED_CHECKPOINT_HASH is not None:
        return VERIFIED_CHECKPOINT_HASH

    try:
        auth_info = verify_checkpoint_authenticity(model_path, verify_signature=True)
        VERIFIED_CHECKPOINT_HASH = auth_info["sha256"]
        return VERIFIED_CHECKPOINT_HASH
    except SecurityVerificationError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Checkpoint authenticity failure: {err}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Checkpoint verification error: {err}",
        )


@app.get("/v1/ready")
def readiness_check():
    """Readiness probe checking model initialization, weights presence or fallback status, and SHA-256 manifest integrity."""
    if runtime_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model runtime not ready: {model_load_error or 'Initializing'}",
        )

    if getattr(runtime_model, "is_fallback", False):
        if is_prod:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Production readiness check failed: rule-based fallback mode is active in production.",
            )
        return {
            "status": "ready",
            "model_loaded": True,
            "mode": "rule_based_fallback",
            "weights_verified": False,
            "manifest_signature_verified": False,
            "weights_sha256": None,
            "parameters": "0",
            "timestamp": time.time(),
        }

    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"
    weights_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(weights_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Checkpoint weights missing at {weights_path}",
        )

    verified_hash = _verify_checkpoint_integrity(model_path, weights_path)

    return {
        "status": "ready",
        "model_loaded": True,
        "mode": "neural",
        "weights_verified": True,
        "manifest_signature_verified": True,
        "weights_sha256": verified_hash,
        "parameters": "158M",
        "timestamp": time.time(),
    }


@app.get("/v1/version")
def version_info():
    return {
        "service": "CDM-1 Commercial Decision API",
        "version": __version__,
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
        decision_id=f"dec_{uuid.uuid4().hex}",
        model="CDM-1-general",
        task=req.task,
        decision=data["decision"],
        confidence=data["confidence"],
        calibrated=data.get("calibrated", True),
        score=data["score"],
        neural_score=data.get("neural_score"),
        heuristic_score=data.get("heuristic_score"),
        used_neural_inference=data.get("used_neural_inference", True),
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

        # Durable usage metering for cognitive process
        USAGE_METER.record_request(
            tenant_id=tenant["tenant_id"],
            endpoint="/v1/cognitive/process",
            task=req.task,
            latency_ms=res.latency_ms,
            status_code=200,
        )

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
            embedding_mode=getattr(res, "embedding_mode", "structured_fallback"),
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
    """Processes batch decision requests charging and enforcing rate limit per individual item."""
    batch_count = len(batch_req.requests)
    rpm_limit = tenant.get("rate_limit_rpm", 120)

    # Check capacity for full batch against durable meter
    allowed, current_count = USAGE_METER.check_rate_limit(tenant["tenant_id"], max_rpm=rpm_limit)
    if not allowed or (current_count + batch_count > rpm_limit):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Batch size ({batch_count}) exceeds available rate limit capacity ({rpm_limit - current_count} remaining).",
            headers={"Retry-After": "60"},
        )

    results = []
    for req in batch_req.requests:
        res = create_decision(req, tenant)
        results.append(res)
    return {"results": results, "count": len(results)}


# =====================================================================
# CDM-1 Agent Control Loop REST Endpoints (Phase 4)
# =====================================================================

@app.post("/v1/agent/runs", response_model=AgentRunResponse)
def create_agent_run(
    req: CreateAgentRunRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Initializes and optionally executes a CDM-1 autonomous agent control loop run."""
    if runtime_agent_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent control loop runtime not initialized.",
        )

    start_t = time.time()
    run = runtime_agent_loop.create_run(
        goal=req.goal,
        task_description=req.task_description,
        tenant_id=tenant["tenant_id"],
        step_budget=req.step_budget,
        cost_budget_usd=req.cost_budget_usd,
        tenant_policies=req.tenant_policies,
        prohibited_actions=req.prohibited_actions,
        verification_spec=req.verification_spec,
    )

    if req.auto_run:
        run = runtime_agent_loop.run_until_pause_or_finish(run.run_id)

    latency = round((time.time() - start_t) * 1000, 2)
    USAGE_METER.record_request(
        tenant_id=tenant["tenant_id"],
        endpoint="/v1/agent/runs",
        task="agent_run",
        latency_ms=latency,
        status_code=200,
    )
    return AgentRunResponse(**run.model_dump())


@app.get("/v1/agent/runs/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    run_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Retrieves current state, history, and approval requirements for an agent run."""
    if runtime_agent_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent control loop runtime not initialized.",
        )

    run = runtime_agent_loop.store.load_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run '{run_id}' not found.",
        )

    # Strict tenant isolation
    if run.tenant_id != tenant["tenant_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot access agent run belonging to another tenant.",
        )

    return AgentRunResponse(**run.model_dump())


@app.post("/v1/agent/runs/{run_id}/approve", response_model=AgentRunResponse)
def approve_agent_run(
    run_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Human approval gate: approves suspended high-risk action and resumes the agent loop."""
    if runtime_agent_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent control loop runtime not initialized.",
        )

    run = runtime_agent_loop.store.load_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run '{run_id}' not found.",
        )

    if run.tenant_id != tenant["tenant_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot operate on another tenant's agent run.",
        )

    try:
        updated_run = runtime_agent_loop.approve_pending(run_id)
        USAGE_METER.record_request(
            tenant_id=tenant["tenant_id"],
            endpoint="/v1/agent/runs/approve",
            task="agent_approve",
            latency_ms=10.0,
            status_code=200,
        )
        return AgentRunResponse(**updated_run.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/v1/agent/runs/{run_id}/reject", response_model=AgentRunResponse)
def reject_agent_run(
    run_id: str,
    req: AgentRejectRequest = AgentRejectRequest(),
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Human approval gate: rejects suspended action with an operator reason."""
    if runtime_agent_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent control loop runtime not initialized.",
        )

    run = runtime_agent_loop.store.load_run(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run '{run_id}' not found.",
        )

    if run.tenant_id != tenant["tenant_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: cannot operate on another tenant's agent run.",
        )

    try:
        updated_run = runtime_agent_loop.reject_pending(run_id, rejection_reason=req.reason)
        USAGE_METER.record_request(
            tenant_id=tenant["tenant_id"],
            endpoint="/v1/agent/runs/reject",
            task="agent_reject",
            latency_ms=10.0,
            status_code=200,
        )
        return AgentRunResponse(**updated_run.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


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
