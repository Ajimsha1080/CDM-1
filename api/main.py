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
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, Depends, HTTPException, Security, Request, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
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

from cdm.context.session_manager import SessionManager, ConversationSession
from cdm.retrieval.document_extractor import TenantDocumentManager, DocumentMetadata
from cdm.security.approval import ActionApproval, generate_action_approval, verify_action_approval

SESSION_MANAGER = SessionManager(db_path=MEMORY_DB_PATH)
DOCUMENT_MANAGER = TenantDocumentManager(db_path=MEMORY_DB_PATH)

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


def sanitize_trace_payload(data: Any) -> Any:
    """Recursively masks secrets, tokens, API keys, passwords, and sensitive credentials in traces."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            k_lower = str(k).lower()
            if isinstance(v, (dict, list)):
                sanitized[k] = sanitize_trace_payload(v)
            elif any(s in k_lower for s in ("token", "secret", "password", "key", "authorization", "credential")):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = v
        return sanitized
    elif isinstance(data, list):
        return [sanitize_trace_payload(item) for item in data]
    return data


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
    # Versioned Response Contract (v1.2.0)
    schema_version: str = "v1.2.0"

    # Core Natural-Language Cognitive Response Fields (Item 3 Specification)
    answer: str = Field(..., description="Direct natural-language response grounded in verified cognitive results.")
    decision: Optional[Any] = Field(None, description="The selected candidate decision or null if abstained/informational.")
    reasoning: Dict[str, Any] = Field(default_factory=dict, description="Structured reasoning summary, deduction steps, rankings, and assumptions.")
    confidence: float = Field(..., description="Calibrated confidence score (0.0 to 1.0).")
    risk_assessment: Dict[str, Any] = Field(default_factory=dict, description="Assessed risk score, risk origin, priority, uncertainty, and identified risks.")
    safety_status: str = Field(..., description="Safety verification status: 'verified', 'needs_approval', 'abstained', 'policy_violation'.")
    abstention_reason: Optional[str] = Field(None, description="Exact threshold or policy rule causing abstention, or null if not abstained.")
    required_approval: Dict[str, Any] = Field(default_factory=dict, description="Human approval requirements: role, rule, and status.")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Evidence grounding: citations, retrieved passages, observed facts, and unsupported claims.")
    execution_status: str = Field(..., description="Execution status: 'authorized', 'proposed', 'needs_approval', 'abstained', 'rejected'.")
    cognitive_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Sanitized stage-level execution trace.")
    latency: Dict[str, Any] = Field(default_factory=dict, description="Latency metrics: total ms, total seconds, and stage durations.")

    # P1 Contract & Provenance Fields
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    proposed_decision: Optional[Any] = None
    execution_authorized: bool = False
    abstention_status: str = "none"
    uncertainty: float = 0.0
    assessed_risk: float = 0.0
    policy_check: Dict[str, Any] = Field(default_factory=dict)
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    candidate_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    approval_requirements: Dict[str, Any] = Field(default_factory=dict)
    verification: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    is_calibrated: bool = False
    confidence_origin: str = "unspecified"
    risk_origin: str = "unspecified"

    # Cognitive Answer Synthesis Fields
    answer_type: str = "general"
    explanation: str = ""
    risks_and_uncertainty: str = ""
    limitations: List[str] = Field(default_factory=list)
    important_assumptions: List[str] = Field(default_factory=list)
    recommended_next_steps: List[str] = Field(default_factory=list)
    retrieved_external_facts: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    retrieved_passages: List[Dict[str, Any]] = Field(default_factory=list)
    intent_task: str = "question"
    needs_clarification: bool = False
    clarification_prompt: Optional[str] = None
    validation_result: Optional[Dict[str, Any]] = None

    # Backward Compatibility Fields
    execution_id: str
    calibrated_confidence: float
    priority: str
    risk: float
    operational_score: Optional[float] = None
    candidate_scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    used_neural_inference: bool = False
    predicted_outcomes: Dict[str, Any] = Field(default_factory=dict)
    selected_plan: Optional[Dict[str, Any]] = None
    reasoning_evidence: Dict[str, Any] = Field(default_factory=dict)
    world_state_summary: Dict[str, Any] = Field(default_factory=dict)
    abstain: bool = False
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
    description="""Commercial REST API for neural decision models, predictive estimation, and calibrated actions.

### 🔑 Quick Authorization for Swagger UI
1. Scroll down to **Development** and run **`GET /v1/dev/default-key`** (or click 'Try it out' -> 'Execute').
2. Copy the returned `api_key` (e.g. `cdm_dev_...`).
3. Click the green **Authorize 🔓** button at the top-right of this page.
4. Paste the key into the **Value** box and click **Authorize**.
5. All endpoints will now execute with full authorization!
""",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_REQUEST_BODY_BYTES = 1048576  # 1 MB


@app.middleware("http")
async def limit_request_body_size(request: Request, call_next):
    """Enforces global maximum request body size (1 MB) for standard and streamed requests."""
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

    # Enforce streaming byte limits for chunked requests
    bytes_received = 0
    original_receive = request._receive

    async def limited_receive():
        nonlocal bytes_received
        message = await original_receive()
        if message.get("type") == "http.request":
            body = message.get("body", b"")
            bytes_received += len(body)
            if bytes_received > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Streamed payload exceeds maximum allowed size of {MAX_REQUEST_BODY_BYTES} bytes (1 MB).",
                )
        return message

    request._receive = limited_receive
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

    is_prod = (os.environ.get("CDM_ENV", "production").lower() in ("production", "prod")) or (
        os.environ.get("CDM_PRODUCTION_MODE", "").lower() in ("1", "true")
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
    model_entries = [
        {
            "id": "cdm-1",
            "name": "CDM-1 General Operational Decision Model",
            "object": "model",
            "created": 1794944000,
            "owned_by": "cdm-organization",
            "parameters": "158M",
            "tasks": ["binary", "choice", "score", "priority", "risk", "predict", "chat"],
            "status": "active",
        },
        {
            "id": "cdm-1-general",
            "name": "CDM-1 General Operational Decision Model",
            "object": "model",
            "created": 1794944000,
            "owned_by": "cdm-organization",
            "parameters": "158M",
            "tasks": ["binary", "choice", "score", "priority", "risk", "predict", "chat"],
            "status": "active",
        },
    ]
    return {
        "object": "list",
        "data": model_entries,
        "models": model_entries,
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
        session_obj = SESSION_MANAGER.get_or_create_session(
            tenant_id=tenant["tenant_id"],
            session_id=req.session_id,
        )
        effective_session_id = session_obj.session_id

        res = runtime_cognitive_engine.process(
            text=req.text,
            structured_features=req.structured_features,
            candidates=req.candidates,
            prohibited_actions=req.prohibited_actions,
            task=req.task,
            goals=req.goals,
            constraints=req.constraints,
            session_id=effective_session_id,
            tenant_id=tenant["tenant_id"],
        )
        SESSION_MANAGER.update_session_activity(
            session_id=effective_session_id,
            tenant_id=tenant["tenant_id"],
            turn_increment=1,
            goals=req.goals,
            constraints=req.constraints,
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

        # 1. Structured Reasoning
        reasoning_data = {
            "summary": getattr(res, "explanation", "") or "",
            "deduction_steps": [s.model_dump() for s in getattr(res.reasoning_evidence, "steps", [])] if hasattr(res, "reasoning_evidence") and hasattr(res.reasoning_evidence, "steps") else [],
            "candidate_rankings": getattr(res, "candidate_rankings", []) or [],
            "assumptions": getattr(res, "important_assumptions", []) or [],
        }

        # 2. Structured Risk Assessment
        risk_data = {
            "assessed_risk": getattr(res, "assessed_risk", res.risk),
            "risk_origin": getattr(res, "risk_origin", "unspecified"),
            "priority": getattr(res, "priority", "MEDIUM"),
            "uncertainty": getattr(res, "uncertainty", round(1.0 - res.calibrated_confidence, 4)),
            "identified_risks": getattr(res, "risks_and_uncertainty", "") or "",
        }

        # 3. Structured Required Approval
        appr_req = getattr(res, "approval_requirements", {}) or {}
        req_approval = bool(appr_req.get("requires_human_approval") or appr_req.get("required"))
        required_approval_data = {
            "requires_human_approval": req_approval,
            "required_role": appr_req.get("required_role") or appr_req.get("approver_role"),
            "approval_rule": appr_req.get("approval_rule") or appr_req.get("rule"),
            "status": "pending" if req_approval else "none",
        }

        # 4. Structured Evidence
        evidence_data = {
            "supporting_evidence": getattr(res, "supporting_evidence", {}) or {},
            "citations": getattr(res, "citations", []) or [],
            "retrieved_passages": getattr(res, "retrieved_passages", []) or [],
            "unsupported_claims": getattr(res, "unsupported_claims", []) or [],
            "observed_facts": getattr(res, "world_state_summary", {}).get("observed_facts", {}) if isinstance(getattr(res, "world_state_summary", None), dict) else {},
        }

        # 5. Structured Latency
        latency_data = {
            "latency_ms": getattr(res, "latency_ms", 0.0),
            "total_seconds": round(getattr(res, "latency_ms", 0.0) / 1000.0, 4),
            "stage_latencies_ms": {
                stage.get("stage_name", f"stage_{i}"): stage.get("duration_ms", 0.0)
                for i, stage in enumerate(getattr(res, "cognitive_trace", []))
                if isinstance(stage, dict) and "stage_name" in stage
            },
        }

        return CognitiveProcessResponse(
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
            cognitive_trace=sanitize_trace_payload(getattr(res, "cognitive_trace", [])),
            latency=latency_data,
            request_id=getattr(res, "request_id", f"req_{uuid.uuid4().hex[:8]}"),
            proposed_decision=getattr(res, "proposed_decision", res.decision),
            execution_authorized=getattr(res, "execution_authorized", False),
            abstention_status=getattr(res, "abstention_status", "none"),
            uncertainty=getattr(res, "uncertainty", round(1.0 - res.calibrated_confidence, 4)),
            assessed_risk=getattr(res, "assessed_risk", res.risk),
            policy_check=getattr(res, "policy_check", {}),
            supporting_evidence=getattr(res, "supporting_evidence", {}),
            candidate_rankings=getattr(res, "candidate_rankings", []),
            plan=plan_dict,
            approval_requirements=getattr(res, "approval_requirements", {}),
            verification=getattr(res, "verification", {}),
            errors=getattr(res, "errors", []),
            is_calibrated=getattr(res, "is_calibrated", False),
            confidence_origin=getattr(res, "confidence_origin", "unspecified"),
            risk_origin=getattr(res, "risk_origin", "unspecified"),
            answer_type=getattr(res, "answer_type", "general"),
            explanation=getattr(res, "explanation", ""),
            risks_and_uncertainty=getattr(res, "risks_and_uncertainty", ""),
            limitations=getattr(res, "limitations", []),
            important_assumptions=getattr(res, "important_assumptions", []),
            recommended_next_steps=getattr(res, "recommended_next_steps", []),
            retrieved_external_facts=getattr(res, "retrieved_external_facts", []),
            citations=getattr(res, "citations", []),
            unsupported_claims=getattr(res, "unsupported_claims", []),
            retrieved_passages=getattr(res, "retrieved_passages", []),
            intent_task=getattr(res, "intent_task", "question"),
            needs_clarification=getattr(res, "needs_clarification", False),
            clarification_prompt=getattr(res, "clarification_prompt", None),
            validation_result=getattr(res, "validation_result", None),
            execution_id=res.execution_id,
            calibrated_confidence=res.calibrated_confidence,
            priority=res.priority,
            risk=res.risk,
            operational_score=res.operational_score,
            candidate_scores=getattr(res, "candidate_scores", {}) or {},
            alternatives=getattr(res, "alternatives", []) or [],
            used_neural_inference=getattr(res, "used_neural_inference", False),
            predicted_outcomes=res.predicted_outcomes,
            selected_plan=plan_dict,
            reasoning_evidence=res.reasoning_evidence.model_dump(),
            world_state_summary=res.world_state_summary,
            abstain=res.abstain,
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


@app.post("/v1/cognitive/process/stream")
def cognitive_process_stream(
    req: CognitiveProcessRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Streams real-time cognitive stage progress, durations, and synthesized answers as SSE."""
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )

    def event_generator():
        yield f"data: {json.dumps({'event': 'stage_start', 'stage': 'Perception', 'timestamp': time.time()})}\n\n"
        session_obj = SESSION_MANAGER.get_or_create_session(
            tenant_id=tenant["tenant_id"],
            session_id=req.session_id,
        )
        effective_session_id = session_obj.session_id

        res = runtime_cognitive_engine.process(
            text=req.text,
            structured_features=req.structured_features,
            candidates=req.candidates,
            prohibited_actions=req.prohibited_actions,
            task=req.task,
            goals=req.goals,
            constraints=req.constraints,
            session_id=effective_session_id,
            tenant_id=tenant["tenant_id"],
        )
        SESSION_MANAGER.update_session_activity(
            session_id=effective_session_id,
            tenant_id=tenant["tenant_id"],
            turn_increment=1,
            goals=req.goals,
            constraints=req.constraints,
        )
        for stage in res.cognitive_trace:
            yield f"data: {json.dumps({'event': 'stage_complete', 'stage': stage['stage_name'], 'duration_ms': stage.get('duration_ms', 0), 'status': stage.get('execution_status', 'success')})}\n\n"

        final_payload = {
            "event": "final_result",
            "session_id": effective_session_id,
            "request_id": getattr(res, "request_id", ""),
            "answer": getattr(res, "answer", ""),
            "answer_type": getattr(res, "answer_type", "general"),
            "proposed_decision": getattr(res, "proposed_decision", res.decision),
            "execution_status": getattr(res, "execution_status", "proposed"),
            "execution_authorized": getattr(res, "execution_authorized", False),
            "safety_status": getattr(res, "safety_status", "verified"),
            "confidence": res.confidence,
            "assessed_risk": getattr(res, "assessed_risk", res.risk),
            "citations": getattr(res, "citations", []),
            "retrieved_passages": getattr(res, "retrieved_passages", []),
        }
        yield f"data: {json.dumps(final_payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =====================================================================
# Universal LLM Compatibility Layer (OpenAI, Claude, Gemini Compatible)
# =====================================================================

class OpenAIChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: Any = Field(..., description="Message content string or structured block")

class OpenAIChatCompletionRequest(BaseModel):
    model: str = Field(default="cdm-1", description="Model identifier (e.g. 'cdm-1', 'cdm-1-general')")
    messages: List[OpenAIChatMessage] = Field(..., description="Conversation history")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens")
    stream: Optional[bool] = Field(default=False, description="Whether to stream response tokens")
    session_id: Optional[str] = Field(default=None, description="Optional persistent session identifier")

@app.post("/v1/chat/completions")
async def chat_completions(
    req: OpenAIChatCompletionRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """OpenAI / Gemini compatible Chat Completions API.
    Enables drop-in replacement for OpenAI SDK, LangChain, LlamaIndex, Cursor, and any standard LLM client.
    """
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )

    # 1. Extract user message and conversation context
    user_prompt = ""
    system_prompt = ""
    for m in reversed(req.messages):
        if m.role == "user":
            if isinstance(m.content, str):
                user_prompt = m.content
            elif isinstance(m.content, list):
                user_prompt = " ".join(item.get("text", "") for item in m.content if isinstance(item, dict))
            break
    if not user_prompt:
        user_prompt = "Hello"

    for m in req.messages[:-1]:
        if m.role == "system":
            system_prompt += (m.content if isinstance(m.content, str) else "") + "\n"

    full_text = f"{system_prompt.strip()}\n\n{user_prompt}".strip() if system_prompt.strip() else user_prompt

    # 2. Execute authoritative cognitive processing
    session_obj = SESSION_MANAGER.get_or_create_session(
        tenant_id=tenant["tenant_id"],
        session_id=req.session_id,
    )
    res = runtime_cognitive_engine.process(
        text=full_text,
        session_id=session_obj.session_id,
        tenant_id=tenant["tenant_id"],
    )
    answer_text = res.answer or res.explanation or "Decision processed successfully."
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    # 3. Handle Streaming (Server-Sent Events)
    if req.stream:
        def stream_generator():
            words = answer_text.split(" ")
            for i, word in enumerate(words):
                chunk_str = word + (" " if i < len(words) - 1 else "")
                chunk_payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk_str},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_payload)}\n\n"
                time.sleep(0.01)
            stop_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(stop_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # 4. Standard Non-Streaming JSON Response
    prompt_tokens = len(user_prompt.split()) * 2
    comp_tokens = len(answer_text.split()) * 2
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens,
        },
        "cognitive_metadata": {
            "decision": getattr(res, "selected_action", None),
            "confidence": getattr(res, "calibrated_confidence", None),
            "risk": getattr(res, "assessed_risk", None),
            "citations": getattr(res, "citations", []),
        },
    }


class AnthropicMessageItem(BaseModel):
    role: str
    content: Any

class AnthropicMessagesRequest(BaseModel):
    model: str = Field(default="cdm-1", description="Model identifier")
    messages: List[AnthropicMessageItem] = Field(..., description="Conversation messages")
    max_tokens: Optional[int] = Field(default=1024, description="Maximum tokens")
    system: Optional[str] = Field(default=None, description="System prompt instructions")
    stream: Optional[bool] = Field(default=False, description="Streaming flag")

@app.post("/v1/messages")
async def anthropic_messages(
    req: AnthropicMessagesRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Anthropic Claude compatible Messages API.
    Enables drop-in replacement for Anthropic SDK clients and Claude tools.
    """
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )

    user_prompt = ""
    for m in reversed(req.messages):
        if m.role == "user":
            if isinstance(m.content, str):
                user_prompt = m.content
            elif isinstance(m.content, list):
                user_prompt = " ".join(item.get("text", "") for item in m.content if isinstance(item, dict))
            break
    if not user_prompt:
        user_prompt = "Hello"

    full_text = f"{req.system}\n\n{user_prompt}".strip() if req.system else user_prompt

    res = runtime_cognitive_engine.process(
        text=full_text,
        tenant_id=tenant["tenant_id"],
    )
    answer_text = res.answer or res.explanation or "Decision processed successfully."
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"
    in_tok = len(user_prompt.split()) * 2
    out_tok = len(answer_text.split()) * 2

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": req.model,
        "content": [
            {
                "type": "text",
                "text": answer_text,
            }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        },
        "cognitive_metadata": {
            "confidence": getattr(res, "calibrated_confidence", None),
            "risk": getattr(res, "assessed_risk", None),
            "citations": getattr(res, "citations", []),
        },
    }


class UserFeedbackRequest(BaseModel):
    session_id: str
    rating: int = Field(..., ge=1, le=5, description="User satisfaction rating (1-5 stars)")
    decision_id: Optional[str] = None
    user_comment: Optional[str] = None
    category: str = "general"


@app.post("/v1/cognitive/feedback")
def submit_user_feedback(
    feedback: UserFeedbackRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Submits user feedback and stores it in segregated storage permanently excluded from retraining."""
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )
    opinion = runtime_cognitive_engine.feedback_collector.record_user_opinion(
        tenant_id=tenant["tenant_id"],
        session_id=feedback.session_id,
        rating=feedback.rating,
        user_comment=feedback.user_comment,
        decision_id=feedback.decision_id,
        category=feedback.category,
    )
    return {
        "status": "recorded",
        "opinion_id": opinion.opinion_id,
        "eligible_for_training": False,
        "message": "User opinion recorded in segregated store. Excluded from automated model adaptation.",
    }


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


class AgentApproveRequest(BaseModel):
    operator_id: str = Field(default="operator_admin", max_length=128, description="Operator user ID")
    approval_token: Optional[str] = Field(None, description="Cryptographically verifiable signed action-bound approval token")


class ApprovalTokenResponse(BaseModel):
    run_id: str
    tenant_id: str
    action_tool: str
    operator_id: str
    approval_token: str
    expires_at: float


@app.post("/v1/agent/runs/{run_id}/approval-token", response_model=ApprovalTokenResponse)
def create_approval_token(
    run_id: str,
    operator_id: str = "operator_admin",
    ttl_seconds: float = 300.0,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Generates a cryptographically signed expiring action-bound approval token for an operator."""
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
    if run.status != "waiting_for_approval" or not run.pending_action:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run '{run_id}' is not suspended waiting for approval (current status: {run.status}).",
        )
    token_str = runtime_agent_loop.generate_approval_token(
        run_id=run_id, operator_id=operator_id, ttl_seconds=ttl_seconds
    )
    from cdm.security.approval import ActionApproval
    token_obj = ActionApproval.deserialize(token_str)
    return ApprovalTokenResponse(
        run_id=run_id,
        tenant_id=run.tenant_id,
        action_tool=run.pending_action.get("tool", ""),
        operator_id=operator_id,
        approval_token=token_str,
        expires_at=token_obj.expires_at,
    )


@app.post("/v1/agent/runs/{run_id}/approve", response_model=AgentRunResponse)
def approve_agent_run(
    run_id: str,
    req: AgentApproveRequest = AgentApproveRequest(),
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Human approval gate: approves suspended high-risk action using cryptographically verifiable action-bound validation."""
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
        updated_run = runtime_agent_loop.approve_pending(
            run_id=run_id,
            operator_id=req.operator_id,
            approval_token=req.approval_token,
        )
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


@app.get("/v1/agent/runs", response_model=List[AgentRunResponse])
def list_agent_runs(
    limit: int = 50,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Lists recent agent runs for the authenticated tenant."""
    if runtime_agent_loop is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent control loop runtime not initialized.",
        )
    runs = runtime_agent_loop.store.list_runs(tenant_id=tenant["tenant_id"], limit=limit)
    return [AgentRunResponse(**r.model_dump()) for r in runs]


@app.get("/v1/memory/records")
def get_memory_records(
    limit: int = 50,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Inspect episodic, semantic (rules), and procedural memory records."""
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )
    storage = runtime_cognitive_engine.storage
    episodic = storage.list_episodic(tenant_id=tenant["tenant_id"], limit=limit)
    semantic = storage.list_all_rules(tenant_id=tenant["tenant_id"], limit=limit)
    procedural = storage.list_all_procedures(tenant_id=tenant["tenant_id"], limit=limit)
    return {
        "tenant_id": tenant["tenant_id"],
        "episodic": [e.model_dump() for e in episodic],
        "semantic_rules": semantic,
        "procedural": procedural,
    }


@app.get("/v1/world-state")
def get_world_state(
    session_id: str = "default_session",
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Inspects real-time world-state telemetry without fabrication."""
    if runtime_cognitive_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cognitive Engine runtime not initialized.",
        )
    world_model = runtime_cognitive_engine._get_or_create_world_model(
        tenant_id=tenant["tenant_id"], session_id=session_id
    )
    state = world_model.get_state(session_id=session_id)
    return {
        "tenant_id": tenant["tenant_id"],
        "session_id": session_id,
        "world_state": state.model_dump(),
    }


# ==========================================
# Session Management & Lifecycle (Section 3)
# ==========================================

class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(None, max_length=128)
    user_id: str = Field(default="default_user", max_length=128)
    title: str = Field(default="New Conversation", max_length=256)
    goals: Optional[List[str]] = Field(default_factory=list)
    constraints: Optional[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    title: str
    created_at: float
    updated_at: float
    turn_count: int
    active_goals: List[str]
    active_constraints: List[str]
    metadata: Dict[str, Any]


class SessionDetailResponse(SessionResponse):
    dialogue_history: List[Dict[str, Any]] = Field(default_factory=list)
    detected_contradictions: List[str] = Field(default_factory=list)
    world_state: Optional[Dict[str, Any]] = None


@app.post("/v1/sessions", response_model=SessionResponse)
def create_session(
    req: CreateSessionRequest = CreateSessionRequest(),
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Creates a unique conversation session enforcing tenant and user isolation."""
    session = SESSION_MANAGER.create_session(
        tenant_id=tenant["tenant_id"],
        user_id=req.user_id,
        session_id=req.session_id,
        title=req.title,
        goals=req.goals,
        constraints=req.constraints,
        metadata=req.metadata,
    )
    return SessionResponse(**session.model_dump())


@app.get("/v1/sessions", response_model=List[SessionResponse])
def list_sessions(
    limit: int = 50,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Lists active conversation sessions for the authenticated tenant."""
    sessions = SESSION_MANAGER.list_sessions(tenant_id=tenant["tenant_id"], limit=limit)
    return [SessionResponse(**s.model_dump()) for s in sessions]


@app.get("/v1/sessions/{session_id}", response_model=SessionDetailResponse)
def get_session_details(
    session_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Retrieves session details, conversation history, and isolated world state."""
    session = SESSION_MANAGER.get_session(session_id=session_id, tenant_id=tenant["tenant_id"])
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    dialogue_history = []
    contradictions = []
    world_state_data = None

    if runtime_cognitive_engine is not None:
        history_frames = runtime_cognitive_engine.context_engine.get_history(
            session_id=session_id, tenant_id=tenant["tenant_id"]
        )
        if history_frames:
            latest_frame = history_frames[-1]
            dialogue_history = getattr(latest_frame, "dialogue_history", [])
            contradictions = getattr(latest_frame, "detected_contradictions", [])

        world_model = runtime_cognitive_engine._get_or_create_world_model(
            tenant_id=tenant["tenant_id"], session_id=session_id
        )
        state = world_model.get_state(session_id=session_id)
        world_state_data = state.model_dump()

    return SessionDetailResponse(
        **session.model_dump(),
        dialogue_history=dialogue_history,
        detected_contradictions=contradictions,
        world_state=world_state_data,
    )


@app.delete("/v1/sessions/{session_id}")
def delete_session(
    session_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Deletes an isolated session and cleans up context history."""
    deleted = SESSION_MANAGER.delete_session(session_id=session_id, tenant_id=tenant["tenant_id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    if runtime_cognitive_engine is not None:
        runtime_cognitive_engine.context_engine.clear_session(session_id=session_id, tenant_id=tenant["tenant_id"])
    return {"deleted": True, "session_id": session_id}


# ==========================================
# Document Intelligence & Lifecycle (Section 5)
# ==========================================

class DocumentUploadResponse(BaseModel):
    doc_id: str
    tenant_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    title: str
    source_repository: str
    version: str
    uploaded_at: float
    sha256_hash: str
    chunk_count: int
    total_characters: int


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    top_k: int = Field(default=3, ge=1, le=20)
    min_score: float = Field(default=0.05, ge=0.0, le=1.0)


class DocumentSearchHit(BaseModel):
    passage_id: str
    doc_id: str
    title: str
    content: str
    source_repository: str
    version: str
    relevance_score: float
    retrieval_timestamp: float


@app.post("/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source_repository: str = Form("Enterprise Knowledge Repository"),
    version: str = Form("1.0.0"),
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Uploads and extracts text from PDF, DOCX, or TXT files, indexing chunks with tenant isolation."""
    contents = await file.read()
    if len(contents) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded document exceeds maximum allowed size of 1 MB.",
        )

    try:
        doc_meta, chunks = DOCUMENT_MANAGER.index_uploaded_file(
            tenant_id=tenant["tenant_id"],
            filename=file.filename or "uploaded_document.txt",
            file_bytes=contents,
            title=title,
            source_repository=source_repository,
            version=version,
        )

        # Index into runtime cognitive engine document retriever if available
        if runtime_cognitive_engine and hasattr(runtime_cognitive_engine, "answer_synthesizer"):
            doc_retriever = runtime_cognitive_engine.answer_synthesizer.doc_retriever
            doc_dict = {
                "doc_id": doc_meta.doc_id,
                "title": doc_meta.title,
                "source_repository": doc_meta.source_repository,
                "version": doc_meta.version,
                "chunks": [c.content for c in chunks],
            }
            doc_retriever.index_document(doc_dict=doc_dict, tenant_id=tenant["tenant_id"])

        return DocumentUploadResponse(**doc_meta.model_dump())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.get("/v1/documents", response_model=List[DocumentUploadResponse])
def list_documents(
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Lists all uploaded enterprise documents for the authenticated tenant."""
    docs = DOCUMENT_MANAGER.list_documents(tenant_id=tenant["tenant_id"])
    return [DocumentUploadResponse(**d.model_dump()) for d in docs]


@app.get("/v1/documents/{doc_id}")
def get_document(
    doc_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Retrieves document details and chunks for the authenticated tenant."""
    doc = DOCUMENT_MANAGER.get_document(doc_id=doc_id, tenant_id=tenant["tenant_id"])
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return doc


@app.delete("/v1/documents/{doc_id}")
def delete_document(
    doc_id: str,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Deletes an uploaded document and removes its vector embeddings."""
    deleted = DOCUMENT_MANAGER.delete_document(doc_id=doc_id, tenant_id=tenant["tenant_id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )

    if runtime_cognitive_engine and hasattr(runtime_cognitive_engine, "answer_synthesizer"):
        runtime_cognitive_engine.answer_synthesizer.doc_retriever.delete_document(
            doc_id=doc_id, tenant_id=tenant["tenant_id"]
        )

    return {"deleted": True, "doc_id": doc_id}


@app.post("/v1/documents/search", response_model=List[DocumentSearchHit])
def search_documents(
    req: DocumentSearchRequest,
    tenant: Dict[str, str] = Depends(authenticate_and_rate_limit),
):
    """Searches indexed documents using hybrid dense-vector + lexical token ranking."""
    if runtime_cognitive_engine is None or not hasattr(runtime_cognitive_engine, "answer_synthesizer"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document retrieval engine not initialized.",
        )

    passages = runtime_cognitive_engine.answer_synthesizer.doc_retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        min_score=req.min_score,
        tenant_id=tenant["tenant_id"],
    )

    results = []
    for p in passages:
        results.append(
            DocumentSearchHit(
                passage_id=p.passage_id,
                doc_id=p.doc_id,
                title=p.title,
                content=p.content,
                source_repository=p.source_repository,
                version=p.version,
                relevance_score=p.relevance_score,
                retrieval_timestamp=p.retrieval_timestamp,
            )
        )
    return results


@app.get("/v1/diagnostics")
def get_diagnostics():
    """Provides system diagnostics, runtime parameters, checkpoint status, and DB presence."""
    import torch
    import platform

    is_prod = (os.environ.get("CDM_ENV", "production").lower() in ("production", "prod")) or (
        os.environ.get("CDM_PRODUCTION_MODE", "").strip().lower() in ("1", "true", "yes")
    )
    model_path = os.environ.get("CDM_MODEL_PATH", "experiments/release/cdm-1-general")
    if not os.path.exists(model_path):
        model_path = "experiments/checkpoints/cdm-1-general"
    weights_path = os.path.join(model_path, "model.safetensors")
    manifest_path = os.path.join(model_path, "release_manifest.json")

    return {
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(getattr(runtime_model, "device", "cpu")),
        },
        "model": {
            "initialized": runtime_model is not None,
            "mode": "rule_based_fallback" if getattr(runtime_model, "is_fallback", False) else "neural",
            "is_fallback": getattr(runtime_model, "is_fallback", False),
            "parameters": "158M" if not getattr(runtime_model, "is_fallback", False) else "0",
            "model_path": model_path,
            "weights_exist": os.path.exists(weights_path),
            "manifest_exist": os.path.exists(manifest_path),
            "weights_sha256": VERIFIED_CHECKPOINT_HASH,
            "load_error": model_load_error,
        },
        "databases": {
            "api_keys": os.path.exists(KEY_DB_PATH),
            "metering": os.path.exists(METER_DB_PATH),
            "memory": os.path.exists(MEMORY_DB_PATH),
            "agent_runs": os.path.exists(AGENT_DB_PATH),
        },
        "environment": {
            "is_production": is_prod,
            "data_dir": DATA_DIR,
        },
    }


@app.get("/v1/dev/default-key")
def get_dev_default_key(request: Request):
    """Securely issues an ephemeral developer key for the local playground on loopback only.
    Strictly disabled in production environments.
    """
    is_prod = (os.environ.get("CDM_ENV", "").lower() in ("production", "prod")) or (
        os.environ.get("CDM_PRODUCTION_MODE", "").strip().lower() in ("1", "true", "yes")
    )
    if is_prod:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development key issuance is strictly disabled in production environments.",
        )

    client_host = getattr(request.client, "host", "") if request.client else ""
    is_local = client_host in ("127.0.0.1", "localhost", "::1", "testclient")
    if not is_local:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development key issuance is strictly restricted to local loopback traffic.",
        )

    import secrets
    ephemeral_key = f"cdm_dev_{secrets.token_hex(16)}"
    KEY_MANAGER.register_key(
        raw_key=ephemeral_key,
        tenant_id="tenant_dev_default",
        tier="developer",
        rate_limit_rpm=600,
        label="Ephemeral Local Dev Key",
    )
    return {"api_key": ephemeral_key, "tenant_id": "tenant_dev_default"}


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
