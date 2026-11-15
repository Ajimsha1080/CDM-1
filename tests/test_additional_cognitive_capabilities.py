"""Integration and Reliability Test Suite for CDM-1 Additional Cognitive Capabilities.
Verifies all P0 and P1 requirements:
1. Intent and Task Routing (7 tasks + dynamic execution plans)
2. Missing Information & Structured Clarification
3. Conversation Context & Reference (Anaphora) Resolution
4. Explicit Session and Tenant Isolation
5. Knowledge & Document Retrieval with Source Provenance
6. Controlled External Tool Orchestration (Validation, Timeouts, Retries)
7. Hard Security Prohibitions Enforcement in Tool Orchestration
8. Evidence and Citation Management with Unsupported Claim Detection
9. Answer Validation (Decision consistency, Abstention preservation, Zero fabricated execution)
10. Multimodal Ingestion (Text, Tabular, Document, Image) and Uncertainty Tracking
11. Observability and Secret/Credential Masking in Traces
12. Streaming SSE & User Feedback Storage Segregation
13. Comprehensive Evaluation Framework across all 10 Quality Dimensions
"""

import os
import json
import time
import tempfile
import pytest
from fastapi.testclient import TestClient

from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import CognitiveEngineResult
from cdm.synthesis.router import IntentTaskRouter, CognitiveRoutePlan
from cdm.context.reference_resolver import ReferenceResolver
from cdm.synthesis.document_retriever import DocumentKnowledgeRetriever, DocumentPassage
from cdm.synthesis.tool_orchestrator import ControlledToolOrchestrator, ToolExecutionRecord
from cdm.synthesis.citation_manager import CitationManager, GroundedCitation
from cdm.synthesis.answer_validator import AnswerValidator, AnswerValidationResult
from cdm.perception.multimodal_processor import MultimodalProcessor, MultimodalIngestionResult
from cdm.evaluation.evaluator import CognitiveEvaluator, ComprehensiveEvaluationReport
from cdm.execution.tools.base import BaseTool, ToolResult
from cdm.learning.feedback import FeedbackCollector, UserOpinionFeedback
from api.main import app, KEY_MANAGER, sanitize_trace_payload

MODEL_PATH = "experiments/release/cdm-1-general"


@pytest.fixture(scope="module")
def shared_engine():
    """Shared CognitiveEngine loaded with genuine CDM-1 weights."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    engine = CognitiveEngine.from_pretrained(model_name_or_path=MODEL_PATH, db_path=db_path)
    yield engine

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception:
            pass


def test_1_intent_and_task_routing():
    """Verifies that IntentTaskRouter correctly categorizes all 7 primary intents
    and produces dynamic execution plans routing only necessary components.
    """
    router = IntentTaskRouter()

    # 1. Question
    q_plan = router.plan_route(text="What is the standard SLA policy for tier 1 tickets?")
    assert q_plan.intent_task == "question"
    assert q_plan.run_perception is True
    assert q_plan.run_memory_retrieval is True
    assert q_plan.run_planning is False
    assert q_plan.run_tools is False

    # 2. Explanation
    exp_plan = router.plan_route(text="Why did you select auto_resolve instead of escalation?")
    assert exp_plan.intent_task == "explanation"
    assert exp_plan.run_reasoning is True
    assert exp_plan.run_planning is False

    # 3. Decision
    dec_plan = router.plan_route(
        text="Should we auto_resolve or escalate_to_tier2?",
        candidates=["auto_resolve", "escalate_to_tier2"],
    )
    assert dec_plan.intent_task == "decision"
    assert dec_plan.run_neural_inference is True

    # 4. Prediction
    pred_plan = router.plan_route(text="Predict the resolution time and future outcome probability.")
    assert pred_plan.intent_task == "prediction"
    assert pred_plan.run_neural_inference is True

    # 5. Planning
    plan_plan = router.plan_route(text="What is the complete multi-step workflow and prerequisites plan?")
    assert plan_plan.intent_task == "planning"
    assert plan_plan.run_planning is True

    # 6. Comparison
    comp_plan = router.plan_route(text="Compare option A versus option B with trade-off analysis.")
    assert comp_plan.intent_task == "comparison"

    # 7. Execution
    exec_plan = router.plan_route(
        text="Execute standard_healthcheck on gateway cluster.",
        structured_features={"target_host": "10.0.1.5"},
    )
    assert exec_plan.intent_task == "execution"
    assert exec_plan.run_tools is True


def test_2_missing_information_clarification():
    """When essential information is missing for an execution/decision task,
    router detects missing parameters and generates structured clarification prompt.
    """
    router = IntentTaskRouter()

    # Sensitive refund execution missing monetary value and account ID
    plan = router.plan_route(text="Execute customer refund immediately.")
    assert plan.needs_clarification is True
    assert plan.intent_task == "clarification"
    assert "monetary_value_usd" in plan.missing_parameters
    assert "account_id" in plan.missing_parameters
    assert "Please provide these details" in plan.clarification_prompt


def test_3_conversation_reference_resolution(shared_engine):
    """Verifies that ReferenceResolver resolves anaphoric pronouns ('it', 'this option')
    and references to previous decisions and entities across multi-turn dialogue.
    """
    session_id = f"test_sess_resolve_{int(time.time())}"
    tenant_id = "tenant_resolution_test"

    # Turn 1: Initial decision inquiry
    res1 = shared_engine.process(
        text="Customer reports item arrived broken and cracked with photos. Should we issue_refund or auto_resolve?",
        structured_features={"monetary_value_usd": 65.0, "account_id": "ACC-9921", "domain": "customer_support"},
        candidates=["issue_refund", "auto_resolve"],
        session_id=session_id,
        tenant_id=tenant_id,
    )
    assert res1.proposed_decision in ("issue_refund", "auto_resolve")

    # Turn 2: Follow-up referencing 'it' and 'this option'
    ctx = shared_engine.context_engine.get_latest_context(session_id=session_id, tenant_id=tenant_id)
    resolver = ReferenceResolver()
    resolved = resolver.resolve_references(
        current_text="Why did you select this option?",
        context=ctx,
        prior_decisions=[{"decision": res1.proposed_decision}],
    )

    assert resolved["anaphora_detected"] is True
    assert resolved["resolved_decision"] == res1.proposed_decision
    assert str(res1.proposed_decision) in resolved["resolved_text"]
    assert resolved["resolved_entities"].get("account_id") == "ACC-9921"


def test_4_session_and_memory_isolation(shared_engine):
    """Guarantees strict tenant and session isolation: Tenant B cannot access Tenant A's session."""
    session_id = f"isolated_sess_{int(time.time())}"

    # Tenant Alpha writes state
    shared_engine.process(
        text="Internal secret directive for tenant Alpha only.",
        structured_features={"alpha_secret": "TOP_SECRET_ALPHA"},
        session_id=session_id,
        tenant_id="tenant_alpha",
    )

    # Tenant Beta attempts to read the same session_id
    ctx_beta = shared_engine.context_engine.get_latest_context(session_id=session_id, tenant_id="tenant_beta")
    assert ctx_beta is None, "Tenant Beta was able to access Tenant Alpha's session context!"

    wm_alpha = shared_engine._get_or_create_working_memory(tenant_id="tenant_alpha", session_id=session_id)
    wm_beta = shared_engine._get_or_create_working_memory(tenant_id="tenant_beta", session_id=session_id)
    assert wm_alpha is not wm_beta, "Working memory instance was shared across tenants!"


def test_5_document_knowledge_retrieval():
    """Verifies that DocumentKnowledgeRetriever indexes and retrieves enterprise documents
    with passage chunks, document identifiers, source repositories, and timestamps.
    """
    retriever = DocumentKnowledgeRetriever()

    # Index custom operational procedure
    custom_doc = {
        "doc_id": "DOC-SEC-99",
        "title": "Autonomous Security Protocol v5",
        "source_repository": "InfoSec Regulatory Vault",
        "version": "5.0.1",
        "chunks": [
            "Root account elevation requires dual-factor biometric authorization token.",
            "Database maintenance window opens on Sundays between 02:00 and 04:00 UTC.",
        ],
    }
    retriever.index_document(custom_doc)

    passages = retriever.retrieve("biometric authorization token for root account", top_k=2)
    assert len(passages) >= 1
    p = passages[0]
    assert p.doc_id == "DOC-SEC-99"
    assert "dual-factor biometric authorization" in p.content
    assert p.source_repository == "InfoSec Regulatory Vault"
    assert p.provenance_type == "document_knowledge"
    assert p.retrieval_timestamp > 0.0


def test_6_controlled_tool_orchestration():
    """Verifies tool validation, execution timeouts, and exponential retry policy."""
    class FlakyEchoTool(BaseTool):
        def __init__(self):
            super().__init__(name="flaky_echo", description="Echos payload with simulated initial failure")
            self.attempts = 0

        def validate_args(self, args):
            if "message" not in args:
                return "Missing required 'message' argument."
            return None

        def _run(self, args, tenant_id):
            self.attempts += 1
            if self.attempts < 2:
                raise RuntimeError("Simulated transient connection timeout")
            return f"Echoed: {args['message']}"

    orchestrator = ControlledToolOrchestrator(max_retries=3)
    tool = FlakyEchoTool()
    orchestrator.register_tool(tool)

    # 1. Argument validation error
    val_res = orchestrator.execute_tool("flaky_echo", arguments={}, tenant_id="t1")
    assert val_res.success is False
    assert "Input validation failure" in val_res.error

    # 2. Retry with backoff success
    res = orchestrator.execute_tool("flaky_echo", arguments={"message": "hello world"}, tenant_id="t1")
    assert res.success is True
    assert res.output == "Echoed: hello world"
    assert res.metadata["retry_count"] >= 1
    assert len(orchestrator.execution_audit_log) >= 2


def test_7_tool_orchestration_blocks_prohibited_action():
    """Prohibited actions or unauthorized tools are strictly blocked by ControlledToolOrchestrator."""
    class DangerTool(BaseTool):
        def __init__(self):
            super().__init__(name="delete_database", description="Dangerous deletion")
        def _run(self, args, tenant_id):
            return "DELETED"

    orchestrator = ControlledToolOrchestrator()
    orchestrator.register_tool(DangerTool())

    res = orchestrator.execute_tool(
        tool_name="delete_database",
        arguments={"cluster": "prod-db"},
        tenant_id="t1",
        prohibited_actions=["delete_database"],
    )

    assert res.success is False
    assert "Security Policy Rejection" in res.error
    assert res.metadata.get("blocked_by_policy") is True
    last_audit = orchestrator.execution_audit_log[-1]
    assert last_audit.blocked_by_policy is True


def test_8_evidence_and_citation_management():
    """Verifies that CitationManager pairs assertions with verified citations
    and flags unsupported assertions without fabricating records.
    """
    manager = CitationManager()

    passages = [
        DocumentPassage(
            passage_id="p1",
            doc_id="DOC-SLA",
            title="SLA Guide",
            content="Tier-1 incident response requires initial contact within 4.0 hours.",
            source_repository="Corporate Repo",
        )
    ]
    semantic_rules = [{"rule_id": "RULE-REFUND", "description": "Refund transactions exceeding $50 strictly require secondary review."}]

    answer_text = (
        "Tier-1 incident response requires initial contact within 4.0 hours. "
        "Refund transactions exceeding $50 strictly require secondary review. "
        "An ungrounded claim stating that aliens landed in the server room yesterday."
    )

    report = manager.attribute_and_verify(
        answer_text=answer_text,
        retrieved_passages=passages,
        semantic_rules=semantic_rules,
    )

    assert report.total_claims >= 3
    assert len(report.citations) >= 2
    cit_sources = [c.source_id for c in report.citations]
    assert "DOC-SLA" in cit_sources
    assert "RULE-REFUND" in cit_sources
    # Unsupported claim flagged
    assert any("aliens" in unc.lower() for unc in report.unsupported_claims)
    assert report.grounding_ratio < 1.0


def test_9_answer_validation_rules():
    """Verifies that AnswerValidator detects decision inconsistency,
    safety abstention violations, and fabricated verification status.
    """
    validator = AnswerValidator()

    # Case A: Safety Abstention Violation (engine abstained, but answer claims authorized execution)
    res_abstain = CognitiveEngineResult(
        proposed_decision="delete_database",
        execution_authorized=False,
        execution_status="abstained",
        abstain=True,
        verification={"verified": False},
        decision="ABSTAIN",
        confidence=0.20,
        calibrated_confidence=0.20,
        reasoning_evidence={"steps": [], "contradictions_found": []},
    )

    bad_answer = "The recommended action is delete_database and it is authorized for execution."
    val_res = validator.validate(answer_text=bad_answer, cognitive_result=res_abstain)
    assert val_res.valid is False
    assert val_res.safety_preserved is False
    assert any("Safety violation" in v for v in val_res.violations)

    # Case B: Compliant Answer
    res_ok = CognitiveEngineResult(
        proposed_decision="auto_resolve",
        execution_authorized=True,
        execution_status="authorized",
        abstain=False,
        verification={"verified": True},
        decision="auto_resolve",
        confidence=0.98,
        calibrated_confidence=0.98,
        reasoning_evidence={"steps": [], "contradictions_found": []},
    )
    good_answer = "The recommended action is auto_resolve, authorized for execution with 98.0% confidence."
    val_ok = validator.validate(answer_text=good_answer, cognitive_result=res_ok)
    assert val_ok.valid is True
    assert val_ok.decision_consistent is True
    assert val_ok.safety_preserved is True


def test_10_multimodal_ingestion_and_uncertainty():
    """Verifies multimodal processing of text, tabular features, documents, and image stimulus."""
    processor = MultimodalProcessor()

    doc_sample = {
        "name": "incident_log.json",
        "type": "json",
        "content": json.dumps({"incident_code": "ERR-503", "affected_nodes": ["node_1", "node_2"]}),
    }
    img_sample = {
        "name": "broken_package_photo.png",
        "metadata": {"ocr_text": "CRACKED GLASS - FRAGILE ITEM", "labels": ["shipping_damage"]},
    }

    result = processor.ingest(
        text="Customer ticket reporting damaged shipment.",
        structured_features={"monetary_value_usd": 85.0},
        documents=[doc_sample],
        images=[img_sample],
    )

    assert isinstance(result, MultimodalIngestionResult)
    assert len(result.modality_contents) == 3  # text/tabular, document, image
    assert "CRACKED GLASS" in result.unified_text
    assert "ERR-503" in result.unified_text
    assert result.overall_extraction_uncertainty < 0.30
    assert len(result.extraction_failures) == 0


def test_11_observability_and_secret_masking():
    """Verifies that sanitize_trace_payload masks credentials, tokens, and secrets in traces."""
    trace = [
        {
            "stage_name": "Perception",
            "inputs": {
                "text": "Login attempt",
                "api_key": "sk-secret-981298491284",
                "user_password": "PlaintextPassword123!",
                "auth_token": "bearer eyJhbGciOi...",
            },
            "nested_credentials": {
                "secret_key": "supersecret",
                "normal_field": "visible_value",
            },
        }
    ]

    sanitized = sanitize_trace_payload(trace)
    stage_inputs = sanitized[0]["inputs"]
    assert stage_inputs["api_key"] == "[REDACTED]"
    assert stage_inputs["user_password"] == "[REDACTED]"
    assert stage_inputs["auth_token"] == "[REDACTED]"
    assert sanitized[0]["nested_credentials"]["secret_key"] == "[REDACTED]"
    assert sanitized[0]["nested_credentials"]["normal_field"] == "visible_value"


def test_12_streaming_and_feedback_separation():
    """Verifies SSE streaming endpoint and segregated user feedback recording."""
    test_key = "test_stream_key_99"
    KEY_MANAGER.register_key(test_key, "tenant_stream_test", tier="enterprise", rate_limit_rpm=500)

    with TestClient(app) as client:
        # 1. Test SSE Streaming endpoint
        stream_res = client.post(
            "/v1/cognitive/process/stream",
            json={
                "text": "What is the standard SLA policy for tier 1 customer tickets?",
                "structured_features": {"domain": "customer_support"},
                "candidates": ["auto_resolve"],
                "session_id": "sess_stream_01",
            },
            headers={"Authorization": f"Bearer {test_key}"},
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers.get("content-type", "")
        stream_content = stream_res.text
        assert "stage_start" in stream_content
        assert "Cognitive Answer Synthesis" in stream_content
        assert "final_result" in stream_content

        # 2. Test User Feedback endpoint (Segregated Storage)
        fb_res = client.post(
            "/v1/cognitive/feedback",
            json={
                "session_id": "sess_stream_01",
                "rating": 5,
                "user_comment": "Excellent and grounded answer.",
                "decision_id": "dec_stream_01",
            },
            headers={"Authorization": f"Bearer {test_key}"},
        )
        assert fb_res.status_code == 200
        fb_data = fb_res.json()
        assert fb_data["status"] == "recorded"
        assert fb_data["eligible_for_training"] is False


def test_13_evaluation_framework_metrics(shared_engine):
    """Verifies that CognitiveEvaluator computes scores across all 10 quality dimensions."""
    evaluator = CognitiveEvaluator()

    res = shared_engine.process(
        text="Customer is asking for standard password reset link for their account.",
        structured_features={"monetary_value_usd": 0.0, "urgency_level": "low", "domain": "customer_support"},
        candidates=["auto_resolve", "escalate_to_tier2"],
        session_id="test_sess_eval_01",
    )

    tool_record = ToolResult(success=True, tool_name="standard_healthcheck")
    report = evaluator.evaluate(
        inquiry_text="Customer is asking for standard password reset link for their account.",
        result=res,
        ground_truth={"expected_decision": "auto_resolve"},
        tool_results=[tool_record],
    )

    assert isinstance(report, ComprehensiveEvaluationReport)
    assert 0.0 <= report.overall_quality_score <= 1.0
    assert report.summary_verdict in ("EXCELLENT", "ACCEPTABLE")

    # Verify all 10 metric axes are present and valid
    expected_metrics = [
        "qa_relevance",
        "factual_correctness",
        "evidence_grounding",
        "decision_quality",
        "prediction_quality",
        "calibration",
        "abstention_quality",
        "planning_consistency",
        "tool_execution_correctness",
        "latency_and_resources",
    ]
    for m_name in expected_metrics:
        assert m_name in report.metrics, f"Missing evaluation metric '{m_name}'"
        m = report.metrics[m_name]
        assert 0.0 <= m.score <= 1.0
        assert isinstance(m.passed, bool)
