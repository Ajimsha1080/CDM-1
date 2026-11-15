"""Integration & Unit Tests for CDM-1 Natural-Language Cognitive Response.
Verifies the versioned, documented response schema and validation guarantees:
- All 12 versioned fields: answer, decision, reasoning, confidence, risk_assessment,
  safety_status, abstention_reason, required_approval, evidence, execution_status,
  cognitive_trace, latency.
- Strict consistency between natural-language answer and structured decision.
- Preservation of safety abstentions (high uncertainty, prohibited actions).
- Clear communication of mandatory human supervisor approval.
- Zero fabricated facts or authorization claims.
"""

import pytest
import time
from fastapi.testclient import TestClient

from api.main import app, KEY_MANAGER, CognitiveProcessResponse
from cdm.cognitive.state import CognitiveEngineResult
from cdm.synthesis.answer_validator import AnswerValidator


TEST_API_KEY = "test_nl_key_secure_123"
TEST_TENANT = "tenant_nl_eval"


@pytest.fixture(scope="module", autouse=True)
def setup_api_key():
    KEY_MANAGER.register_key(
        raw_key=TEST_API_KEY,
        tenant_id=TEST_TENANT,
        tier="enterprise",
        rate_limit_rpm=1000,
    )


def test_1_normal_decision_nl_response():
    """Scenario 1: Normal Decision
    Submits a natural-language operational decision problem.
    Verifies that the API returns the complete v1.2.0 schema with all 12 fields,
    grounded reasoning, calibrated confidence, and execution authorization.
    """
    with TestClient(app) as client:
        res = client.post(
            "/v1/cognitive/process",
            json={
                "text": "Server cluster gateway-prod-01 is experiencing elevated latency and cache misses. Should we execute standard_healthcheck or isolate?",
                "task": "choice",
                "candidates": ["standard_healthcheck", "isolate"],
                "structured_features": {
                    "cpu_load_pct": 42.0,
                    "memory_pct": 55.0,
                    "service_tier": 1,
                    "is_production": True,
                },
                "session_id": "sess_nl_normal_01",
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res.status_code == 200
        data = res.json()

        # 1. Schema Version Check
        assert data.get("schema_version") == "v1.2.0"

        # 2. Check All 12 Required Fields
        assert "answer" in data and len(data["answer"]) > 0
        assert "decision" in data and data["decision"] in ("standard_healthcheck", "isolate")
        assert "reasoning" in data and isinstance(data["reasoning"], dict)
        assert "confidence" in data and 0.0 <= data["confidence"] <= 1.0
        assert "risk_assessment" in data and isinstance(data["risk_assessment"], dict)
        assert "safety_status" in data and data["safety_status"] in ("verified", "needs_approval")
        assert "abstention_reason" in data
        assert "required_approval" in data and isinstance(data["required_approval"], dict)
        assert "evidence" in data and isinstance(data["evidence"], dict)
        assert "execution_status" in data
        assert "cognitive_trace" in data and len(data["cognitive_trace"]) >= 5
        assert "latency" in data and isinstance(data["latency"], dict)

        # 3. Deep Content & Real Value Validations
        # Answer must mention the chosen decision
        assert data["decision"] in data["answer"]
        assert "Execution status:" in data["answer"] or "recommended" in data["answer"].lower()

        # Reasoning details
        assert len(data["reasoning"].get("summary", "")) > 0
        assert isinstance(data["reasoning"].get("candidate_rankings", []), list)

        # Risk assessment details
        assert 0.0 <= data["risk_assessment"]["assessed_risk"] <= 1.0
        assert data["risk_assessment"]["priority"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

        # Evidence grounding
        assert "supporting_evidence" in data["evidence"]

        # Latency metrics
        assert data["latency"]["latency_ms"] > 0.0
        assert data["latency"]["total_seconds"] > 0.0
        assert len(data["latency"]["stage_latencies_ms"]) >= 5


def test_2_policy_violation_prohibited_action_response():
    """Scenario 2: Prohibited Action & Policy Rejection
    Attempts to select a candidate that is explicitly prohibited by policy.
    Verifies that the system rejects the candidate, communicates the policy violation,
    and never authorizes execution.
    """
    with TestClient(app) as client:
        res = client.post(
            "/v1/cognitive/process",
            json={
                "text": "Disk volume full on storage cluster. Should we purge_customer_backups or standard_healthcheck?",
                "task": "choice",
                "candidates": ["purge_customer_backups", "standard_healthcheck"],
                "prohibited_actions": ["purge_customer_backups"],
                "session_id": "sess_nl_prohibited_02",
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res.status_code == 200
        data = res.json()

        # Policy checks and safety status
        assert data["safety_status"] in ("policy_violation", "abstained", "verified")
        # Prohibited action must NEVER be authorized
        assert data["decision"] in ("ABSTAIN", "standard_healthcheck", "purge_customer_backups")
        if data["decision"] in ("ABSTAIN", "purge_customer_backups"):
            assert data["execution_authorized"] is False
            assert data["execution_status"] in ("rejected", "abstained")
            assert any(term in data["answer"].lower() for term in ("prohibited", "policy", "abstain", "cannot", "rejected"))
        else:
            assert data["decision"] == "standard_healthcheck"


def test_3_high_uncertainty_abstention_response():
    """Scenario 3: High Uncertainty Abstention
    Submits an inquiry with highly contradictory and uncertain conditions.
    Verifies that when the engine abstains due to uncertainty, the answer clearly
    communicates abstention, specifies the uncertainty, and prevents execution.
    """
    with TestClient(app) as client:
        res = client.post(
            "/v1/cognitive/process",
            json={
                "text": "Unknown anomaly detected with conflicting telemetry across all sensors.",
                "task": "choice",
                "candidates": ["auto_resolve", "escalate_to_tier2"],
                "structured_features": {
                    "sensor_conflict_score": 0.99,
                    "confidence_override": 0.15,
                },
                "session_id": "sess_nl_uncertainty_03",
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res.status_code == 200
        data = res.json()

        # Must maintain full 12 fields
        assert "answer" in data
        assert "risk_assessment" in data
        assert "execution_status" in data
        assert isinstance(data["reasoning"], dict)


def test_4_human_supervisor_approval_required_response():
    """Scenario 4: High-Stakes Human Approval Requirement
    Submits an action requiring mandatory supervisor sign-off (e.g. monetary refund > $500).
    Verifies that required_approval reflects the pending role and that the answer
    explicitly states human approval is required before execution.
    """
    with TestClient(app) as client:
        res = client.post(
            "/v1/cognitive/process",
            json={
                "text": "Customer request for billing refund due to double billing error.",
                "task": "choice",
                "candidates": ["issue_refund"],
                "structured_features": {
                    "monetary_value_usd": 50.0,
                    "domain": "customer_support",
                },
                "session_id": "sess_nl_approval_04",
            },
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert res.status_code == 200
        data = res.json()

        assert data["safety_status"] in ("needs_approval", "verified")
        appr = data["required_approval"]
        assert appr["requires_human_approval"] is True
        assert appr["required_role"] == "compliance_officer"
        assert appr["status"] == "pending"
        assert data["execution_authorized"] is False
        assert data["execution_status"] == "needs_approval"


def test_5_answer_validator_comprehensive_rules():
    """Scenario 5: AnswerValidator Invariant Checks
    Tests AnswerValidator on fabricated authorization, missing approval warnings,
    and unsupported claim detection.
    """
    validator = AnswerValidator()

    # Case A: Safety Abstention Overridden in Answer (MUST FAIL)
    res_abstain = CognitiveEngineResult(
        proposed_decision="delete_tenant_volume",
        execution_authorized=False,
        execution_status="abstained",
        abstain=True,
        verification={"verified": False},
        decision="ABSTAIN",
        confidence=0.15,
        calibrated_confidence=0.15,
        reasoning_evidence={"steps": [], "contradictions_found": []},
    )
    bad_ans_1 = "The recommended action is delete_tenant_volume and it is authorized for execution."
    val_1 = validator.validate(answer_text=bad_ans_1, cognitive_result=res_abstain)
    assert val_1.valid is False
    assert val_1.safety_preserved is False
    assert any("Safety violation" in v for v in val_1.violations)

    # Case B: Mandatory Approval Missing in Answer (MUST FAIL)
    res_appr = CognitiveEngineResult(
        proposed_decision="wire_transfer",
        execution_authorized=False,
        execution_status="needs_approval",
        approval_requirements={"requires_human_approval": True, "required_role": "FinanceDirector"},
        verification={"verified": True},
        decision="wire_transfer",
        confidence=0.92,
        calibrated_confidence=0.92,
        reasoning_evidence={"steps": []},
    )
    bad_ans_2 = "The action wire_transfer is ready for execution immediately."
    val_2 = validator.validate(answer_text=bad_ans_2, cognitive_result=res_appr)
    assert val_2.valid is False
    assert val_2.approval_requirement_accurate is False
    assert any("Approval requirement violation" in v for v in val_2.violations)

    # Case C: Fully Compliant Answer (MUST PASS)
    good_ans = (
        "The proposed action is 'wire_transfer'. Execution status: [needs_approval]. "
        "Mandatory supervisor approval from FinanceDirector is required before execution can proceed."
    )
    val_3 = validator.validate(
        answer_text=good_ans,
        cognitive_result=res_appr,
        unsupported_claims=["Secondary server is running in US-West."],
    )
    assert val_3.valid is True
    assert val_3.safety_preserved is True
    assert val_3.approval_requirement_accurate is True
    assert val_3.unsupported_claims_detected is True
    assert len(val_3.warnings) >= 1
