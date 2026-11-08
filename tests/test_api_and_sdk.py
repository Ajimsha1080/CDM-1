"""Integration tests for CDM-1 Production-Hardened FastAPI endpoints and Python SDK."""

import pytest
from fastapi.testclient import TestClient
from api.main import app, KEY_MANAGER
from sdk.cdm_sdk import CDM, CDMAuthenticationError, CDMError

TEST_API_KEY = "cdm_test_runner_valid_suite_key"


@pytest.fixture(scope="module", autouse=True)
def setup_test_credentials():
    KEY_MANAGER.register_key(TEST_API_KEY, "tenant_dev_default", tier="developer", rate_limit_rpm=500)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_readiness(client):
    r_health = client.get("/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_ready = client.get("/v1/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["status"] == "ready"


def test_auth_rejection_missing_and_invalid(client):
    # Missing authorization header
    r_missing = client.get("/v1/models")
    assert r_missing.status_code == 401

    # Invalid authorization token
    headers = {"Authorization": "Bearer invalid_secret_key"}
    r_invalid = client.get("/v1/models", headers=headers)
    assert r_invalid.status_code == 401


def test_post_decision_authenticated(client):
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    payload = {
        "text": "Production server memory leak detected on payment cluster.",
        "structured_features": {
            "sentiment_score": -0.8,
            "interaction_count": 2,
            "account_tenure_months": 24,
            "monetary_value_usd": 15000,
            "sla_remaining_hours": 1.0,
            "domain": "business_operations",
            "urgency_level": "critical",
        },
        "task": "binary",
    }
    r = client.post("/v1/decisions", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] in ["YES", "NO"]
    assert "confidence" in data
    assert "priority" in data
    assert "risk" in data
    assert data["tenant_id"] == "tenant_dev_default"


def test_candidate_choice_with_constraint(client):
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    payload = {
        "text": "Customer requests full refund on order delivered 80 days ago.",
        "task": "choice",
        "candidates": ["auto_resolve", "issue_refund", "escalate_to_tier2"],
        "prohibited_actions": ["issue_refund"],
    }
    r = client.post("/v1/choice", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["selected_candidate"] != "issue_refund"
    assert data["candidate_scores"]["issue_refund"] == 0.0


def test_payload_limits_validation(client):
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    # Exceed candidate count (max 32)
    too_many_cands = [f"cand_{i}" for i in range(35)]
    payload = {
        "text": "Testing limits",
        "task": "choice",
        "candidates": too_many_cands,
    }
    r = client.post("/v1/choice", json=payload, headers=headers)
    assert r.status_code == 422


def test_sdk_explicit_mode_enforcement():
    # Local mode succeeds offline
    client_local = CDM(mode="local", local_model_path="experiments/release/cdm-1-general")
    assert client_local.mode == "local"
    res = client_local.decide(text="Inquiry regarding billing invoice.", task="binary")
    assert res.decision in ["YES", "NO"]

    # Hosted mode with invalid base_url raises CDMError instead of silently switching to local!
    client_hosted = CDM(mode="hosted", base_url="http://127.0.0.1:59999", api_key="sk_test", timeout=1.0)
    assert client_hosted.mode == "hosted"
    with pytest.raises(CDMError):
        client_hosted.decide(text="Test inquiry", task="binary")


def test_api_and_sdk_cognitive_process(client):
    """Verifies the new /v1/cognitive/process REST endpoint and SDK .process() method."""
    # 1. REST API endpoint
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    payload = {
        "text": "Customer reports critical security breach on server ACC-99212.",
        "structured_features": {
            "monetary_value_usd": 15000.0,
            "sla_remaining_hours": 0.5,
            "domain": "business_operations",
            "urgency_level": "critical",
        },
        "candidates": ["auto_resolve", "isolate_server", "escalate_to_tier2"],
        "prohibited_actions": ["auto_resolve"],
        "goals": ["Contain breach and prevent lateral movement"],
        "session_id": "test_api_session",
    }
    r = client.post("/v1/cognitive/process", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["decision"] != "auto_resolve"
    assert data["selected_plan"] is not None
    assert "reasoning_evidence" in data
    assert "world_state_summary" in data

    # 2. Local SDK .process()
    client_local = CDM(mode="local", local_model_path="experiments/release/cdm-1-general")
    sdk_res = client_local.process(
        text="Customer reports item arrived damaged with broken glass.",
        structured_features={"monetary_value_usd": 45.0, "urgency_level": "medium"},
        candidates=["auto_resolve", "issue_refund", "escalate_to_tier2"],
        prohibited_actions=["auto_resolve"],
        goals=["Resolve customer damaged delivery"],
    )
    assert sdk_res.decision != "auto_resolve"
    assert sdk_res.selected_plan is not None
    assert len(sdk_res.selected_plan["steps"]) >= 1
