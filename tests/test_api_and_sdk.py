"""Integration tests for CDM-1 FastAPI endpoints and Python SDK.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from sdk.cdm_sdk import CDM


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_and_version(client):
    r_health = client.get("/v1/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_ver = client.get("/v1/version")
    assert r_ver.status_code == 200
    assert "version" in r_ver.json()


def test_auth_rejection(client):
    headers = {"Authorization": "Bearer invalid_key_12345"}
    r = client.get("/v1/models", headers=headers)
    assert r.status_code == 401


def test_post_decision(client):
    headers = {"Authorization": "Bearer sk_test_developer_key"}
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
    assert "latency_ms" in data


def test_candidate_choice_with_constraint(client):
    headers = {"Authorization": "Bearer sk_test_developer_key"}
    payload = {
        "text": "Customer requests full refund on order delivered 80 days ago.",
        "task": "choice",
        "candidates": ["auto_resolve", "issue_refund", "escalate_to_tier2"],
        "prohibited_actions": ["issue_refund"],  # Hard policy constraint
    }
    r = client.post("/v1/choice", json=payload, headers=headers)
    assert r.status_code == 200
    data = r.json()
    assert data["selected_candidate"] != "issue_refund"
    assert data["candidate_scores"]["issue_refund"] == 0.0


def test_sdk_local_execution():
    client_local = CDM(local_model_path="experiments/release/cdm-1-general")
    res = client_local.decide(
        text="Customer asks for order status update.",
        structured_features={"urgency_level": "low"},
        task="binary",
    )
    assert res.decision in ["YES", "NO"]
    assert res.confidence >= 0.0
    assert not res.abstain
