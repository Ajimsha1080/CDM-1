"""Tests for API payload size limits, request validation bounds, and structured error responses (Item 6).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, KEY_MANAGER
from cdm import __version__


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token, _ = KEY_MANAGER.generate_key("test_hardening_tenant")
    return {"Authorization": f"Bearer {token}"}


def test_api_version_matches_package(client):
    """Verifies that /v1/version returns package version 0.1.0."""
    resp = client.get("/v1/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == __version__
    assert data["version"] == "0.1.0"


def test_payload_larger_than_1mb_rejected_413(client):
    """Verifies that request payloads exceeding 1MB are rejected by middleware with HTTP 413."""
    oversized_headers = {"content-length": str(2 * 1024 * 1024)}  # 2MB header
    resp = client.post("/v1/decisions", headers=oversized_headers, json={"text": "oversized"})
    assert resp.status_code == 413
    data = resp.json()
    assert data["error"] == "PayloadTooLarge"
    assert "1 MB" in data["detail"]


def test_structured_features_exceeding_100_keys_rejected(client, auth_headers):
    """Verifies that structured_features with >100 entries is rejected with HTTP 422."""
    huge_dict = {f"k_{i}": i for i in range(101)}
    resp = client.post(
        "/v1/decisions",
        headers=auth_headers,
        json={"text": "testing bounds", "structured_features": huge_dict},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "ValidationError"


def test_candidate_exceeding_max_length_rejected(client, auth_headers):
    """Verifies that candidate strings exceeding 1024 characters are rejected."""
    long_candidate = "A" * 1025
    resp = client.post(
        "/v1/decisions",
        headers=auth_headers,
        json={"text": "testing bounds", "candidates": [long_candidate]},
    )
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "ValidationError"


def test_unauthenticated_request_returns_structured_error(client):
    """Verifies that 401 unauthorized returns structured error response without stack trace."""
    resp = client.post("/v1/decisions", json={"text": "testing auth error"})
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"] == "HTTPException"
    assert "Bearer token required" in data["detail"]
