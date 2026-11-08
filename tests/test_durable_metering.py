"""Tests for Durable Transactional Metering, Tenant Isolation, and Concurrency."""

import os
import tempfile
import pytest
from concurrent.futures import ThreadPoolExecutor
from api.metering import DurableUsageMeter
from fastapi.testclient import TestClient
from api.main import app, KEY_MANAGER


@pytest.fixture
def temp_meter():
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as tf:
        db_path = tf.name
    meter = DurableUsageMeter(db_path=db_path)
    yield meter
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_meter_record_and_summary(temp_meter):
    temp_meter.record_request("tenant_alpha", "/v1/decisions", "binary", 24.5, 200)
    temp_meter.record_request("tenant_alpha", "/v1/decisions", "choice", 31.0, 200)

    summary = temp_meter.get_tenant_summary("tenant_alpha")
    assert summary["tenant_id"] == "tenant_alpha"
    assert summary["total_requests"] == 2
    assert summary["average_latency_ms"] == pytest.approx(27.75, abs=0.1)
    assert summary["last_request_timestamp"] is not None


def test_meter_tenant_isolation(temp_meter):
    # Record requests for tenant_alpha
    for _ in range(5):
        temp_meter.record_request("tenant_alpha", "/v1/decisions", "binary", 15.0, 200)

    # Record requests for tenant_beta
    for _ in range(3):
        temp_meter.record_request("tenant_beta", "/v1/choice", "choice", 40.0, 200)

    summary_a = temp_meter.get_tenant_summary("tenant_alpha")
    summary_b = temp_meter.get_tenant_summary("tenant_beta")
    summary_c = temp_meter.get_tenant_summary("tenant_gamma")

    assert summary_a["total_requests"] == 5
    assert summary_b["total_requests"] == 3
    assert summary_c["total_requests"] == 0


def test_concurrent_request_accounting(temp_meter):
    num_threads = 10
    requests_per_thread = 20

    def worker(worker_id):
        tenant_id = f"tenant_{worker_id % 3}"
        for _ in range(requests_per_thread):
            temp_meter.record_request(tenant_id, "/v1/decisions", "binary", 12.0, 200)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    total_expected = num_threads * requests_per_thread
    total_recorded = (
        temp_meter.get_tenant_summary("tenant_0")["total_requests"]
        + temp_meter.get_tenant_summary("tenant_1")["total_requests"]
        + temp_meter.get_tenant_summary("tenant_2")["total_requests"]
    )
    assert total_recorded == total_expected


def test_api_metering_endpoint():
    test_key = "cdm_test_metering_valid_key"
    KEY_MANAGER.register_key(test_key, "tenant_dev_default", tier="developer", rate_limit_rpm=500)
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {test_key}"}
        # Make a decision to record usage
        r_post = client.post(
            "/v1/decisions",
            json={"text": "Test inquiry for metering.", "task": "binary"},
            headers=headers,
        )
        assert r_post.status_code == 200

        # Query metering endpoint
        r_meter = client.get("/v1/metering", headers=headers)
        assert r_meter.status_code == 200
        data = r_meter.json()
        assert data["tenant_id"] == "tenant_dev_default"
        assert "usage" in data
        assert data["usage"]["total_requests"] >= 1
        assert "average_latency_ms" in data["usage"]
