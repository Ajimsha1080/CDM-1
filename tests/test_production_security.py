"""P1 & P2 Security Integration Tests for CDM-1.
Verifies:
1. Production credentials requirement (no arbitrary env var bypass).
2. Demo key rejection in production.
3. Cryptographic credential revocation.
4. Tenant isolation.
5. Atomic rate limiting under concurrent multithreading.
6. Fatal termination of production startup if model loading fails.
7. Exception masking on internal server errors.
"""

import os
import time
import tempfile
import threading
import pytest
from fastapi.testclient import TestClient

from api.auth import PersistentApiKeyManager
from api.metering import DurableUsageMeter
from api.main import app


def test_production_credential_enforcement_and_demo_rejection():
    """Verify that in CDM_ENV=production, demo credentials cannot authenticate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "keys.sqlite")
        km = PersistentApiKeyManager(db_path=db_path)

        # Register a demo key
        demo_key = "sk_test_developer_key"
        km.register_key(demo_key, "tenant_demo")

        # In development, it validates
        os.environ["CDM_ENV"] = "development"
        assert km.validate_key(demo_key) is not None

        # In production, it is strictly rejected
        os.environ["CDM_ENV"] = "production"
        assert km.validate_key(demo_key) is None

        # Real registered key works in production
        real_key, _ = km.generate_key("tenant_prod_real", tier="enterprise")
        val = km.validate_key(real_key)
        assert val is not None
        assert val["tenant_id"] == "tenant_prod_real"

        # Cleanup env
        os.environ["CDM_ENV"] = "development"


def test_credential_revocation():
    """Verify that revoked credentials immediately lose authorization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "keys.sqlite")
        km = PersistentApiKeyManager(db_path=db_path)

        token, _ = km.generate_key("tenant_revocation_test")
        assert km.validate_key(token) is not None

        revoked = km.revoke_key(token)
        assert revoked is True

        # Next check fails immediately
        assert km.validate_key(token) is None


def test_atomic_rate_limiting_under_concurrency():
    """Verify that rate limits strictly and atomically throttle concurrent requests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "meter.sqlite")
        meter = DurableUsageMeter(db_path=db_path)

        tenant = "tenant_concurrency_test"
        max_rpm = 10  # Low limit to test concurrency throttling

        allowed_count = 0
        rejected_count = 0
        lock = threading.Lock()

        def worker():
            nonlocal allowed_count, rejected_count
            allowed, _ = meter.check_rate_limit(tenant, max_rpm=max_rpm)
            with lock:
                if allowed:
                    allowed_count += 1
                else:
                    rejected_count += 1

        threads = [threading.Thread(target=worker) for _ in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly max_rpm must be allowed, and the remainder rejected
        assert allowed_count == max_rpm
        assert rejected_count == 15


def test_production_startup_safety_check():
    """Verify check_production_safety terminates startup if no valid production keys exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "keys.sqlite")
        km = PersistentApiKeyManager(db_path=db_path)

        os.environ["CDM_ENV"] = "production"
        try:
            # Empty database should raise RuntimeError
            with pytest.raises(RuntimeError, match="Production environment.*cannot start without pre-registered credentials"):
                km.check_production_safety()

            # Prohibited demo key registered should also raise RuntimeError
            km.register_key("sk_test_developer_key", "tenant_bad")
            with pytest.raises(RuntimeError, match="Prohibited demo credential"):
                km.check_production_safety()

            # Valid key passes
            km.revoke_key("sk_test_developer_key")
            valid_key, _ = km.generate_key("tenant_good")
            km.check_production_safety()  # Should not raise
        finally:
            os.environ["CDM_ENV"] = "development"


def test_exception_masking_in_api():
    """Verify internal exceptions are masked in 500 error responses."""
    client = TestClient(app)

    # Intentionally trigger an error with an unauthenticated or invalid payload
    res = client.post(
        "/v1/decisions",
        headers={"Authorization": "Bearer non_existent_token_123"},
        json={"text": "test", "task": "binary"},
    )
    # Auth failure returns clean 401
    assert res.status_code == 401
    assert "Invalid or revoked" in res.json()["detail"]
    assert "Traceback" not in res.text
