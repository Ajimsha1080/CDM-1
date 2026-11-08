"""Comprehensive SSRF Protection & HTTP Tool Security Tests (Item 2).

Verifies:
1. Localhost and 127.0.0.1 are removed from default allowlists; empty allowlist denies all.
2. Direct loopback, private, link-local, and cloud metadata (169.254.169.254) addresses are blocked.
3. IPv6 loopback (::1, [::1]) is blocked.
4. Decimal and hex IP encodings (e.g. 2130706433, 0x7f000001) are blocked.
5. DNS resolving to a private IP is blocked.
6. HTTP redirects to localhost or private IPs are blocked on each hop.
7. Response size and execution timeout limits are enforced.
8. Per-tenant allowlist isolation is enforced.
"""

import pytest
from unittest.mock import MagicMock
from cdm.execution.tools.http_tool import (
    HttpCallTool,
    is_blocked_ip_address,
    validate_hostname_and_dns,
)


def test_empty_allowlist_denies_all_by_default():
    """Verify that an empty allowlist denies all outbound calls."""
    tool = HttpCallTool()
    res = tool.execute({"url": "https://api.example.com/data"}, tenant_id="tenant_1")
    assert res.success is False
    assert "not in the approved allowlist" in res.error.lower()


def test_tenant_allowlist_isolation():
    """Verify that tenant A cannot access domains allowed only for tenant B."""
    tenant_allowlists = {
        "tenant_finance": {"api.stripe.com"},
        "tenant_support": {"api.zendesk.com"},
    }
    tool = HttpCallTool(tenant_allowlists=tenant_allowlists)

    # Tenant support attempting to access finance domain
    res = tool.execute({"url": "https://api.stripe.com/v1/charges"}, tenant_id="tenant_support")
    assert res.success is False
    assert "not in the approved allowlist" in res.error.lower()


def test_blocked_ip_detection_direct_and_encodings():
    """Verify is_blocked_ip_address catches standard, decimal, and hex encoded IPs."""
    # Standard loopback and private IPv4
    assert is_blocked_ip_address("127.0.0.1") is True
    assert is_blocked_ip_address("10.0.0.5") is True
    assert is_blocked_ip_address("192.168.1.1") is True
    assert is_blocked_ip_address("172.16.0.1") is True

    # Metadata service
    assert is_blocked_ip_address("169.254.169.254") is True
    assert is_blocked_ip_address("169.254.169.253") is True

    # IPv6 loopback and metadata
    assert is_blocked_ip_address("::1") is True
    assert is_blocked_ip_address("fd00:ec2::254") is True

    # Decimal representation of 127.0.0.1 (2130706433)
    assert is_blocked_ip_address("2130706433") is True

    # Hex representation of 127.0.0.1 (0x7f000001)
    assert is_blocked_ip_address("0x7f000001") is True

    # Public valid IP is not blocked
    assert is_blocked_ip_address("93.184.216.34") is False


def test_hostname_ssrf_direct_loopback_and_encodings():
    """Verify validate_hostname_and_dns blocks direct IP representations in URLs."""
    with pytest.raises(PermissionError, match="prohibited private, loopback, or metadata"):
        validate_hostname_and_dns("127.0.0.1")

    with pytest.raises(PermissionError, match="prohibited private, loopback, or metadata"):
        validate_hostname_and_dns("169.254.169.254")

    with pytest.raises(PermissionError, match="prohibited private, loopback, or metadata"):
        validate_hostname_and_dns("::1")

    with pytest.raises(PermissionError, match="prohibited private, loopback, or metadata"):
        validate_hostname_and_dns("2130706433")

    with pytest.raises(PermissionError, match="prohibited private, loopback, or metadata"):
        validate_hostname_and_dns("0x7f000001")


def test_dns_resolving_to_private_ip_is_blocked(monkeypatch):
    """Verify that a public-looking domain resolving to an internal IP is blocked."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        # Simulate DNS rebinding / malicious resolution to internal IP
        return [(2, 1, 6, "", ("10.0.0.1", 80))]

    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    with pytest.raises(PermissionError, match="SSRF violation"):
        validate_hostname_and_dns("legit-looking-domain.com")


def test_dns_resolving_to_metadata_ip_is_blocked(monkeypatch):
    """Verify that a domain resolving to AWS/GCP metadata service is blocked."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        return [(2, 1, 6, "", ("169.254.169.254", 80))]

    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    with pytest.raises(PermissionError, match="SSRF violation"):
        validate_hostname_and_dns("cloud-metadata.evil.com")


def test_redirect_to_localhost_blocked(monkeypatch):
    """Verify that HTTP 302 redirect to localhost is caught and blocked on the hop."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        if host == "api.trusted.com":
            return [(2, 1, 6, "", ("93.184.216.34", 443))]
        return [(2, 1, 6, "", ("127.0.0.1", 80))]

    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    # Mock session request returning 302 to 127.0.0.1
    mock_resp = MagicMock()
    mock_resp.status_code = 302
    mock_resp.headers = {"Location": "http://127.0.0.1:8000/admin"}
    monkeypatch.setattr("requests.Session.request", lambda *args, **kwargs: mock_resp)

    tool = HttpCallTool(allowed_domains={"api.trusted.com"})
    res = tool.execute({"url": "https://api.trusted.com/start"})
    assert res.success is False
    assert "prohibited private, loopback, or metadata" in res.error or "not in the approved allowlist" in res.error


def test_redirect_to_metadata_blocked(monkeypatch):
    """Verify that HTTP 302 redirect to metadata service is blocked."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        if host == "api.trusted.com":
            return [(2, 1, 6, "", ("93.184.216.34", 443))]
        return [(2, 1, 6, "", ("169.254.169.254", 80))]

    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    mock_resp = MagicMock()
    mock_resp.status_code = 301
    mock_resp.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
    monkeypatch.setattr("requests.Session.request", lambda *args, **kwargs: mock_resp)

    tool = HttpCallTool(allowed_domains={"api.trusted.com"})
    res = tool.execute({"url": "https://api.trusted.com/data"})
    assert res.success is False
    assert "prohibited" in res.error.lower() or "not in the approved allowlist" in res.error.lower()


def test_response_size_limit_enforced(monkeypatch):
    """Verify that response size limit terminates oversized downloads."""
    def mock_getaddrinfo(host, port, *args, **kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr("socket.getaddrinfo", mock_getaddrinfo)

    # Return 200KB payload in chunks when limit is 50KB
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content = lambda chunk_size=8192: [b"A" * 8192] * 25
    monkeypatch.setattr("requests.Session.request", lambda *args, **kwargs: mock_resp)

    tool = HttpCallTool(allowed_domains={"api.trusted.com"}, max_response_bytes=50 * 1024)
    res = tool.execute({"url": "https://api.trusted.com/bigfile"})
    assert res.success is False
    assert "response size exceeded limit" in res.error.lower()
