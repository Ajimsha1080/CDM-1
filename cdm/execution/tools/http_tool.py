"""Layer 4: Sandboxed HTTP Client Tool with Domain Allowlist for CDM-1.
Guarantees network isolation by strictly blocking calls to unapproved external domains.
"""

from typing import Dict, Any, Optional, Set
import urllib.parse
import requests
from cdm.execution.tools.base import BaseTool


class HttpCallTool(BaseTool):
    """Sandboxed HTTP call tool with domain allowlist."""

    def __init__(self, allowed_domains: Optional[Set[str]] = None, timeout_seconds: float = 5.0):
        super().__init__(name="http_request", description="Performs an HTTP request to an allowlisted endpoint.")
        self.allowed_domains = allowed_domains or {"localhost", "127.0.0.1", "mock-service.local"}
        self.timeout_seconds = min(timeout_seconds, 15.0)

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        url = args.get("url")
        if not url or not isinstance(url, str):
            raise ValueError("Parameter 'url' is required.")

        parsed = urllib.parse.urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise ValueError(f"Invalid URL structure: {url}")

        if hostname not in self.allowed_domains:
            raise PermissionError(
                f"Domain '{hostname}' is not in the approved tenant allowlist. "
                f"Allowed domains: {sorted(list(self.allowed_domains))}"
            )

        method = args.get("method", "GET").upper()
        headers = args.get("headers", {})
        body = args.get("body")

        resp = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=body if method in ("POST", "PUT", "PATCH") and body else None,
            timeout=self.timeout_seconds,
        )

        try:
            resp_body = resp.json()
        except Exception:
            resp_body = resp.text[:4096]

        return {
            "status_code": resp.status_code,
            "url": url,
            "response": resp_body,
        }
