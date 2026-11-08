"""Layer 4: Sandboxed HTTP Client Tool with SSRF Defense & Domain Allowlist for CDM-1.
Guarantees network isolation by:
- Requiring per-tenant approved domain allowlists (empty/missing means DENY ALL by default).
- Zero localhost/127.0.0.1 default allowlist.
- Full DNS resolution and blocking of private, loopback, link-local, and cloud metadata (169.254.169.254) addresses.
- Detection of decimal, hex, and octal encoded IP addresses.
- Re-validation of each redirect hop (SSRF redirect protection).
- Strict response size limits (preventing memory exhaustion) and execution timeouts.
"""

from typing import Dict, Any, Optional, Set
import ipaddress
import socket
import urllib.parse
import requests
from cdm.execution.tools.base import BaseTool


def is_blocked_ip_address(ip_str: str) -> bool:
    """Check if an IP string is private, loopback, link-local, multicast, metadata, or reserved."""
    clean = ip_str.strip("[]").strip()
    try:
        ip = ipaddress.ip_address(clean)
    except ValueError:
        # Check if integer / hex encoded IP (e.g. 2130706433 or 0x7f000001)
        if clean.isdigit() or (clean.startswith(("0x", "0X")) and all(c in "0123456789abcdefABCDEF" for c in clean[2:])):
            try:
                ip = ipaddress.ip_address(int(clean, 0))
            except (ValueError, OverflowError):
                return False
        else:
            return False

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return True

    # Cloud metadata IP addresses (AWS, GCP, Azure, OpenStack)
    if str(ip) in ("169.254.169.254", "169.254.169.253", "fd00:ec2::254"):
        return True

    return False


def validate_hostname_and_dns(hostname: str) -> None:
    """Resolve hostname and verify that no resolved IP is private or metadata."""
    if not hostname:
        raise ValueError("Empty hostname.")

    clean_host = hostname.strip("[]").strip()
    if is_blocked_ip_address(clean_host):
        raise PermissionError(f"Target host '{hostname}' is a prohibited private, loopback, or metadata address.")

    # Resolve all DNS A and AAAA records
    try:
        addr_info = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise PermissionError(f"DNS resolution failed for host '{hostname}': {e}")

    if not addr_info:
        raise PermissionError(f"No IP addresses resolved for host '{hostname}'.")

    for family, socktype, proto, canonname, sockaddr in addr_info:
        resolved_ip = sockaddr[0]
        if is_blocked_ip_address(resolved_ip):
            raise PermissionError(
                f"Host '{hostname}' resolved to prohibited address '{resolved_ip}' (SSRF violation)."
            )


class HttpCallTool(BaseTool):
    """Sandboxed HTTP call tool with per-tenant allowlists and strict SSRF defenses."""

    def __init__(
        self,
        tenant_allowlists: Optional[Dict[str, Set[str]]] = None,
        allowed_domains: Optional[Set[str]] = None,
        timeout_seconds: float = 5.0,
        max_response_bytes: int = 1048576,  # 1 MB max response size
        max_redirects: int = 5,
    ):
        super().__init__(name="http_request", description="Performs an HTTP request to an allowlisted endpoint.")
        self.tenant_allowlists: Dict[str, Set[str]] = tenant_allowlists or {}
        # Empty default allowlist = deny all
        self.default_allowed_domains: Set[str] = allowed_domains or set()
        self.timeout_seconds = min(float(timeout_seconds), 15.0)
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects

    def _get_tenant_allowlist(self, tenant_id: str) -> Set[str]:
        """Retrieve the configured allowed domains for a given tenant."""
        if tenant_id in self.tenant_allowlists:
            return self.tenant_allowlists[tenant_id]
        return self.default_allowed_domains

    def _validate_target_url(self, url: str, tenant_id: str) -> urllib.parse.ParseResult:
        """Validate URL scheme, tenant domain allowlist, and resolve DNS against SSRF."""
        if not url or not isinstance(url, str):
            raise ValueError("Parameter 'url' is required and must be a string.")

        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError(f"Prohibited URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

        hostname = (parsed.hostname or "").lower()
        if not hostname:
            raise ValueError(f"Invalid URL structure: {url}")

        allowed = self._get_tenant_allowlist(tenant_id)
        if not allowed or hostname not in allowed:
            raise PermissionError(
                f"Domain '{hostname}' is not in the approved allowlist for tenant '{tenant_id}'. "
                f"Approved domains: {sorted(list(allowed))}"
            )

        validate_hostname_and_dns(hostname)
        return parsed

    def _run(self, args: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        url = args.get("url")
        self._validate_target_url(url, tenant_id)

        method = args.get("method", "GET").upper()
        headers = args.get("headers", {})
        body = args.get("body")

        current_url = url
        redirect_count = 0

        session = requests.Session()

        while True:
            resp = session.request(
                method=method,
                url=current_url,
                headers=headers,
                json=body if method in ("POST", "PUT", "PATCH") and body else None,
                timeout=self.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )

            # Check if this is a redirect hop
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("Location")
                if not location:
                    break

                redirect_count += 1
                if redirect_count > self.max_redirects:
                    raise PermissionError(f"Exceeded maximum allowed redirects ({self.max_redirects}).")

                # Form new URL and re-validate against tenant allowlist and SSRF
                next_url = urllib.parse.urljoin(current_url, location)
                self._validate_target_url(next_url, tenant_id)
                current_url = next_url
                # On 303 or 302/301, typical HTTP behavior switches method to GET
                if resp.status_code in (301, 302, 303):
                    method = "GET"
                    body = None
                continue

            # Read bounded response
            content_bytes = bytearray()
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    content_bytes.extend(chunk)
                    if len(content_bytes) > self.max_response_bytes:
                        raise ValueError(f"Response size exceeded limit of {self.max_response_bytes} bytes.")

            text_content = content_bytes.decode("utf-8", errors="replace")
            try:
                import json
                resp_body = json.loads(text_content)
            except Exception:
                resp_body = text_content[:4096]

            return {
                "status_code": resp.status_code,
                "url": current_url,
                "response": resp_body,
                "redirects": redirect_count,
            }
