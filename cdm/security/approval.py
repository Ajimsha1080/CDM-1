"""Cryptographically verifiable, expiring, action-bound approvals for CDM-1.
Replaces legacy string prefix authorization with cryptographic HMAC-SHA256 tokens
strictly binding approval to tenant, user, plan, action, and expiration timestamp.
"""

from typing import Dict, Any, Optional, Tuple, Union
import time
import os
import json
import base64
import hmac
import hashlib
import secrets
from pydantic import BaseModel, Field


def get_hmac_secret() -> bytes:
    """Returns HMAC secret from environment or deterministic fallback."""
    secret = os.environ.get("CDM_HMAC_SECRET", "cdm-secure-audit-secret-key-32b-prod")
    return secret.encode("utf-8")


class ActionApproval(BaseModel):
    """Cryptographically verifiable, expiring, action-bound approval token."""
    tenant_id: str
    user_id: str
    plan_id: str
    action_name: str
    expires_at: float
    issued_at: float = Field(default_factory=time.time)
    nonce: str
    signature: str

    def serialize(self) -> str:
        """Serializes token into base64url-encoded string."""
        data = self.model_dump()
        json_bytes = json.dumps(data, sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(json_bytes).decode("ascii")

    @classmethod
    def deserialize(cls, token_str: str) -> "ActionApproval":
        """Deserializes base64url token string into ActionApproval."""
        try:
            json_bytes = base64.urlsafe_b64decode(token_str.encode("ascii"))
            data = json.loads(json_bytes.decode("utf-8"))
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid approval token encoding: {e}")


def compute_approval_signature(
    tenant_id: str,
    user_id: str,
    plan_id: str,
    action_name: str,
    issued_at: float,
    expires_at: float,
    nonce: str,
    secret_key: Optional[bytes] = None,
) -> str:
    """Computes HMAC-SHA256 signature binding approval to tenant, user, plan, action, and expiry."""
    key = secret_key or get_hmac_secret()
    payload = f"{tenant_id}:{user_id}:{plan_id}:{action_name}:{issued_at:.3f}:{expires_at:.3f}:{nonce}"
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def generate_action_approval(
    tenant_id: str,
    user_id: str,
    plan_id: str,
    action_name: str,
    ttl_seconds: float = 300.0,
    secret_key: Optional[bytes] = None,
) -> str:
    """Generates and cryptographically signs an expiring action-bound approval token."""
    now = time.time()
    expires_at = now + ttl_seconds
    nonce = secrets.token_hex(8)
    sig = compute_approval_signature(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id=plan_id,
        action_name=action_name,
        issued_at=now,
        expires_at=expires_at,
        nonce=nonce,
        secret_key=secret_key,
    )
    token = ActionApproval(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id=plan_id,
        action_name=action_name,
        expires_at=expires_at,
        issued_at=now,
        nonce=nonce,
        signature=sig,
    )
    return token.serialize()


def verify_action_approval(
    token_str: str,
    expected_tenant_id: Optional[str] = None,
    expected_plan_id: Optional[str] = None,
    expected_action_name: Optional[str] = None,
    current_time: Optional[float] = None,
    secret_key: Optional[bytes] = None,
) -> Tuple[bool, Optional[str], Optional[ActionApproval]]:
    """Validates an action approval token:
    1. Deserializes and decodes token.
    2. Validates HMAC-SHA256 signature against secret.
    3. Enforces expiration window (now <= expires_at).
    4. Validates matching tenant, plan, and action bounds.
    """
    if not token_str:
        return False, "Missing approval token.", None

    try:
        token = ActionApproval.deserialize(token_str)
    except Exception as e:
        return False, f"Malformed approval token: {e}", None

    now = current_time if current_time is not None else time.time()

    # 1. Enforce expiration
    if now > token.expires_at:
        return False, f"Approval token expired at {token.expires_at:.1f} (current time: {now:.1f}).", token

    # 2. Verify cryptographic signature
    expected_sig = compute_approval_signature(
        tenant_id=token.tenant_id,
        user_id=token.user_id,
        plan_id=token.plan_id,
        action_name=token.action_name,
        issued_at=token.issued_at,
        expires_at=token.expires_at,
        nonce=token.nonce,
        secret_key=secret_key,
    )
    if not hmac.compare_digest(token.signature, expected_sig):
        return False, "Invalid cryptographic approval signature: signature mismatch or forgery detected.", token

    # 3. Enforce tenant binding
    if expected_tenant_id and token.tenant_id != expected_tenant_id:
        return False, f"Tenant mismatch: token bound to '{token.tenant_id}', expected '{expected_tenant_id}'.", token

    # 4. Enforce plan binding
    if expected_plan_id and token.plan_id != expected_plan_id:
        return False, f"Plan mismatch: token bound to plan '{token.plan_id}', expected '{expected_plan_id}'.", token

    # 5. Enforce action binding
    if expected_action_name and token.action_name != expected_action_name:
        return False, f"Action mismatch: token bound to action '{token.action_name}', expected '{expected_action_name}'.", token

    return True, None, token
