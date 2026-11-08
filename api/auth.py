"""Persistent API-Key Management, Tenant Isolation, and Production Secret Security for CDM-1.
Stores key hashes in SQLite, prevents default credentials in production, and provides
constant-time cryptographic token verification.
"""

import os
import time
import secrets
import hashlib
import sqlite3
import threading
from typing import Dict, Any, Optional, Tuple


class PersistentApiKeyManager:
    """Manages persistent cryptographic API keys with tenant mapping."""

    def __init__(self, db_path: str = "experiments/api_keys.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_hash TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        tier TEXT NOT NULL DEFAULT 'standard',
                        rate_limit_rpm INTEGER NOT NULL DEFAULT 120,
                        created_at REAL NOT NULL,
                        revoked INTEGER NOT NULL DEFAULT 0,
                        label TEXT
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Computes SHA-256 digest of API key token."""
        return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()

    def register_key(
        self,
        raw_key: str,
        tenant_id: str,
        tier: str = "standard",
        rate_limit_rpm: int = 120,
        label: Optional[str] = None,
    ) -> str:
        """Stores a new API key hash with tenant metadata."""
        k_hash = self.hash_key(raw_key)
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO api_keys (key_hash, tenant_id, tier, rate_limit_rpm, created_at, revoked, label)
                    VALUES (?, ?, ?, ?, ?, 0, ?);
                    """,
                    (k_hash, tenant_id, tier, rate_limit_rpm, now, label),
                )
                conn.commit()
            finally:
                conn.close()
        return k_hash

    def generate_key(
        self,
        tenant_id: str,
        tier: str = "standard",
        rate_limit_rpm: int = 120,
        label: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generates a secure random 32-byte hexadecimal key."""
        raw_token = f"cdm_{tier}_{secrets.token_hex(24)}"
        k_hash = self.register_key(raw_token, tenant_id, tier, rate_limit_rpm, label)
        return raw_token, k_hash

    PROHIBITED_DEMO_KEYS = {
        "sk_live_cdm_demo_key",
        "sk_test_developer_key",
        "demo",
        "test",
        "default",
        "admin",
        "secret",
    }

    def validate_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Validates key token, checking for revocation and returning tenant record.
        Strictly requires registered credentials in the persistent database.
        """
        if not raw_key:
            return None

        clean_key = raw_key.strip()
        env = os.environ.get("CDM_ENV", "production").lower()

        # In production, immediately reject all demo / default credentials
        if env == "production" and clean_key in self.PROHIBITED_DEMO_KEYS:
            return None

        target_hash = self.hash_key(clean_key)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT key_hash, tenant_id, tier, rate_limit_rpm, revoked
                    FROM api_keys
                    WHERE key_hash = ?;
                    """,
                    (target_hash,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                db_hash, tenant_id, tier, rpm, revoked = row
                if revoked or not secrets.compare_digest(target_hash, db_hash):
                    return None
                return {
                    "tenant_id": tenant_id,
                    "tier": tier,
                    "rate_limit_rpm": rpm,
                }
            finally:
                conn.close()

    def revoke_key(self, raw_key: str) -> bool:
        """Revokes an active key."""
        target_hash = self.hash_key(raw_key.strip())
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE api_keys SET revoked = 1 WHERE key_hash = ?;",
                    (target_hash,),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def check_production_safety(self) -> None:
        """Enforces P1: Production requires registered credentials in the persistent store.
        Prohibits satisfying auth checks via arbitrary environment variables or demo keys.
        """
        env = os.environ.get("CDM_ENV", "production").lower()
        if env == "production":
            with self._lock:
                conn = self._get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT key_hash, tenant_id FROM api_keys WHERE revoked = 0;")
                    active_keys = cur.fetchall()
                finally:
                    conn.close()

            if not active_keys:
                raise RuntimeError(
                    "CRITICAL SECURITY FAILURE: Production environment (CDM_ENV=production) cannot start "
                    "without pre-registered credentials in the persistent store. Configure registered keys."
                )

            # Ensure no active production key matches any prohibited demo credential
            prohibited_hashes = {self.hash_key(k) for k in self.PROHIBITED_DEMO_KEYS}
            for k_hash, tenant_id in active_keys:
                if k_hash in prohibited_hashes:
                    raise RuntimeError(
                        f"CRITICAL SECURITY FAILURE: Prohibited demo credential registered for tenant '{tenant_id}' "
                        "in production database. Demo credentials must be revoked before production deployment."
                    )
