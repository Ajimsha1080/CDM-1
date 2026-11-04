"""Durable Transactional Usage Metering & Distributed Rate Limiting for CDM-1.
Uses SQLite in Write-Ahead-Logging (WAL) mode with immediate transactions to guarantee
durability across sudden process terminations or multi-worker concurrency.
Supports optional Redis backend for multi-instance distributed deployments.
"""

import os
import time
import secrets
import sqlite3
import threading
from typing import Dict, Any, Tuple, Optional

try:
    import redis
except ImportError:
    redis = None


class DurableUsageMeter:
    """Thread-safe, transaction-durable usage ledger and atomic rate limiter."""

    def __init__(self, db_path: str = "experiments/api_usage_meter.sqlite", redis_url: Optional[str] = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

        # Optional Redis connection for multi-instance distributed deployments
        self.redis_url = redis_url or os.environ.get("REDIS_URL") or os.environ.get("CDM_REDIS_URL")
        self._redis_client = None
        if self.redis_url and redis:
            try:
                self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
                self._redis_client.ping()
            except Exception:
                self._redis_client = None

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        task TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        latency_ms REAL NOT NULL,
                        status_code INTEGER NOT NULL DEFAULT 200
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_tenant_time ON usage_ledger(tenant_id, timestamp);")

                # Dedicated atomic rate limit reservation table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rate_limit_tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tenant_id TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rate_tokens ON rate_limit_tokens(tenant_id, timestamp);")
            finally:
                conn.close()

    def record_request(
        self,
        tenant_id: str,
        endpoint: str,
        task: str,
        latency_ms: float,
        status_code: int = 200,
    ) -> None:
        """Records an API invocation durably in the transactional ledger."""
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO usage_ledger (tenant_id, endpoint, task, timestamp, latency_ms, status_code)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (tenant_id, endpoint, task, now, latency_ms, status_code),
                )
            finally:
                conn.close()

    def check_rate_limit(self, tenant_id: str, max_rpm: int = 120) -> Tuple[bool, int]:
        """Atomic sliding 60-second window rate limiting across concurrent workers.

        Enforces atomic reservation under immediate transactions to prevent
        time-of-check to time-of-use (TOCTOU) concurrency bypass.

        Returns:
            (is_allowed: bool, current_count: int)
        """
        now = time.time()
        window_start = now - 60.0

        # Path A: Distributed Redis rate limiting (if configured)
        if self._redis_client:
            try:
                key = f"cdm:ratelimit:{tenant_id}"
                pipe = self._redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                res = pipe.execute()
                current_count = res[1]

                if current_count < max_rpm:
                    pipe2 = self._redis_client.pipeline()
                    pipe2.zadd(key, {f"{now}_{secrets.token_hex(4)}": now})
                    pipe2.expire(key, 70)
                    pipe2.execute()
                    return True, current_count + 1
                return False, current_count
            except Exception:
                # Fallback to transactional SQLite if Redis encounters transient error
                pass

        # Path B: Atomic transactional SQLite rate limiting
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                cur = conn.cursor()

                # 1. Prune expired rate limit tokens
                cur.execute("DELETE FROM rate_limit_tokens WHERE timestamp < ?;", (window_start,))

                # 2. Count active tokens in sliding 60-second window
                cur.execute(
                    "SELECT COUNT(*) FROM rate_limit_tokens WHERE tenant_id = ? AND timestamp >= ?;",
                    (tenant_id, window_start),
                )
                count = cur.fetchone()[0]

                if count < max_rpm:
                    # Atomically reserve slot
                    cur.execute(
                        "INSERT INTO rate_limit_tokens (tenant_id, timestamp) VALUES (?, ?);",
                        (tenant_id, now),
                    )
                    conn.execute("COMMIT;")
                    return True, count + 1
                else:
                    conn.execute("ROLLBACK;")
                    return False, count
            except Exception:
                try:
                    conn.execute("ROLLBACK;")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def get_tenant_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Retrieve aggregated usage statistics for a specific tenant."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COUNT(*), AVG(latency_ms), MAX(timestamp)
                    FROM usage_ledger
                    WHERE tenant_id = ?;
                    """,
                    (tenant_id,),
                )
                row = cur.fetchone()
                total = row[0] or 0
                avg_lat = round(row[1] or 0.0, 2)
                last_req = row[2]
                return {
                    "tenant_id": tenant_id,
                    "total_requests": total,
                    "average_latency_ms": avg_lat,
                    "last_request_timestamp": last_req,
                }
            finally:
                conn.close()
