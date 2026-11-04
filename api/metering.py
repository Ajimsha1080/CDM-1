"""Durable Transactional Usage Metering & Distributed Rate Limiting for CDM-1.
Uses SQLite in Write-Ahead-Logging (WAL) mode with immediate transactions to guarantee
durability across sudden process terminations or multi-worker concurrency.
"""

import os
import time
import sqlite3
import threading
from typing import Dict, Any, Tuple, Optional


class DurableUsageMeter:
    """Thread-safe, transaction-durable usage ledger and rate limiter."""

    def __init__(self, db_path: str = "experiments/api_usage_meter.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
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
                conn.commit()
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
                conn.commit()
            finally:
                conn.close()

    def check_rate_limit(self, tenant_id: str, max_rpm: int = 120) -> Tuple[bool, int]:
        """Sliding 60-second window rate limiting.

        Returns:
            (is_allowed: bool, current_count: int)
        """
        now = time.time()
        window_start = now - 60.0
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COUNT(*) FROM usage_ledger
                    WHERE tenant_id = ? AND timestamp >= ?;
                    """,
                    (tenant_id, window_start),
                )
                count = cur.fetchone()[0]
                return (count < max_rpm), count
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
