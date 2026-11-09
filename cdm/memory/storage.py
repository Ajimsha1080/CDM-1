"""Storage abstraction for CDM-1 Cognitive Memory.
Provides SQLite file-backed persistent storage by default and PostgreSQL/pgvector support,
enforcing tenant isolation, TTL expiration, and vector similarity retrieval.
"""

from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import json
import time
import math
import os


class BaseMemoryStorage:
    """Abstract base class for persistent cognitive memory storage."""

    def insert_episodic(self, record_id: str, tenant_id: str, data: Dict[str, Any], vector: Optional[List[float]], ttl: Optional[int]) -> None:
        raise NotImplementedError

    def query_similar_episodic(self, tenant_id: str, query_vector: List[float], top_k: int = 5, min_sim: float = 0.5) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def update_verified_outcome(self, record_id: str, outcome: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        raise NotImplementedError

    def list_episodic(self, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT record_id, tenant_id, created_at, payload_json, verified_outcome_json FROM episodic_memory WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
        )
        results = []
        for r in rows:
            payload = json.loads(r["payload_json"])
            payload["record_id"] = r["record_id"]
            if r["verified_outcome_json"]:
                payload["verified_outcome"] = json.loads(r["verified_outcome_json"])
            results.append(payload)
        return results

    def insert_rule(self, rule_id: str, tenant_id: str, domain: str, rule_data: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_rules_for_domain(self, tenant_id: str, domain: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def insert_procedure(self, proc_id: str, tenant_id: str, domain: str, proc_data: Dict[str, Any]) -> None:
        raise NotImplementedError

    def get_procedure(self, proc_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class SQLiteMemoryStorage(BaseMemoryStorage):
    """File-backed SQLite storage providing out-of-the-box persistent memory with vector cosine similarity."""

    def __init__(self, db_path: str = "experiments/cognitive_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _execute(self, query: str, params: Tuple = ()) -> List[sqlite3.Row]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()
        finally:
            conn.close()

    def _execute_commit(self, query: str, params: Tuple = ()) -> int:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                cursor = conn.execute(query, params)
                return cursor.rowcount
        finally:
            conn.close()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS episodic_memory (
                        record_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        ttl_seconds INTEGER,
                        vector_json TEXT,
                        payload_json TEXT NOT NULL,
                        verified_outcome_json TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_episodic_tenant ON episodic_memory (tenant_id)")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS semantic_rules (
                        rule_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        rule_json TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_tenant_domain ON semantic_rules (tenant_id, domain)")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS procedural_memory (
                        procedure_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        domain TEXT NOT NULL,
                        procedure_json TEXT NOT NULL
                    )
                """)
        finally:
            conn.close()

    def insert_episodic(
        self,
        record_id: str,
        tenant_id: str,
        data: Dict[str, Any],
        vector: Optional[List[float]],
        ttl: Optional[int],
    ) -> None:
        now = time.time()
        vec_str = json.dumps(vector) if vector else None
        payload_str = json.dumps(data)
        outcome_str = json.dumps(data["verified_outcome"]) if data.get("verified_outcome") else None
        self._execute_commit(
            """
            INSERT OR REPLACE INTO episodic_memory 
            (record_id, tenant_id, created_at, ttl_seconds, vector_json, payload_json, verified_outcome_json)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (record_id, tenant_id, now, ttl, vec_str, payload_str, outcome_str),
        )

    def query_similar_episodic(
        self,
        tenant_id: str,
        query_vector: List[float],
        top_k: int = 5,
        min_sim: float = 0.5,
    ) -> List[Dict[str, Any]]:
        now = time.time()
        results = []

        rows = self._execute("SELECT * FROM episodic_memory WHERE tenant_id = ?", (tenant_id,))

        for row in rows:
            ttl = row["ttl_seconds"]
            if ttl and (now - row["created_at"] > ttl):
                continue

            vec_json = row["vector_json"]
            if not vec_json:
                continue

            rec_vec = json.loads(vec_json)
            if len(rec_vec) != len(query_vector):
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vector, rec_vec))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in rec_vec))
            if norm_a == 0 or norm_b == 0:
                continue

            sim = dot / (norm_a * norm_b)
            if sim >= min_sim:
                payload = json.loads(row["payload_json"])
                payload["record_id"] = row["record_id"]
                payload["similarity"] = round(float(sim), 4)
                if row["verified_outcome_json"]:
                    payload["verified_outcome"] = json.loads(row["verified_outcome_json"])
                results.append((sim, payload))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in results[:top_k]]

    def update_verified_outcome(self, record_id: str, outcome: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        outcome_str = json.dumps(outcome)
        if tenant_id:
            rowcount = self._execute_commit(
                "UPDATE episodic_memory SET verified_outcome_json = ? WHERE record_id = ? AND tenant_id = ?",
                (outcome_str, record_id, tenant_id),
            )
        else:
            rowcount = self._execute_commit(
                "UPDATE episodic_memory SET verified_outcome_json = ? WHERE record_id = ?",
                (outcome_str, record_id),
            )
        return rowcount > 0

    def list_episodic(self, tenant_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT record_id, tenant_id, created_at, payload_json, verified_outcome_json FROM episodic_memory WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
            (tenant_id, limit),
        )
        results = []
        for r in rows:
            payload = json.loads(r["payload_json"])
            payload["record_id"] = r["record_id"]
            if r["verified_outcome_json"]:
                payload["verified_outcome"] = json.loads(r["verified_outcome_json"])
            results.append(payload)
        return results

    def insert_rule(self, rule_id: str, tenant_id: str, domain: str, rule_data: Dict[str, Any]) -> None:
        self._execute_commit(
            "INSERT OR REPLACE INTO semantic_rules (rule_id, tenant_id, domain, rule_json) VALUES (?, ?, ?, ?)",
            (rule_id, tenant_id, domain, json.dumps(rule_data)),
        )

    def get_rules_for_domain(self, tenant_id: str, domain: str) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT rule_json FROM semantic_rules WHERE tenant_id = ? AND (domain = ? OR domain = 'global')",
            (tenant_id, domain),
        )
        return [json.loads(r["rule_json"]) for r in rows]

    def insert_procedure(self, proc_id: str, tenant_id: str, domain: str, proc_data: Dict[str, Any]) -> None:
        self._execute_commit(
            "INSERT OR REPLACE INTO procedural_memory (procedure_id, tenant_id, domain, procedure_json) VALUES (?, ?, ?, ?)",
            (proc_id, tenant_id, domain, json.dumps(proc_data)),
        )

    def get_procedure(self, proc_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        rows = self._execute(
            "SELECT procedure_json FROM procedural_memory WHERE procedure_id = ? AND tenant_id = ?",
            (proc_id, tenant_id),
        )
        return json.loads(rows[0]["procedure_json"]) if rows else None

    def get_procedures_for_domain(self, tenant_id: str, domain: str) -> List[Dict[str, Any]]:
        rows = self._execute(
            "SELECT procedure_json FROM procedural_memory WHERE tenant_id = ? AND (domain = ? OR domain = 'global')",
            (tenant_id, domain),
        )
        return [json.loads(r["procedure_json"]) for r in rows]
