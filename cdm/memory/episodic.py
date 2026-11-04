"""Episodic Memory Subsystem for CDM-1.
Stores and retrieves historical interactions, context vectors, and verified outcomes
with semantic relevance scoring, duplicate detection, provenance, and TTL expiration
backed by persistent storage.
"""

from typing import Dict, List, Optional, Any
import time
from pydantic import BaseModel, Field
from cdm.memory.storage import BaseMemoryStorage, SQLiteMemoryStorage


class MemoryRecord(BaseModel):
    record_id: str
    tenant_id: str = "default_tenant"
    timestamp: float = Field(default_factory=time.time)
    context_text: str
    features: Dict[str, Any] = Field(default_factory=dict)
    decision_made: Dict[str, Any]
    verified_outcome: Optional[Dict[str, Any]] = None
    provenance: str = "production_telemetry"
    vector: Optional[List[float]] = None
    ttl_seconds: Optional[int] = 86400 * 90  # 90 days default expiration


class EpisodicMemory:
    """Persistent storage-backed episodic memory with tenant isolation and cosine retrieval."""

    def __init__(self, storage: Optional[BaseMemoryStorage] = None):
        self.storage = storage or SQLiteMemoryStorage()

    def insert(self, record: MemoryRecord) -> None:
        """Insert verified interaction record into persistent memory."""
        data = {
            "context_text": record.context_text,
            "features": record.features,
            "decision_made": record.decision_made,
            "provenance": record.provenance,
        }
        self.storage.insert_episodic(
            record_id=record.record_id,
            tenant_id=record.tenant_id,
            data=data,
            vector=record.vector,
            ttl=record.ttl_seconds,
        )

    def retrieve_similar(
        self,
        query_vector: List[float],
        tenant_id: str = "default_tenant",
        top_k: int = 3,
        min_relevance: float = 0.5,
        prohibited_actions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Cosine similarity vector retrieval over episodic memory for a specific tenant.

        Guarantees memory retrieval does not surface recommendations that violate policy constraints.
        """
        raw_matches = self.storage.query_similar_episodic(
            tenant_id=tenant_id,
            query_vector=query_vector,
            top_k=top_k * 2,
            min_sim=min_relevance,
        )

        prohibited_set = set(prohibited_actions or [])
        guarded_results = []
        for m in raw_matches:
            dec = m.get("decision_made", {})
            action = dec.get("selected_candidate") or dec.get("decision")
            # If the past decision is currently prohibited by policy, flag or skip
            if action in prohibited_set:
                m["policy_compliant"] = False
                continue
            m["policy_compliant"] = True
            guarded_results.append(m)
            if len(guarded_results) >= top_k:
                break

        return guarded_results

    def update_verified_outcome(self, record_id: str, outcome: Dict[str, Any]) -> bool:
        """Update a historical record with verified ground-truth outcome once resolved."""
        return self.storage.update_verified_outcome(record_id, outcome)
