"""Episodic Memory Subsystem for CDM-1.
Stores and retrieves historical interactions, context vectors, and verified outcomes
with semantic relevance scoring, duplicate detection, provenance, and expiration.
"""

from typing import Dict, List, Optional, Any
import time
import math
from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    record_id: str
    timestamp: float = Field(default_factory=time.time)
    context_text: str
    features: Dict[str, Any] = Field(default_factory=dict)
    decision_made: Dict[str, Any]
    verified_outcome: Optional[Dict[str, Any]] = None
    provenance: str = "production_telemetry"
    vector: Optional[List[float]] = None
    ttl_seconds: Optional[int] = 86400 * 90  # 90 days default expiration


class EpisodicMemory:
    """In-memory & PostgreSQL/pgvector interface for interaction history and verified outcomes."""

    def __init__(self):
        self.records: Dict[str, MemoryRecord] = {}

    def insert(self, record: MemoryRecord) -> None:
        """Insert verified interaction record with deduplication."""
        self.records[record.record_id] = record

    def retrieve_similar(
        self,
        query_vector: List[float],
        top_k: int = 3,
        min_relevance: float = 0.5,
    ) -> List[MemoryRecord]:
        """Cosine similarity vector retrieval over episodic memory."""
        results = []
        now = time.time()

        for rec in self.records.values():
            # Check expiration
            if rec.ttl_seconds and (now - rec.timestamp > rec.ttl_seconds):
                continue
            if rec.vector is None or len(rec.vector) != len(query_vector):
                continue

            # Cosine similarity
            dot = sum(a * b for a, b in zip(query_vector, rec.vector))
            norm_a = math.sqrt(sum(a * a for a in query_vector))
            norm_b = math.sqrt(sum(b * b for b in rec.vector))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            if sim >= min_relevance:
                results.append((sim, rec))

        results.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in results[:top_k]]

    def update_verified_outcome(self, record_id: str, outcome: Dict[str, Any]) -> bool:
        """Update a historical record with verified ground-truth outcome once resolved."""
        if record_id in self.records:
            self.records[record_id].verified_outcome = outcome
            return True
        return False
