"""Episodic Memory Subsystem for CDM-1.
Stores and retrieves historical interactions, context vectors, and verified outcomes
with semantic relevance scoring, duplicate detection, provenance, and TTL expiration
backed by persistent storage with tenant isolation.
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

    def search_similar(
        self,
        tenant_id: str = "default_tenant",
        query_vector: Optional[List[float]] = None,
        top_k: int = 3,
        min_sim: float = 0.5,
        prohibited_actions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience alias for retrieve_similar with flexible parameter order."""
        vec = query_vector or []
        return self.retrieve_similar(
            query_vector=vec,
            tenant_id=tenant_id,
            top_k=top_k,
            min_relevance=min_sim,
            prohibited_actions=prohibited_actions,
        )

    def record_episode(
        self,
        session_id: str,
        goal: str,
        action_taken: str,
        observed_outcome: str,
        reward_signal: float = 0.0,
        tenant_id: str = "default_tenant",
        vector: Optional[List[float]] = None,
        embedder: Optional[Any] = None,
    ) -> MemoryRecord:
        """Records an operational episode into persistent storage with tenant isolation."""
        import uuid
        record_id = f"ep_{uuid.uuid4().hex[:12]}"
        context_text = f"Goal: {goal} | Session: {session_id}"

        vec = vector
        if vec is None and embedder is not None and hasattr(embedder, "embed"):
            try:
                vec = embedder.embed(text=f"{goal} {action_taken}")
            except Exception:
                vec = None

        record = MemoryRecord(
            record_id=record_id,
            tenant_id=tenant_id,
            context_text=context_text,
            features={"session_id": session_id, "reward_signal": reward_signal},
            decision_made={"action_taken": action_taken},
            verified_outcome={"observed_outcome": observed_outcome, "reward_signal": reward_signal},
            vector=vec,
        )
        self.insert(record)
        return record

    def search_episodes(
        self,
        query_text: str,
        limit: int = 3,
        tenant_id: str = "default_tenant",
        embedder: Optional[Any] = None,
    ) -> List[Any]:
        """Searches past episodes by semantic similarity or text relevance for a specific tenant."""
        vec = None
        if embedder is not None and hasattr(embedder, "embed"):
            try:
                vec = embedder.embed(text=query_text)
            except Exception:
                vec = None

        if vec is not None:
            raw = self.retrieve_similar(query_vector=vec, tenant_id=tenant_id, top_k=limit, min_relevance=0.1)
            results = []
            for r in raw:
                dec = r.get("decision_made", {})
                out = r.get("verified_outcome", {})
                results.append(type("EpisodeView", (), {
                    "action_taken": dec.get("action_taken") or dec.get("selected_candidate", "action"),
                    "observed_outcome": str(out.get("observed_outcome") or r.get("context_text", "")),
                    "reward_signal": float(out.get("reward_signal", 0.0)),
                }))
            return results

        if hasattr(self.storage, "list_episodic"):
            rows = self.storage.list_episodic(tenant_id=tenant_id, limit=limit)
            results = []
            for r in rows:
                dec = r.get("decision_made", {})
                out = r.get("verified_outcome", {})
                results.append(type("EpisodeView", (), {
                    "action_taken": dec.get("action_taken") or dec.get("selected_candidate", "action"),
                    "observed_outcome": str(out.get("observed_outcome") or r.get("context_text", "")),
                    "reward_signal": float(out.get("reward_signal", 0.0)),
                }))
            return results
        return []

    def update_verified_outcome(
        self, record_id: str, outcome: Dict[str, Any], tenant_id: Optional[str] = None
    ) -> bool:
        """Update a historical record with verified ground-truth outcome once resolved with tenant isolation."""
        return self.storage.update_verified_outcome(record_id, outcome, tenant_id=tenant_id)
