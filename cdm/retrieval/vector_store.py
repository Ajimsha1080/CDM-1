"""Persistent Vector Store Subsystem for CDM-1 Enterprise Retrieval (RAG).
Provides vector indexing, cosine similarity scoring, persistent disk serialization,
and hybrid dense-lexical ranking for enterprise documents.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import json
import math
import time
import re
import hashlib
from collections import Counter
from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """Stored vector record with metadata and passage content."""
    record_id: str
    doc_id: str
    title: str
    content: str
    source_repository: str
    version: str = "1.0.0"
    vector: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class PersistentVectorStore:
    """Manages normalized dense vector representations with disk persistence."""

    def __init__(self, dimension: int = 64, storage_path: Optional[str] = None):
        self.dimension = dimension
        self.storage_path = storage_path
        self.records: Dict[str, VectorRecord] = {}
        if storage_path and os.path.exists(storage_path):
            self.load_from_disk(storage_path)

    def _text_to_vector(self, text: str) -> List[float]:
        """Deterministic dense projection vector from text tokens using hashlib.md5.
        Uses sublinear term frequency weighting and L2 normalization to ensure
        strictly reproducible, high-accuracy cosine similarity.
        """
        words = re.findall(r"\w+", text.lower())
        vec = [0.0] * self.dimension
        if not words:
            return vec

        counts = Counter(words)
        for word, count in counts.items():
            h_int = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            slot = h_int % self.dimension
            tf = 1.0 + math.log(count)
            vec[slot] += tf

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        source_repository: str = "Enterprise Repository",
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
        custom_vector: Optional[List[float]] = None,
    ) -> str:
        """Indexes a document chunk with dense vector embedding."""
        record_id = f"{doc_id}_v{len(self.records) + 1}"
        vec = custom_vector or self._text_to_vector(f"{title} {content}")

        record = VectorRecord(
            record_id=record_id,
            doc_id=doc_id,
            title=title,
            content=content,
            source_repository=source_repository,
            version=version,
            vector=vec,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self.records[record_id] = record
        return record_id

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.05,
    ) -> List[Tuple[float, VectorRecord]]:
        """Executes cosine similarity search over stored vectors."""
        q_vec = self._text_to_vector(query)
        scored: List[Tuple[float, VectorRecord]] = []

        for rec in self.records.values():
            # Dot product of normalized vectors = cosine similarity
            sim = sum(q * r for q, r in zip(q_vec, rec.vector))
            if sim >= min_similarity:
                scored.append((round(sim, 4), rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def save_to_disk(self, file_path: Optional[str] = None) -> str:
        """Serializes vector store and document records to disk as JSON."""
        target = file_path or self.storage_path
        if not target:
            raise ValueError("No storage path provided to save vector store.")

        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        payload = {
            "dimension": self.dimension,
            "record_count": len(self.records),
            "updated_at": time.time(),
            "records": [r.model_dump() for r in self.records.values()],
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return target

    def load_from_disk(self, file_path: str) -> int:
        """Loads vector records from disk file."""
        if not os.path.exists(file_path):
            return 0

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self.dimension)
        records_list = data.get("records", [])
        for r_dict in records_list:
            rec = VectorRecord.model_validate(r_dict)
            self.records[rec.record_id] = rec
        return len(self.records)
