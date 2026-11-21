"""Document and Knowledge-Base Retrieval Subsystem for CDM-1.
Supports indexing and querying verified enterprise documentation and knowledge bases.
Returns grounded passages with document identifiers, authoritative source repositories,
version metadata, and retrieval timestamps while strictly distinguishing retrieved facts
from model hypotheses.
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import re
from pydantic import BaseModel, Field

from cdm.retrieval.vector_store import PersistentVectorStore


class DocumentPassage(BaseModel):
    """Grounded passage retrieved from enterprise knowledge base."""
    passage_id: str
    doc_id: str
    tenant_id: str = "default_tenant"
    title: str
    content: str
    source_repository: str
    version: str = "1.0.0"
    retrieval_timestamp: float = Field(default_factory=time.time)
    relevance_score: float = 1.0
    provenance_type: str = "document_knowledge"  # 'document_knowledge', 'semantic_rule', 'external_telemetry'


class DocumentKnowledgeRetriever:
    """Manages verified document repository indexing and multi-criteria factual retrieval."""

    def __init__(
        self,
        initial_documents: Optional[List[Dict[str, Any]]] = None,
        vector_store: Optional[PersistentVectorStore] = None,
        vector_storage_path: Optional[str] = None,
    ):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.passages: List[DocumentPassage] = []
        if vector_store is not None:
            self.vector_store = vector_store
        elif vector_storage_path:
            self.vector_store = PersistentVectorStore(storage_path=vector_storage_path)
        else:
            self.vector_store = None

        # Populate baseline corporate governance and operational knowledge base
        self._load_baseline_knowledge()
        if initial_documents:
            for d in initial_documents:
                self.index_document(d)

    def _load_baseline_knowledge(self) -> None:
        """Loads canonical verified operational policies and standard operating procedures."""
        base_docs = [
            {
                "doc_id": "DOC-SLA-2026",
                "title": "Corporate Standard Level Agreement Policy (v2.4)",
                "source_repository": "Corporate Governance Repository",
                "version": "2.4.1",
                "chunks": [
                    "Tier-1 incident response requires initial contact within 4.0 hours and resolution target within 24.0 hours.",
                    "Disputed billing items under $50.00 qualify for autonomous auto-resolution if customer account is in good standing.",
                    "Refund transactions exceeding $50.00 strictly require secondary review and authorization by an authorized Compliance Officer.",
                ],
            },
            {
                "doc_id": "DOC-INFRA-OPS",
                "title": "Infrastructure Incident Remediation Runbook",
                "source_repository": "Engineering Operations Manual",
                "version": "3.1.0",
                "chunks": [
                    "Direct database drop or record deletion actions are strictly prohibited in autonomous execution tiers without executive sign-off.",
                    "Cluster nodes reporting memory pressure above 85% must undergo safe telemetry diagnosis prior to service isolation.",
                    "Standard health checks must be executed every 60 seconds across all production gateway nodes.",
                ],
            },
            {
                "doc_id": "DOC-DATA-PRIVACY",
                "title": "Global Data Protection and Privacy Compliance",
                "source_repository": "Legal & Privacy Archive",
                "version": "4.0.0",
                "chunks": [
                    "Customer PII including full credit card numbers and passwords must be redacted from operational logs and telemetry traces.",
                    "Account deactivation requests require verified user identity confirmation.",
                ],
            },
        ]
        for d in base_docs:
            self.index_document(d)

    def index_document(self, doc_dict: Dict[str, Any], tenant_id: str = "default_tenant") -> None:
        """Indexes a document and slices it into searchable grounded passages with tenant isolation."""
        doc_id = doc_dict["doc_id"]
        title = doc_dict.get("title", "Untitled Document")
        repo = doc_dict.get("source_repository", "Enterprise Repository")
        version = doc_dict.get("version", "1.0.0")
        chunks = doc_dict.get("chunks", [])

        self.documents[doc_id] = {
            "title": title,
            "source_repository": repo,
            "version": version,
            "chunks_count": len(chunks),
            "tenant_id": tenant_id,
        }

        for idx, chunk in enumerate(chunks):
            passage = DocumentPassage(
                passage_id=f"{doc_id}_p{idx + 1}",
                doc_id=doc_id,
                tenant_id=tenant_id,
                title=title,
                content=chunk,
                source_repository=repo,
                version=version,
                retrieval_timestamp=time.time(),
            )
            self.passages.append(passage)
            if self.vector_store is not None:
                self.vector_store.add_document(
                    doc_id=doc_id,
                    title=title,
                    content=chunk,
                    source_repository=repo,
                    version=version,
                    tenant_id=tenant_id,
                    metadata={"passage_id": passage.passage_id},
                )

    def delete_document(self, doc_id: str, tenant_id: Optional[str] = None) -> int:
        """Deletes a document and its passages from the retriever and vector store."""
        removed_count = 0
        new_passages = []
        for p in self.passages:
            if p.doc_id == doc_id:
                if tenant_id and p.tenant_id != tenant_id:
                    new_passages.append(p)
                    continue
                removed_count += 1
            else:
                new_passages.append(p)
        self.passages = new_passages
        if doc_id in self.documents:
            if not tenant_id or self.documents[doc_id].get("tenant_id") == tenant_id:
                del self.documents[doc_id]
        if self.vector_store is not None:
            self.vector_store.delete_document(doc_id=doc_id, tenant_id=tenant_id)
        return removed_count

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.1,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentPassage]:
        """Retrieves passages matching the query with verified provenance and tenant isolation.
        Supports hybrid dense-vector + lexical token retrieval when vector store is present.
        Guaranteed zero fabrication: only existing indexed passages can be returned.
        """
        q_terms = set(re.findall(r"\w+", query.lower()))
        passage_map = {p.passage_id: p for p in self.passages}
        scores_by_id: Dict[str, float] = {}

        # Filter candidate passages by tenant
        tenant_passages = [
            p for p in self.passages
            if not tenant_id or p.tenant_id == tenant_id or p.tenant_id == "default_tenant"
        ]

        # 1. Lexical token overlap scoring
        for p in tenant_passages:
            p_terms = set(re.findall(r"\w+", p.content.lower()))
            overlap = len(q_terms.intersection(p_terms))
            if overlap > 0:
                bm25_score = round(overlap / max(1, len(q_terms)), 4)
                scores_by_id[p.passage_id] = bm25_score

        # 2. Dense vector similarity scoring (if vector store enabled)
        if self.vector_store is not None:
            v_matches = self.vector_store.search(query, top_k=top_k * 2, min_similarity=min_score, tenant_id=tenant_id)
            for v_score, rec in v_matches:
                p_id = rec.metadata.get("passage_id") or rec.record_id
                if p_id in scores_by_id:
                    # Hybrid fusion: 50% lexical + 50% vector similarity
                    scores_by_id[p_id] = round(0.5 * scores_by_id[p_id] + 0.5 * v_score, 4)
                elif p_id in passage_map and v_score >= min_score:
                    p = passage_map[p_id]
                    if not tenant_id or p.tenant_id == tenant_id or p.tenant_id == "default_tenant":
                        scores_by_id[p_id] = round(v_score * 0.7, 4)

        scored_passages: List[Tuple[float, DocumentPassage]] = []
        for p_id, score in scores_by_id.items():
            if score >= min_score and p_id in passage_map:
                p = passage_map[p_id]
                matched = DocumentPassage(
                    passage_id=p.passage_id,
                    doc_id=p.doc_id,
                    tenant_id=p.tenant_id,
                    title=p.title,
                    content=p.content,
                    source_repository=p.source_repository,
                    version=p.version,
                    retrieval_timestamp=time.time(),
                    relevance_score=score,
                    provenance_type="document_knowledge",
                )
                scored_passages.append((score, matched))

        scored_passages.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored_passages[:top_k]]
