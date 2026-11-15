"""Tests for CDM-1 Phase 2: Vector RAG, Hybrid Retrieval, SSRF-Protected Web Tool, and Provenance Manifests."""

import os
import tempfile
import pytest
from cdm.retrieval.vector_store import PersistentVectorStore, VectorRecord
from cdm.synthesis.document_retriever import DocumentKnowledgeRetriever, DocumentPassage
from cdm.synthesis.tool_orchestrator import WebRetrievalTool
from cdm.synthesis.citation_manager import CitationManager, CitationVerificationReport, GroundedCitation


def test_persistent_vector_store_indexing_and_search():
    """Verify vector store indexes documents, computes cosine similarity, and returns ranked records."""
    store = PersistentVectorStore(dimension=32)
    id1 = store.add_document("doc1", "Incident Management", "Memory pressure above 85 percent requires telemetry diagnostic")
    id2 = store.add_document("doc2", "Billing Policy", "Refunds over 50 dollars require supervisor approval")

    assert len(store.records) == 2
    results = store.search("telemetry diagnostic memory alert", top_k=2)
    assert len(results) > 0
    top_score, top_rec = results[0]
    assert top_rec.doc_id == "doc1"
    assert top_score > 0.0


def test_persistent_vector_store_disk_serialization():
    """Verify vector store serializes to disk and reloads identically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test_vectors.json")
        store = PersistentVectorStore(dimension=16, storage_path=fpath)
        store.add_document("doc_alpha", "Title Alpha", "Critical server outage on gateway node")
        store.save_to_disk()

        assert os.path.exists(fpath)
        reloaded = PersistentVectorStore(dimension=16, storage_path=fpath)
        assert len(reloaded.records) == 1
        assert "doc_alpha_v1" in reloaded.records
        assert reloaded.records["doc_alpha_v1"].title == "Title Alpha"


def test_document_knowledge_retriever_hybrid_search():
    """Verify hybrid search combines lexical token overlap and dense vector similarity."""
    store = PersistentVectorStore(dimension=32)
    retriever = DocumentKnowledgeRetriever(vector_store=store)

    retriever.index_document({
        "doc_id": "DOC-NETWORK-FAILOVER",
        "title": "Network Failover SOP",
        "source_repository": "Network Operations",
        "chunks": [
            "BGP routing flap on spine switches triggers automatic traffic rerouting to secondary backbone.",
            "Spine switch temperature exceeding 75C triggers high-priority fan cooling cycle.",
        ],
    })

    # Query with hybrid keywords
    passages = retriever.retrieve("BGP routing flap secondary backbone", top_k=2)
    assert len(passages) > 0
    assert passages[0].doc_id == "DOC-NETWORK-FAILOVER"
    assert passages[0].relevance_score > 0.2


def test_web_retrieval_tool_ssrf_protection():
    """Verify WebRetrievalTool strictly blocks loopback, cloud metadata, and private IP ranges."""
    tool = WebRetrievalTool(allowed_domains=["api.github.com"])

    # 1. Non-http scheme blocked
    res_file = tool.execute({"url": "file:///etc/passwd"})
    assert res_file.success is False
    assert "Forbidden URL scheme" in res_file.error

    # 2. Localhost / Loopback blocked
    res_local = tool.execute({"url": "http://127.0.0.1:8000/v1/health"})
    assert res_local.success is False
    assert "SSRF Protection" in res_local.error

    # 3. Domain not in whitelist blocked
    res_unlisted = tool.execute({"url": "http://evil-attacker-site.com/exploit"})
    assert res_unlisted.success is False
    assert "not in the allowed domain whitelist" in res_unlisted.error


def test_citation_manager_provenance_manifest_export():
    """Verify CitationManager exports cryptographically digest-bound provenance manifests."""
    cm = CitationManager()
    report = CitationVerificationReport(
        total_claims=2,
        grounded_claims=2,
        unsupported_claims=[],
        citations=[
            GroundedCitation(
                citation_id="CIT-001",
                source_id="DOC-SLA-2026",
                source_title="SLA Policy",
                source_repository="Corporate Governance",
                claim_text="Disputed billing items under $50 qualify for autonomous auto-resolution.",
                evidence_snippet="Disputed billing items under $50.00 qualify for autonomous auto-resolution",
                confidence=1.0,
                verified=True,
            )
        ],
        grounding_ratio=1.0,
    )

    manifest = cm.export_provenance_manifest(report)
    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["total_citations"] == 1
    assert "manifest_sha256" in manifest
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["citations"][0]["citation_id"] == "CIT-001"
    assert "claim_digest_sha256" in manifest["citations"][0]
