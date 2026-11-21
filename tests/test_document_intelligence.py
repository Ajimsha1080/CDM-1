"""Tests for CDM-1 Document Intelligence, PDF/DOCX/TXT extraction, and tenant-isolated RAG."""

import os
import io
import pytest
from cdm.retrieval.document_extractor import DocumentExtractor, TenantDocumentManager, ExtractedChunk
from cdm.synthesis.document_retriever import DocumentKnowledgeRetriever
from cdm.retrieval.vector_store import PersistentVectorStore


def test_txt_extraction_and_chunking():
    """Verify plain text extraction and semantic chunking."""
    sample_text = (
        "Enterprise Security Runbook (v3.2)\n\n"
        "Section 1: Multi-Factor Authentication Requirements.\n"
        "All operators accessing production clusters must authenticate via hardware security keys.\n\n"
        "Section 2: High-Value Financial Transfers.\n"
        "Any outgoing financial transfer exceeding $10,000 strictly requires dual executive sign-off."
    )
    text_bytes = sample_text.encode("utf-8")
    extracted, meta = DocumentExtractor.extract_from_bytes(text_bytes, "security_runbook.txt")
    assert "Multi-Factor Authentication" in extracted
    assert meta["file_type"] == "txt"

    chunks = DocumentExtractor.chunk_text(extracted, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    assert all(isinstance(c, ExtractedChunk) for c in chunks)
    assert any("Dual executive sign-off" in c.content or "$10,000" in c.content for c in chunks)


def test_docx_extraction(tmp_path):
    """Verify python-docx document text extraction."""
    import docx
    doc_path = str(tmp_path / "test_doc.docx")
    doc = docx.Document()
    doc.add_heading("Cloud Incident Response Policy", level=1)
    doc.add_paragraph("In the event of a Tier-1 infrastructure outage, isolation must occur within 120 seconds.")
    doc.add_paragraph("Database failover must preserve zero transaction loss.")
    doc.save(doc_path)

    with open(doc_path, "rb") as f:
        file_bytes = f.read()

    extracted, meta = DocumentExtractor.extract_from_bytes(file_bytes, "test_doc.docx")
    assert "Cloud Incident Response Policy" in extracted
    assert "Tier-1 infrastructure outage" in extracted
    assert meta["file_type"] == "docx"
    assert meta["paragraph_count"] >= 3


def test_pdf_extraction(tmp_path):
    """Verify pypdf extraction from synthetic PDF stream."""
    import pypdf
    pdf_path = str(tmp_path / "test_policy.pdf")
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    writer.write(pdf_path)

    # Test reading valid PDF
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    extracted, meta = DocumentExtractor.extract_from_bytes(pdf_bytes, "test_policy.pdf")
    assert meta["file_type"] == "pdf"
    assert meta["page_count"] == 1


def test_tenant_isolated_document_lifecycle(tmp_path):
    """Verify tenant isolation in document manager: upload, retrieval, and deletion."""
    db_path = str(tmp_path / "test_docs.sqlite")
    mgr = TenantDocumentManager(db_path=db_path)

    content_tenant_a = b"Tenant A Confidential Roadmap for 2027 quantum encryption deployment."
    content_tenant_b = b"Tenant B Healthcare Patient Records and clinical trials summary."

    doc_a, chunks_a = mgr.index_uploaded_file(
        tenant_id="tenant_alpha",
        filename="roadmap.txt",
        file_bytes=content_tenant_a,
        title="Alpha Roadmap",
    )
    doc_b, chunks_b = mgr.index_uploaded_file(
        tenant_id="tenant_beta",
        filename="clinical.txt",
        file_bytes=content_tenant_b,
        title="Beta Clinical",
    )

    # 1. Tenant Alpha lists only its documents
    alpha_docs = mgr.list_documents(tenant_id="tenant_alpha")
    assert len(alpha_docs) == 1
    assert alpha_docs[0].doc_id == doc_a.doc_id
    assert alpha_docs[0].title == "Alpha Roadmap"

    # 2. Cross-tenant access is blocked
    b_doc_from_a = mgr.get_document(doc_id=doc_b.doc_id, tenant_id="tenant_alpha")
    assert b_doc_from_a is None

    # 3. Deletion lifecycle
    deleted = mgr.delete_document(doc_id=doc_a.doc_id, tenant_id="tenant_alpha")
    assert deleted is True
    assert len(mgr.list_documents(tenant_id="tenant_alpha")) == 0
    assert len(mgr.list_documents(tenant_id="tenant_beta")) == 1


def test_hybrid_search_with_tenant_isolation(tmp_path):
    """Verify document knowledge retriever isolates search by tenant."""
    vstore = PersistentVectorStore(dimension=32)
    retriever = DocumentKnowledgeRetriever(vector_store=vstore)

    doc_alpha = {
        "doc_id": "DOC-ALPHA-POLICY",
        "title": "Alpha Cloud Scale Policy",
        "chunks": ["Nodes reporting CPU throttling above 90% must auto-scale immediately."],
    }
    doc_beta = {
        "doc_id": "DOC-BETA-POLICY",
        "title": "Beta Medical Compliance",
        "chunks": ["Patient blood pressure records require HIPAA compliant encryption."],
    }

    retriever.index_document(doc_alpha, tenant_id="tenant_alpha")
    retriever.index_document(doc_beta, tenant_id="tenant_beta")

    # Search for "throttling auto-scale" as Tenant Alpha
    res_alpha = retriever.retrieve("throttling auto-scale", tenant_id="tenant_alpha")
    assert len(res_alpha) > 0
    assert any("DOC-ALPHA-POLICY" in p.doc_id for p in res_alpha)
    # Ensure Tenant Beta document is NOT returned to Tenant Alpha
    assert not any("DOC-BETA-POLICY" in p.doc_id for p in res_alpha)
