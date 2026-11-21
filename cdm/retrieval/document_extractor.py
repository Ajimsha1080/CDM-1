"""Document Extraction and Intelligence Subsystem for CDM-1.
Supports PDF, DOCX, and TXT file parsing, semantic chunking, and tenant-isolated
knowledge indexing with full cryptographic citation provenance.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import io
import time
import hashlib
import sqlite3
import threading
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger("cdm.retrieval.extractor")


class ExtractedChunk(BaseModel):
    """Chunk of extracted text with structural provenance."""
    chunk_index: int
    content: str
    char_count: int
    word_count: int
    sha256: str


class DocumentMetadata(BaseModel):
    """Metadata describing an uploaded and indexed enterprise document."""
    doc_id: str
    tenant_id: str
    filename: str
    file_type: str  # 'pdf', 'docx', 'txt', 'md', 'json'
    file_size_bytes: int
    title: str
    source_repository: str = "Enterprise Knowledge Repository"
    version: str = "1.0.0"
    uploaded_at: float = Field(default_factory=time.time)
    sha256_hash: str
    chunk_count: int = 0
    total_characters: int = 0


class DocumentExtractor:
    """Extracts raw textual content and structured sections from PDF, DOCX, and text documents."""

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Extracts text content and metadata from raw bytes based on file extension.
        Returns (extracted_text, metadata_dict).
        """
        ext = os.path.splitext(filename)[1].lower().strip(".")
        meta: Dict[str, Any] = {"file_type": ext, "file_size": len(file_bytes)}

        if ext == "pdf":
            text = DocumentExtractor._extract_pdf(file_bytes, meta)
        elif ext in ("docx", "doc"):
            text = DocumentExtractor._extract_docx(file_bytes, meta)
        else:
            # Default plain text / markdown / json / csv
            text = DocumentExtractor._extract_text(file_bytes, meta)

        return text.strip(), meta

    @staticmethod
    def _extract_pdf(file_bytes: bytes, meta: Dict[str, Any]) -> str:
        """Extracts text from PDF bytes using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            meta["page_count"] = len(reader.pages)
            pages_text = []
            for idx, page in enumerate(reader.pages):
                page_str = page.extract_text() or ""
                pages_text.append(page_str.strip())
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise ValueError(f"Failed to extract PDF content: {e}")

    @staticmethod
    def _extract_docx(file_bytes: bytes, meta: Dict[str, Any]) -> str:
        """Extracts text from DOCX bytes using python-docx."""
        try:
            import docx
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            meta["paragraph_count"] = len(paragraphs)
            return "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            raise ValueError(f"Failed to extract DOCX content: {e}")

    @staticmethod
    def _extract_text(file_bytes: bytes, meta: Dict[str, Any]) -> str:
        """Extracts text from plain text formats with encoding fallback."""
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="replace")

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[ExtractedChunk]:
        """Splits extracted text into overlapping semantic chunks for vector and lexical indexing."""
        if not text:
            return []

        # Split along paragraph boundaries if possible
        raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []

        current_buf = ""
        for p in raw_paragraphs:
            if len(current_buf) + len(p) + 2 <= chunk_size:
                current_buf = f"{current_buf}\n\n{p}".strip()
            else:
                if current_buf:
                    chunks.append(current_buf)
                if len(p) > chunk_size:
                    # Break long paragraphs into sliding windows
                    start = 0
                    while start < len(p):
                        end = start + chunk_size
                        chunks.append(p[start:end])
                        start = end - overlap
                    current_buf = ""
                else:
                    current_buf = p

        if current_buf:
            chunks.append(current_buf)

        if not chunks:
            chunks = [text[:chunk_size]]

        result: List[ExtractedChunk] = []
        for idx, c in enumerate(chunks):
            c_clean = c.strip()
            c_hash = hashlib.sha256(c_clean.encode("utf-8")).hexdigest()
            result.append(
                ExtractedChunk(
                    chunk_index=idx + 1,
                    content=c_clean,
                    char_count=len(c_clean),
                    word_count=len(c_clean.split()),
                    sha256=c_hash,
                )
            )

        return result


class TenantDocumentManager:
    """Manages persistent document indexing, lifecycle operations, and tenant isolation."""

    def __init__(self, db_path: str = "experiments/cognitive_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS tenant_documents (
                            doc_id TEXT PRIMARY KEY,
                            tenant_id TEXT NOT NULL,
                            filename TEXT NOT NULL,
                            file_type TEXT NOT NULL,
                            file_size_bytes INTEGER NOT NULL,
                            title TEXT NOT NULL,
                            source_repository TEXT NOT NULL,
                            version TEXT NOT NULL,
                            uploaded_at REAL NOT NULL,
                            sha256_hash TEXT NOT NULL,
                            chunk_count INTEGER NOT NULL,
                            total_characters INTEGER NOT NULL,
                            raw_text TEXT NOT NULL,
                            chunks_json TEXT NOT NULL
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_docs_tenant ON tenant_documents (tenant_id);")
            finally:
                conn.close()

    def index_uploaded_file(
        self,
        tenant_id: str,
        filename: str,
        file_bytes: bytes,
        title: Optional[str] = None,
        source_repository: str = "Enterprise Knowledge Repository",
        version: str = "1.0.0",
        doc_id: Optional[str] = None,
    ) -> Tuple[DocumentMetadata, List[ExtractedChunk]]:
        """Extracts, chunks, and persists an uploaded file with tenant isolation."""
        text, meta = DocumentExtractor.extract_from_bytes(file_bytes, filename)
        chunks = DocumentExtractor.chunk_text(text)
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()

        d_id = doc_id or f"doc_{sha256_hash[:12]}"
        doc_title = title or os.path.splitext(filename)[0].replace("_", " ").title()
        ext = meta.get("file_type", "txt")
        now = time.time()

        doc_meta = DocumentMetadata(
            doc_id=d_id,
            tenant_id=tenant_id,
            filename=filename,
            file_type=ext,
            file_size_bytes=len(file_bytes),
            title=doc_title,
            source_repository=source_repository,
            version=version,
            uploaded_at=now,
            sha256_hash=sha256_hash,
            chunk_count=len(chunks),
            total_characters=len(text),
        )

        import json
        chunks_json = json.dumps([c.model_dump() for c in chunks])

        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO tenant_documents (
                            doc_id, tenant_id, filename, file_type, file_size_bytes,
                            title, source_repository, version, uploaded_at, sha256_hash,
                            chunk_count, total_characters, raw_text, chunks_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (
                        d_id, tenant_id, filename, ext, len(file_bytes),
                        doc_title, source_repository, version, now, sha256_hash,
                        len(chunks), len(text), text, chunks_json
                    ))
            finally:
                conn.close()

        return doc_meta, chunks

    def list_documents(self, tenant_id: str) -> List[DocumentMetadata]:
        """Lists all uploaded documents for a tenant."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("""
                    SELECT doc_id, tenant_id, filename, file_type, file_size_bytes,
                           title, source_repository, version, uploaded_at, sha256_hash,
                           chunk_count, total_characters
                    FROM tenant_documents
                    WHERE tenant_id = ?
                    ORDER BY uploaded_at DESC;
                """, (tenant_id,))
                rows = cursor.fetchall()
                results = []
                for r in rows:
                    results.append(DocumentMetadata(**dict(r)))
                return results
            finally:
                conn.close()

    def get_document(self, doc_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete document content and chunks for a tenant."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("""
                    SELECT * FROM tenant_documents
                    WHERE doc_id = ? AND tenant_id = ?;
                """, (doc_id, tenant_id))
                row = cursor.fetchone()
                if not row:
                    return None
                data = dict(row)
                import json
                data["chunks"] = json.loads(data["chunks_json"])
                return data
            finally:
                conn.close()

    def delete_document(self, doc_id: str, tenant_id: str) -> bool:
        """Deletes a document and its chunks for a tenant."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cursor = conn.execute("""
                        DELETE FROM tenant_documents
                        WHERE doc_id = ? AND tenant_id = ?;
                    """, (doc_id, tenant_id))
                    return cursor.rowcount > 0
            finally:
                conn.close()
