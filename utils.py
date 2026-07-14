from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DEFAULT_PDF_PATH = DATA_DIR / "paracetamol.pdf"


def ensure_data_dir() -> Path:
    """Create the data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def ensure_uploads_dir() -> Path:
    """Create the uploaded document directory if it does not already exist."""
    ensure_data_dir()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    return UPLOADS_DIR


def get_default_pdf_path() -> Path:
    """Return the default project PDF path."""
    return DEFAULT_PDF_PATH


def uploaded_file_document_id(uploaded_file) -> str:
    """Return a stable content-based id for an uploaded file."""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()[:12]


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return cleaned or "uploaded.pdf"


def save_uploaded_pdf(uploaded_file, document_id: str | None = None) -> Path:
    """Persist an uploaded PDF in a stable path for downstream loaders."""
    doc_id = document_id or uploaded_file_document_id(uploaded_file)
    document_dir = ensure_uploads_dir() / doc_id
    document_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(uploaded_file.name)
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{safe_name}.pdf"

    target_path = document_dir / safe_name
    if not target_path.exists():
        target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def build_document_id(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int | None = None,
) -> str:
    """Create a stable identifier for the current document and chunking settings."""
    stat = file_path.stat()
    identity = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{chunk_size}|{chunk_overlap}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_document_set_id(
    file_paths: Iterable[Path],
    chunk_size: int,
    chunk_overlap: int,
    top_k: int | None = None,
) -> str:
    """Create a stable identifier for a set of documents and chunking settings."""
    identities: list[str] = []
    for file_path in sorted(Path(path) for path in file_paths):
        stat = file_path.stat()
        identities.append(f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")

    identity = "|".join(identities + [str(chunk_size), str(chunk_overlap)])
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def format_chunk(doc: Document, score: float | None = None) -> str:
    """Render a readable chunk preview with metadata."""
    page = doc.metadata.get("page_number")
    if page is None:
        raw_page = doc.metadata.get("page", "N/A")
        page = raw_page + 1 if isinstance(raw_page, int) else raw_page
    source = doc.metadata.get("filename") or Path(str(doc.metadata.get("source", "uploaded document"))).name
    chunk_id = doc.metadata.get("chunk_id", "chunk_unknown")
    document_id = doc.metadata.get("document_id", "unknown")
    score_text = f"Similarity score: {score:.4f}\n" if score is not None else ""
    return (
        f"Source: {source}\n"
        f"Page: {page}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Document ID: {document_id}\n"
        f"{score_text}"
        f"Content:\n{doc.page_content.strip()}"
    )


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
    """Preserve order while removing duplicate chunks by content and page."""
    seen: set[tuple[str, str, str]] = set()
    unique_docs: list[Document] = []

    for doc in documents:
        key = (
            str(doc.metadata.get("document_id", "")),
            str(doc.metadata.get("chunk_id", "")),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs
