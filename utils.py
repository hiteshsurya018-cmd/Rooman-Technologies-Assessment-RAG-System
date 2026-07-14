from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_PDF_PATH = DATA_DIR / "paracetamol.pdf"


def ensure_data_dir() -> Path:
    """Create the data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def get_default_pdf_path() -> Path:
    """Return the default project PDF path."""
    return DEFAULT_PDF_PATH


def save_uploaded_pdf(uploaded_file) -> Path:
    """Persist an uploaded PDF to a temporary file for downstream loaders."""
    ensure_data_dir()
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=DATA_DIR,
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def build_document_id(file_path: Path, chunk_size: int, chunk_overlap: int, top_k: int) -> str:
    """Create a stable identifier for the current document and retrieval settings."""
    stat = file_path.stat()
    identity = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{chunk_size}|{chunk_overlap}|{top_k}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def format_chunk(doc: Document, score: float | None = None) -> str:
    """Render a readable chunk preview with metadata."""
    raw_page = doc.metadata.get("page_number", doc.metadata.get("page", "N/A"))
    page = raw_page + 1 if isinstance(raw_page, int) else raw_page
    source = doc.metadata.get("filename") or Path(str(doc.metadata.get("source", "uploaded document"))).name
    chunk_id = doc.metadata.get("chunk_id", "chunk_unknown")
    score_text = f"Similarity score: {score:.4f}\n" if score is not None else ""
    return (
        f"Source: {source}\n"
        f"Page: {page}\n"
        f"Chunk ID: {chunk_id}\n"
        f"{score_text}"
        f"Content:\n{doc.page_content.strip()}"
    )


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
    """Preserve order while removing duplicate chunks by content and page."""
    seen: set[tuple[str, str]] = set()
    unique_docs: list[Document] = []

    for doc in documents:
        key = (doc.page_content, str(doc.metadata.get("page", "")))
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs
