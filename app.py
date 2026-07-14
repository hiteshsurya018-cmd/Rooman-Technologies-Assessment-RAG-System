from __future__ import annotations

import streamlit as st
from langchain_core.documents import Document

from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    RAGPipeline,
)
from utils import (
    build_document_set_id,
    format_chunk,
    get_default_pdf_path,
    save_uploaded_pdf,
    uploaded_file_document_id,
)


st.set_page_config(
    page_title="Paracetamol RAG QA System",
    layout="wide",
)


MINIMAL_CSS = """
<style>
.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

.app-note {
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: #f6f8fb;
    border: 1px solid #d9e2ec;
    color: #334e68;
    margin-bottom: 1rem;
}

.answer-box {
    padding: 1rem 1.1rem;
    border-radius: 14px;
    background: #fbfcfe;
    border: 1px solid #d9e2ec;
    color: #102a43;
    line-height: 1.65;
}

.answer-box * {
    color: #102a43 !important;
}

.chunk-box {
    padding: 0.85rem 1rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    margin-bottom: 0.75rem;
}

.chunk-meta {
    font-size: 0.85rem;
    color: #52606d;
    margin-bottom: 0.4rem;
}

.chunk-text {
    color: #102a43;
    line-height: 1.55;
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_pipeline(
    chunk_size: int,
    chunk_overlap: int,
) -> RAGPipeline:
    return RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def format_chunk_payload(chunk: dict) -> str:
    document = Document(
        page_content=chunk["content"],
        metadata=chunk.get("metadata", {}),
    )
    return format_chunk(document, chunk.get("score"))


def render_chunks(chunks: list[dict], empty_message: str) -> None:
    if not chunks:
        st.info(empty_message)
        return

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        filename = metadata.get("filename", "source")
        document_id = metadata.get("document_id", "unknown")
        page = metadata.get("page_number")
        if page is None:
            raw_page = metadata.get("page", "N/A")
            page = raw_page + 1 if isinstance(raw_page, int) else raw_page
        chunk_id = metadata.get("chunk_id", f"chunk_{index:03d}")
        score = chunk.get("score")
        preview = chunk.get("content", "").strip()
        if len(preview) > 260:
            preview = preview[:260].rstrip() + "..."

        score_text = f" | Similarity: {score:.4f}" if score is not None else ""
        st.markdown(
            f"""
            <div class="chunk-box">
                <div class="chunk-meta">{filename} | Page {page} | {chunk_id} | Doc ID: {document_id}{score_text}</div>
                <div class="chunk-text">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"View full chunk {index}", expanded=False):
            st.text(format_chunk_payload(chunk))


def ensure_documents_ingested(
    pipeline: RAGPipeline,
    pdf_paths: list,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    similarity_threshold: float,
) -> dict:
    document_set_id = build_document_set_id(pdf_paths, chunk_size, chunk_overlap)
    current_id = st.session_state.get("current_document_set_id")

    if (
        current_id == document_set_id
        and st.session_state.get("ingestion_info")
        and pipeline.qa_chain is not None
        and pipeline.vectorstore is not None
    ):
        pipeline.configure_retrieval(top_k, similarity_threshold)
        st.session_state["current_top_k"] = top_k
        st.session_state["current_similarity_threshold"] = similarity_threshold
        return st.session_state["ingestion_info"]

    pipeline.configure_retrieval(top_k, similarity_threshold)
    ingestion_info = pipeline.ingest_pdfs(pdf_paths)
    st.session_state["current_document_set_id"] = document_set_id
    st.session_state["current_top_k"] = top_k
    st.session_state["current_similarity_threshold"] = similarity_threshold
    st.session_state["ingestion_info"] = ingestion_info
    return ingestion_info


def invalidate_cached_index() -> None:
    st.session_state.pop("current_document_set_id", None)
    st.session_state.pop("current_top_k", None)
    st.session_state.pop("current_similarity_threshold", None)
    st.session_state.pop("ingestion_info", None)


def register_uploaded_documents(uploaded_files) -> None:
    st.session_state.setdefault("uploaded_documents", {})
    st.session_state.setdefault("active_document_ids", set())

    for uploaded_file in uploaded_files or []:
        document_id = uploaded_file_document_id(uploaded_file)
        if document_id in st.session_state["uploaded_documents"]:
            continue

        saved_path = save_uploaded_pdf(uploaded_file, document_id=document_id)
        st.session_state["uploaded_documents"][document_id] = {
            "document_id": document_id,
            "filename": uploaded_file.name,
            "path": saved_path,
            "size": uploaded_file.size,
        }
        st.session_state["active_document_ids"].add(document_id)
        invalidate_cached_index()


def render_uploaded_document_list() -> list[dict]:
    documents = st.session_state.get("uploaded_documents", {})
    active_ids = st.session_state.setdefault("active_document_ids", set())

    if not documents:
        return []

    st.subheader("Uploaded Documents")
    for document_id, document in documents.items():
        is_active = document_id in active_ids
        status = "Active" if is_active else "Removed"
        col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
        col1.write(document["filename"])
        col2.write(f"{document['size'] / 1024:.1f} KB")
        col3.write(status)
        if is_active:
            if col4.button("Remove", key=f"remove_{document_id}"):
                active_ids.remove(document_id)
                invalidate_cached_index()
                st.rerun()
        else:
            if col4.button("Restore", key=f"restore_{document_id}"):
                active_ids.add(document_id)
                invalidate_cached_index()
                st.rerun()

    return [
        document
        for document_id, document in documents.items()
        if document_id in active_ids
    ]


def main() -> None:
    st.markdown(MINIMAL_CSS, unsafe_allow_html=True)
    st.title("Document Question Answering with RAG")
    st.markdown(
        """
        <div class="app-note">
            Upload a PDF about Paracetamol, retrieve the most relevant chunks with FAISS,
            and generate grounded answers using a Hugging Face model through LangChain.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Retrieval Configuration")
        chunk_size = st.slider(
            "Chunk size",
            min_value=400,
            max_value=800,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Larger chunks preserve more context, while smaller chunks improve retrieval precision.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=50,
            max_value=100,
            value=DEFAULT_CHUNK_OVERLAP,
            step=10,
            help="Overlap keeps adjacent chunks connected so key facts are not split apart.",
        )
        top_k = st.slider(
            "Top-k chunks",
            min_value=1,
            max_value=5,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of chunks retrieved from FAISS before answer generation.",
        )
        similarity_threshold = st.slider(
            "Evidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help="The LLM is skipped unless at least one retrieved chunk exceeds this similarity score.",
        )

    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    register_uploaded_documents(uploaded_files)
    active_documents = render_uploaded_document_list()
    default_pdf = get_default_pdf_path()

    if active_documents:
        pdf_paths = [document["path"] for document in active_documents]
        document_label = ", ".join(document["filename"] for document in active_documents)
    elif not st.session_state.get("uploaded_documents") and default_pdf.exists():
        pdf_paths = [default_pdf]
        document_label = default_pdf.name
    else:
        pdf_paths = []
        document_label = None

    if document_label:
        st.info(f"Selected documents: {document_label}")
    else:
        st.warning("Upload one or more PDFs to begin. No active documents are selected.")

    document_set_id = build_document_set_id(pdf_paths, chunk_size, chunk_overlap) if pdf_paths else None
    index_is_current = (
        document_set_id is not None
        and st.session_state.get("current_document_set_id") == document_set_id
        and st.session_state.get("ingestion_info") is not None
    )
    if pdf_paths:
        st.caption(
            "Index status: current. Top-k changes will reuse the existing FAISS index."
            if index_is_current
            else "Index status: rebuild needed. The index will be built before retrieval."
        )

        rebuild_clicked = st.button(
            "Build / Rebuild Index",
            disabled=index_is_current,
            use_container_width=True,
        )
        if rebuild_clicked:
            pipeline = get_pipeline(
                chunk_size,
                chunk_overlap,
            )
            try:
                with st.spinner("Building one FAISS index from all active PDFs..."):
                    ingestion_info = ensure_documents_ingested(
                        pipeline,
                        pdf_paths,
                        chunk_size,
                        chunk_overlap,
                        top_k,
                        similarity_threshold,
                    )
            except Exception as exc:
                st.exception(exc)
                return
            st.success(
                f"Indexed {len(ingestion_info['documents'])} document(s) into "
                f"{ingestion_info['num_chunks']} chunks."
            )

    question = st.text_input(
        "Ask a question about the document",
        placeholder="Example: What are the uses, dosage notes, or warnings mentioned for Paracetamol?",
    )

    ask_clicked = st.button("Ask Question", type="primary", use_container_width=True)

    if not ask_clicked:
        return

    if not pdf_paths:
        st.error("Please select at least one PDF before asking a question.")
        return

    if not question.strip():
        st.error("Enter a question before running the RAG pipeline.")
        return

    pipeline = get_pipeline(
        chunk_size,
        chunk_overlap,
    )

    try:
        with st.spinner("Indexing selected documents if needed, retrieving relevant chunks, and generating the answer..."):
            ingestion_info = ensure_documents_ingested(
                pipeline,
                pdf_paths,
                chunk_size,
                chunk_overlap,
                top_k,
                similarity_threshold,
            )
            result = pipeline.answer_question(question.strip())
    except Exception as exc:
        st.exception(exc)
        return

    if result.get("llm_called"):
        st.success("Answer generated successfully.")
    else:
        st.warning("No sufficiently relevant evidence was retrieved.")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Pages loaded", ingestion_info["num_pages"])
    metric_col2.metric("Chunks created", ingestion_info["num_chunks"])
    metric_col3.metric("Top-k retrieved", top_k)
    metric_col4.metric("Confidence", f"{result['confidence']}%")

    with st.expander("Indexed documents", expanded=False):
        for document in ingestion_info["documents"]:
            st.write(
                f"{document['filename']} | pages: {document['num_pages']} | "
                f"chunks: {document['num_chunks']} | document_id: {document['document_id']}"
            )

    st.subheader("Final Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Retrieved Evidence")
        render_chunks(result["retrieved_chunks"], "No retrieved chunks were returned.")

    with right_col:
        st.subheader("Source Chunks Used for Answering")
        render_chunks(result["source_documents"], "No source chunks were returned by the QA chain.")


if __name__ == "__main__":
    main()
