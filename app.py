from __future__ import annotations

import streamlit as st
from langchain_core.documents import Document

from rag_pipeline import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    RAGPipeline,
)
from utils import build_document_id, format_chunk, get_default_pdf_path, save_uploaded_pdf


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
    top_k: int,
) -> RAGPipeline:
    return RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
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
        raw_page = metadata.get("page_number", metadata.get("page", "N/A"))
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
                <div class="chunk-meta">{filename} | Page {page} | {chunk_id}{score_text}</div>
                <div class="chunk-text">{preview}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"View full chunk {index}", expanded=False):
            st.text(format_chunk_payload(chunk))


def ensure_document_ingested(
    pipeline: RAGPipeline,
    pdf_path,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> dict:
    document_id = build_document_id(pdf_path, chunk_size, chunk_overlap, top_k)
    current_id = st.session_state.get("current_document_id")

    if (
        current_id == document_id
        and st.session_state.get("ingestion_info")
        and pipeline.qa_chain is not None
        and pipeline.vectorstore is not None
    ):
        return st.session_state["ingestion_info"]

    ingestion_info = pipeline.ingest_pdf(pdf_path)
    st.session_state["current_document_id"] = document_id
    st.session_state["ingestion_info"] = ingestion_info
    return ingestion_info


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

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    default_pdf = get_default_pdf_path()

    if uploaded_file is not None:
        pdf_path = save_uploaded_pdf(uploaded_file)
        document_label = uploaded_file.name
    elif default_pdf.exists():
        pdf_path = default_pdf
        document_label = default_pdf.name
    else:
        pdf_path = None
        document_label = None

    if document_label:
        st.info(f"Selected document: {document_label}")
    else:
        st.warning("Upload a PDF to begin. No default PDF was found in the data folder.")

    question = st.text_input(
        "Ask a question about the document",
        placeholder="Example: What are the uses, dosage notes, or warnings mentioned for Paracetamol?",
    )

    ask_clicked = st.button("Ask Question", type="primary", use_container_width=True)

    if not ask_clicked:
        return

    if pdf_path is None:
        st.error("Please upload a PDF or place `data/paracetamol.pdf` in the project before asking a question.")
        return

    if not question.strip():
        st.error("Enter a question before running the RAG pipeline.")
        return

    pipeline = get_pipeline(
        chunk_size,
        chunk_overlap,
        top_k,
    )

    try:
        with st.spinner("Indexing the document, retrieving relevant chunks, and generating the answer..."):
            ingestion_info = ensure_document_ingested(
                pipeline,
                pdf_path,
                chunk_size,
                chunk_overlap,
                top_k,
            )
            result = pipeline.answer_question(question.strip())
    except Exception as exc:
        st.exception(exc)
        return

    st.success("Answer generated successfully.")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Pages loaded", ingestion_info["num_pages"])
    metric_col2.metric("Chunks created", ingestion_info["num_chunks"])
    metric_col3.metric("Top-k retrieved", top_k)

    st.subheader("Final Answer")
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("Retrieved Chunks")
        render_chunks(result["retrieved_chunks"], "No retrieved chunks were returned.")

    with right_col:
        st.subheader("Source Chunks Used for Answering")
        render_chunks(result["source_documents"], "No source chunks were returned by the QA chain.")


if __name__ == "__main__":
    main()
