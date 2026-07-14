# Assessment Compliance Notes

This file maps the Rooman Technologies Junior AI Research Associate AI Agent Challenge expectations to repository artifacts.

| Requirement | Evidence |
| --- | --- |
| Runnable project | `app.py`, `requirements.txt`, README run instructions |
| Python RAG app | `rag_pipeline.py`, `app.py` |
| Streamlit UI | `app.py` |
| LangChain usage | `RetrievalQA`, document loaders, prompts, retriever wrappers |
| FAISS retrieval | `FAISS.from_documents` in `rag_pipeline.py` |
| sentence-transformers embeddings | `HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` |
| FLAN-T5 generation | `LocalSeq2SeqLLM` using `google/flan-t5-base` |
| Source document upload | Streamlit multi-file PDF uploader |
| Multi-document support | `ingest_pdfs` builds one index over all selected PDFs |
| Inline citations | citation labels generated from filename, page, and chunk id |
| Source chunk display | retrieved evidence and QA source panels in the UI |
| Unsupported-question refusal | similarity threshold check before calling the LLM |
| Sample documents | `sample_docs/` |
| Sample questions | `sample_questions.md` |
| Expected outputs | `expected_outputs.md` |
| Tests | `tests/test_multi_document_rag.py` |
| Tradeoffs and limitations | README sections |

## Manual Review Checklist

- Install dependencies with `python -m pip install -r requirements.txt`.
- Run `python -m unittest discover -s tests -v`.
- Run `python -m compileall app.py rag_pipeline.py utils.py tests`.
- Start `streamlit run app.py`.
- Upload both sample PDFs.
- Ask supported and unsupported sample questions.
- Confirm answers include citations and retrieved chunks.
