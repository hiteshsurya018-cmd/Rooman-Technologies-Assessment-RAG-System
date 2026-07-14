# Paracetamol Document QA with RAG

This project is a Retrieval-Augmented Generation (RAG) application for answering questions across one or more PDF source documents. It follows the required architecture using LangChain, Hugging Face embeddings, FAISS, and Streamlit.

## Project Structure

```text
rag_paracetamol_project/
|-- app.py
|-- rag_pipeline.py
|-- utils.py
|-- requirements.txt
|-- README.md
`-- data/
    |-- paracetamol.pdf
    `-- uploads/
```

## Architecture Flow

```text
User Question + One or More PDFs
    |
Streamlit UI
    |
Independent PDF loading and chunking
    |
Single FAISS Vector DB
    |
MMR candidate retrieval + ranked evidence filtering
    |
Confidence and evidence threshold check
    |
LLM generates answer using context
    |
Return cited answer + source chunks, or no-answer response
```

## Chunking Strategy

The PDF is loaded with `PyPDFLoader` and split with `RecursiveCharacterTextSplitter` using:

- `chunk_size=600`
- `chunk_overlap=80`

These values are intentionally chosen within the required range:

- A chunk size around 600 characters keeps meaningful medical content together, such as usage notes, warnings, or dosage details.
- An overlap of 80 characters reduces boundary loss when a sentence spans two neighboring chunks.
- This balance improves retrieval precision without stripping away too much context.

## Embeddings

The project uses `HuggingFaceEmbeddings` with the model `sentence-transformers/all-MiniLM-L6-v2`.

Embeddings are vector representations of text. They map chunks and questions into the same semantic space so the retriever can compare meaning instead of relying only on exact word overlap.

Embeddings are essential in RAG because semantic search depends on them. Once both the query and the chunks are embedded, FAISS can quickly find the chunks whose meanings are closest to the user question.

## Retrieval Process

1. Load each selected PDF independently with `PyPDFLoader`.
2. Split each document into overlapping chunks with `RecursiveCharacterTextSplitter`.
3. Attach metadata to every chunk: `filename`, `page_number`, `chunk_id`, and `document_id`.
4. Convert each chunk into an embedding using `all-MiniLM-L6-v2`.
5. Store all vectors from all selected PDFs in one FAISS index.
6. Embed the user question.
7. Retrieve a larger candidate set from FAISS using Max Marginal Relevance when the installed LangChain FAISS integration supports it. If MMR is unavailable, the pipeline falls back to FAISS similarity search.
8. Convert FAISS distances into normalized similarity scores from `0.0` to `1.0`.
9. Sort evidence by similarity, remove near-duplicate chunks, then apply the document-diverse top-k selection policy.
10. Compute a retrieval confidence score and check the evidence threshold before calling the LLM.
11. If no chunk exceeds the threshold, skip the LLM and return the no-answer message.
12. If evidence is sufficient, pass those chunks to a LangChain `RetrievalQA` chain.
13. Generate an answer constrained to the retrieved context.
14. Return the final answer with inline citations, confidence, and the source chunks used.

## Streamlit Features

The app includes:

- multi-PDF uploader
- uploaded document list
- per-document remove and restore controls
- index status indicator
- build/rebuild index button, enabled only when the active document set or chunking settings require a new FAISS index
- question input box
- Ask Question button
- loading spinner
- final answer display with inline citations
- retrieval confidence display
- ranked retrieved evidence display
- source chunk display
- similarity score display for each retrieved chunk

## Implementation Notes

- `app.py` contains the Streamlit interface and avoids re-indexing the same document set on every question by caching ingestion state. Changing top-k reuses the current FAISS index; changing active PDFs or chunking settings rebuilds it.
- `rag_pipeline.py` contains the end-to-end RAG pipeline using LangChain, Hugging Face embeddings, FAISS, and a LangChain `RetrievalQA` chain. It processes each PDF independently, then combines the chunks into one searchable index. The QA chain uses the same MMR-backed, document-diverse retriever as the evidence panel so citations and displayed chunks stay aligned.
- `utils.py` contains helper functions for saving uploads, formatting chunks, and tracking the active document configuration.
- The default generation model is `google/flan-t5-base`, loaded locally through `transformers`.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Default Document

If you do not upload files, the app will use:

```text
data/paracetamol.pdf
```

## Multi-Document Usage

The upload control accepts multiple PDFs at once. Uploaded documents are saved under `data/uploads/` using a stable content-based document id. The app shows all uploaded documents, their active or removed status, and controls to remove or restore individual documents.

When multiple PDFs are active, the pipeline builds one FAISS index containing chunks from every active document. Retrieval uses a larger candidate pool and a document-diverse selection step to help return chunks from different PDFs for a single question when the query touches multiple sources. Inline citations use this format:

```text
[source.pdf | p.6 | chunk_024]
```

The source evidence panel also displays the document id, page number, chunk id, similarity score, and retrieved text for each chunk.

The FAISS index is reused when the active document set, file contents, chunk size, and chunk overlap are unchanged. Changing top-k only reconfigures retrieval and does not recompute embeddings. Removing or restoring a document, uploading a new PDF, or changing chunking settings marks the index for rebuild.

## Retrieval Confidence and No-Answer Behavior

The app computes confidence from retrieval scores, not from the generated answer. FAISS returns distance scores where lower is better, so the pipeline normalizes each retrieved chunk with:

```text
similarity = 1 / (1 + distance)
```

The displayed confidence is a weighted retrieval-quality score:

```text
confidence = 70% * best retrieved similarity + 30% * average retrieved similarity
```

The sidebar evidence threshold controls hallucination prevention. Before the LLM is called, at least one retrieved chunk must have a similarity score greater than the threshold. If no chunk exceeds the threshold, the LLM is skipped and the app returns:

```text
I could not find sufficient information in the provided documents to answer this question.

Reason:
No sufficiently relevant evidence was retrieved.
```

This behavior applies to unrelated, empty, and nonsense questions. For contradictory documents, the system still answers only from retrieved context and cites the supporting chunks so the disagreement remains visible in the evidence.

Retrieved evidence is always displayed in descending similarity order and includes similarity score, filename, page, chunk id, document id, and preview text. Near-duplicate chunks are removed before final evidence selection; when chunks are nearly identical, the highest-scoring chunk is kept.

## Verification

Run the multi-document regression test with:

```bash
python -m unittest discover -s tests -v
```

The tests create synthetic PDFs, build one FAISS index across documents, and check multi-document retrieval, source metadata, inline citation labels, no-answer threshold behavior, duplicate chunk removal, confidence score calculation, ranked evidence ordering, empty/nonsense questions, and contradictory-document grounding.

## Error Handling

The app handles the common failure cases gracefully:

- missing PDF
- empty question
- PDF with no extractable text
- model or dependency loading failures surfaced through Streamlit error output
