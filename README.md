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
Document-diverse top-k relevant chunks
    |
LLM generates answer using context
    |
Return cited answer + source chunks
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
7. Retrieve a larger candidate set from FAISS, then select the final top-k chunks with a document-diverse policy so one document does not dominate multi-document queries when other relevant sources are available.
8. Pass those chunks to a LangChain `RetrievalQA` chain.
9. Generate an answer constrained to the retrieved context.
10. Return the final answer with inline citations and the source chunks used.

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
- retrieved chunk display
- source chunk display
- similarity score display for each retrieved chunk

## Implementation Notes

- `app.py` contains the Streamlit interface and avoids re-indexing the same document set on every question by caching ingestion state. Changing top-k reuses the current FAISS index; changing active PDFs or chunking settings rebuilds it.
- `rag_pipeline.py` contains the end-to-end RAG pipeline using LangChain, Hugging Face embeddings, FAISS, and a LangChain `RetrievalQA` chain. It processes each PDF independently, then combines the chunks into one searchable index. The QA chain uses the same document-diverse retriever as the evidence panel so citations and displayed chunks stay aligned.
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

## Verification

Run the multi-document regression test with:

```bash
python -m unittest discover -s tests -v
```

The test creates two different PDFs, builds one FAISS index across both, asks a query that mentions both documents, and checks that retrieval, source documents, metadata, and inline citation labels include both PDFs.

## Error Handling

The app handles the common failure cases gracefully:

- missing PDF
- empty question
- PDF with no extractable text
- model or dependency loading failures surfaced through Streamlit error output
