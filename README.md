# Multi-Document Research Agent with Citations

## Project Overview

This project is a local Streamlit research agent for answering questions over one or more PDF documents. It uses Retrieval-Augmented Generation (RAG): uploaded PDFs are split into chunks, embedded with sentence-transformers, indexed with FAISS, retrieved for each question, and passed to a local FLAN-T5 model through LangChain.

The app is built for reviewable answers. It shows inline citations, retrieved evidence, source chunks, and a confidence score. If retrieval does not find sufficiently relevant evidence, the app refuses to answer instead of asking the language model to guess.

## Architecture

```text
User question + uploaded PDFs
        |
        v
Streamlit UI
        |
        v
PDF loading with PyPDFLoader
        |
        v
Chunking with RecursiveCharacterTextSplitter
        |
        v
Embeddings: sentence-transformers/all-MiniLM-L6-v2
        |
        v
FAISS vector index
        |
        v
Retriever + relevance threshold
        |
        v
FLAN-T5 generation through LangChain
        |
        v
Answer, inline citations, confidence, retrieved chunks
```

## Features

- Accepts one or more source PDF documents.
- Accepts a natural-language user question.
- Builds one FAISS index across all active documents.
- Retrieves relevant passages with document metadata.
- Synthesizes an answer with a local FLAN-T5 model.
- Adds inline citations in the format `[filename.pdf | p.1 | chunk_001]`.
- Displays retrieved evidence and source chunks.
- Refuses unsupported or nonsense questions when retrieval confidence is too low.
- Supports reproducible local execution with pinned dependencies.
- Includes unit tests for retrieval, citations, refusal, confidence, and multi-document behavior.

## Requirements

- Python 3.11
- Internet access for the first install and first model download
- CPU execution is supported; no GPU is required

Main models:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Generation: `google/flan-t5-base`

Main libraries:

- Streamlit
- LangChain / LangChain Community / LangChain Hugging Face
- FAISS CPU
- sentence-transformers
- transformers
- torch
- pypdf

## Installation

```bash
git clone <repository-url>
cd "Rooman Technologies Assessment RAG System"
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Environment Variables

No API key is required. The project runs with local Hugging Face models.

Optional Hugging Face cache variables may be used if a reviewer wants model files stored outside the default cache:

```bash
HF_HOME=/path/to/huggingface/cache
TRANSFORMERS_CACHE=/path/to/huggingface/cache
```

## Running Locally

Start the Streamlit app:

```bash
streamlit run app.py
```

Then:

1. Upload the PDFs in `sample_docs/`.
2. Click `Build / Rebuild Index`.
3. Ask one of the questions in `sample_questions.md`.
4. Review the answer, confidence score, citations, retrieved evidence, and source chunks.

The first run can take several minutes while Hugging Face downloads the embedding and generation models.

## Sample Workflow

1. Upload `sample_docs/paracetamol_usage.pdf`.
2. Upload `sample_docs/paracetamol_safety.pdf`.
3. Build the index.
4. Ask: `What safety risk is mentioned for excess paracetamol?`
5. Expected behavior: the answer should mention liver harm or overdose risk and include a citation from the safety PDF.

## Example Questions

See `sample_questions.md` for the full list. Examples:

- What is paracetamol commonly used for?
- What safety risk is mentioned for excess paracetamol?
- Why should someone avoid taking multiple medicines containing paracetamol?
- Compare the usage notes and safety notes for paracetamol.
- Does the document mention treatment for broken bones?

## Example Outputs

See `expected_outputs.md`.

Supported questions should produce a concise grounded answer with one or more inline citations. Unsupported questions should return:

```text
I could not find sufficient information in the provided documents to answer this question.

Reason:
No sufficiently relevant evidence was retrieved.
```

## Project Structure

```text
.
|-- app.py
|-- rag_pipeline.py
|-- utils.py
|-- requirements.txt
|-- .python-version
|-- .gitignore
|-- README.md
|-- ASSESSMENT.md
|-- LICENSE
|-- sample_questions.md
|-- expected_outputs.md
|-- sample_docs/
|   |-- paracetamol_usage.pdf
|   `-- paracetamol_safety.pdf
`-- tests/
    `-- test_multi_document_rag.py
```

## How Retrieval Works

Each uploaded PDF is loaded with `PyPDFLoader`, split into overlapping chunks with `RecursiveCharacterTextSplitter`, and tagged with filename, page number, chunk id, and document id. All chunks from active PDFs are embedded with MiniLM and stored in one FAISS index.

At question time, the pipeline retrieves candidate chunks, normalizes similarity scores, adds a small generic lexical overlap bonus, removes near-duplicate chunks, and selects chunks across documents where possible.

## How Citations Work

Every chunk receives a citation label:

```text
[filename.pdf | p.<page_number> | chunk_<id>]
```

The prompt tells the model to cite factual statements using the labels included in the retrieved context. After generation, the pipeline ensures uncited generated text receives fallback citations from the source chunks used by the QA chain.

## Hallucination Prevention

- The prompt instructs the model to answer only from retrieved context.
- Retrieval is assessed before generation.
- If no retrieved chunk exceeds the similarity threshold, the LLM is not called.
- The UI shows the exact evidence used so reviewers can inspect grounding.

## Design Decisions

- FAISS was chosen because it is lightweight, local, fast, and does not require a hosted vector database.
- `sentence-transformers/all-MiniLM-L6-v2` was chosen for a practical balance of retrieval quality, speed, and CPU usability.
- `google/flan-t5-base` was chosen because it can run locally and avoids API-key requirements during assessment.
- Streamlit was chosen because it provides a fast reviewable UI for uploading documents and inspecting evidence.

## Tradeoffs

- Local FLAN-T5 is smaller and less fluent than hosted frontier models, but it keeps the project self-contained.
- The confidence score measures retrieval strength, not absolute truth.
- Similarity threshold tuning is dataset-dependent.
- PDF extraction works best on text PDFs; scanned image-only PDFs require OCR, which is not implemented.
- FAISS indexes are held in memory for this assessment app instead of persisted to disk.

## Limitations

- Only PDF upload is supported.
- OCR is not implemented.
- Very large PDFs may be slow on CPU.
- The answer quality depends on extracted text quality and retrieved context.
- The app is intended for local assessment, not production multi-user deployment.

## Testing

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

Run static validation:

```bash
python -m compileall app.py rag_pipeline.py utils.py tests
python -m pip check
```

Manual app check:

```bash
streamlit run app.py
```

Upload both sample PDFs and verify that supported questions receive cited answers, while unsupported questions refuse.

## License

This repository is provided under the MIT License. See `LICENSE`.

## Reviewer Notes

- No external LLM API key is required.
- The first execution requires model downloads from Hugging Face unless models are already cached.
- Runtime uploads are stored under `data/uploads/` and are intentionally ignored by git.
- The repository should be evaluated from source files, sample documents, tests, and documentation only.
