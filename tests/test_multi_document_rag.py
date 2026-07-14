from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

from rag_pipeline import RAGPipeline
from utils import build_document_set_id


class KeywordEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("aspirin")),
            float(lowered.count("ibuprofen")),
            float(lowered.count("paracetamol")),
        ]


class FixedLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "fixed_test_llm"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        return "Aspirin guidance and ibuprofen guidance are both relevant."


def write_text_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    objects.append(
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )

    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")

    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(data))


class MultiDocumentRAGTest(TestCase):
    def test_multi_document_index_retrieves_and_cites_multiple_pdfs(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            aspirin_pdf = tmp_path / "aspirin.pdf"
            ibuprofen_pdf = tmp_path / "ibuprofen.pdf"
            write_text_pdf(
                aspirin_pdf,
                "Aspirin guidance says adults should follow label directions.",
            )
            write_text_pdf(
                ibuprofen_pdf,
                "Ibuprofen guidance says people should consider stomach warnings.",
            )

            with (
                patch.object(RAGPipeline, "_build_embeddings", lambda self: KeywordEmbeddings()),
                patch.object(RAGPipeline, "_build_llm", lambda self: FixedLLM()),
            ):
                pipeline = RAGPipeline(chunk_size=400, chunk_overlap=50, top_k=2)
                ingestion = pipeline.ingest_pdfs([aspirin_pdf, ibuprofen_pdf])
                result = pipeline.answer_question("Compare the aspirin and ibuprofen guidance.")

            self.assertEqual(len(ingestion["documents"]), 2)
            self.assertEqual(ingestion["num_chunks"], 2)
            self.assertIsNotNone(pipeline.vectorstore)

            retrieved_filenames = {
                chunk["metadata"]["filename"] for chunk in result["retrieved_chunks"]
            }
            source_filenames = {
                chunk["metadata"]["filename"] for chunk in result["source_documents"]
            }

            self.assertEqual(retrieved_filenames, {"aspirin.pdf", "ibuprofen.pdf"})
            self.assertEqual(source_filenames, {"aspirin.pdf", "ibuprofen.pdf"})
            self.assertIn("[aspirin.pdf | p.1 | chunk_001]", result["answer"])
            self.assertIn("[ibuprofen.pdf | p.1 | chunk_001]", result["answer"])

            for chunk in result["retrieved_chunks"]:
                self.assertTrue(chunk["metadata"]["filename"])
                self.assertEqual(chunk["metadata"]["page_number"], 1)
                self.assertEqual(chunk["metadata"]["chunk_id"], "chunk_001")
                self.assertTrue(chunk["metadata"]["document_id"])

            self.assertEqual(
                build_document_set_id([aspirin_pdf, ibuprofen_pdf], 400, 50, top_k=2),
                build_document_set_id([aspirin_pdf, ibuprofen_pdf], 400, 50, top_k=5),
            )
