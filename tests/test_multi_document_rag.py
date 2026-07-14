from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document

from rag_pipeline import REFUSAL_ANSWER, REFUSAL_REASON, RAGPipeline, RetrievalChunk
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


class CountingLLM(LLM):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "counting_test_llm"

    def _call(self, prompt: str, stop: list[str] | None = None, **kwargs) -> str:
        self.calls += 1
        return "Grounded test answer."


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
    def build_test_pipeline(
        self,
        pdfs: list[Path],
        *,
        top_k: int = 2,
        similarity_threshold: float = 0.5,
        llm: LLM | None = None,
    ) -> tuple[RAGPipeline, LLM, dict]:
        test_llm = llm or FixedLLM()
        with (
            patch.object(RAGPipeline, "_build_embeddings", lambda self: KeywordEmbeddings()),
            patch.object(RAGPipeline, "_build_llm", lambda self: test_llm),
        ):
            pipeline = RAGPipeline(
                chunk_size=400,
                chunk_overlap=50,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            ingestion = pipeline.ingest_pdfs(pdfs)
        return pipeline, test_llm, ingestion

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

            pipeline, _, ingestion = self.build_test_pipeline(
                [aspirin_pdf, ibuprofen_pdf],
                top_k=2,
                similarity_threshold=0.49,
            )
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
                build_document_set_id([aspirin_pdf, ibuprofen_pdf], 400, 50),
                build_document_set_id([aspirin_pdf, ibuprofen_pdf], 400, 50),
            )

    def test_no_answer_threshold_skips_llm_for_unrelated_question(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "aspirin.pdf"
            write_text_pdf(pdf_path, "Aspirin guidance says adults should follow label directions.")
            counting_llm = CountingLLM()
            pipeline, llm, _ = self.build_test_pipeline(
                [pdf_path],
                similarity_threshold=0.75,
                llm=counting_llm,
            )

            result = pipeline.answer_question("zebra astronomy")

            self.assertEqual(llm.calls, 0)
            self.assertFalse(result["llm_called"])
            self.assertIn(REFUSAL_ANSWER, result["answer"])
            self.assertIn(f"Reason:\n{REFUSAL_REASON}", result["answer"])
            self.assertEqual(result["reason"], REFUSAL_REASON)

    def test_nonsense_question_uses_no_answer_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "aspirin.pdf"
            write_text_pdf(pdf_path, "Aspirin guidance says adults should follow label directions.")
            counting_llm = CountingLLM()
            pipeline, llm, _ = self.build_test_pipeline(
                [pdf_path],
                similarity_threshold=0.75,
                llm=counting_llm,
            )

            result = pipeline.answer_question("qxzv blorf nnn")

            self.assertEqual(llm.calls, 0)
            self.assertFalse(result["llm_called"])
            self.assertEqual(result["reason"], REFUSAL_REASON)

    def test_empty_question_uses_no_answer_path(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "aspirin.pdf"
            write_text_pdf(pdf_path, "Aspirin guidance says adults should follow label directions.")
            counting_llm = CountingLLM()
            pipeline, llm, _ = self.build_test_pipeline(
                [pdf_path],
                similarity_threshold=0.75,
                llm=counting_llm,
            )

            result = pipeline.answer_question("   ")

            self.assertEqual(llm.calls, 0)
            self.assertEqual(result["confidence"], 0)
            self.assertFalse(result["retrieved_chunks"])
            self.assertFalse(result["source_documents"])

    def test_duplicate_chunk_removal_keeps_highest_scoring_chunk(self) -> None:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.duplicate_text_threshold = 0.95
        chunks = [
            RetrievalChunk(
                document=Document(
                    page_content="Aspirin guidance says adults should follow label directions.",
                    metadata={"chunk_id": "chunk_001"},
                ),
                score=0.91,
            ),
            RetrievalChunk(
                document=Document(
                    page_content="Aspirin guidance says adults should follow label directions.",
                    metadata={"chunk_id": "chunk_002"},
                ),
                score=0.82,
            ),
            RetrievalChunk(
                document=Document(
                    page_content="Ibuprofen guidance mentions stomach warnings.",
                    metadata={"chunk_id": "chunk_003"},
                ),
                score=0.77,
            ),
        ]

        unique_chunks = pipeline._remove_near_duplicate_chunks(chunks)

        self.assertEqual([chunk.document.metadata["chunk_id"] for chunk in unique_chunks], ["chunk_001", "chunk_003"])

    def test_confidence_score_calculation(self) -> None:
        chunks = [
            RetrievalChunk(document=Document(page_content="A"), score=0.90),
            RetrievalChunk(document=Document(page_content="B"), score=0.60),
        ]

        self.assertEqual(RAGPipeline.calculate_confidence(chunks), 86)
        self.assertEqual(RAGPipeline.calculate_confidence([]), 0)

    def test_evidence_ordering_by_similarity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            aspirin_pdf = Path(tmp_dir) / "aspirin.pdf"
            ibuprofen_pdf = Path(tmp_dir) / "ibuprofen.pdf"
            write_text_pdf(
                aspirin_pdf,
                "Aspirin aspirin guidance says adults should follow label directions.",
            )
            write_text_pdf(
                ibuprofen_pdf,
                "Ibuprofen guidance says people should consider stomach warnings.",
            )
            pipeline, _, _ = self.build_test_pipeline(
                [aspirin_pdf, ibuprofen_pdf],
                top_k=2,
                similarity_threshold=0.1,
            )

            evidence = pipeline.retrieve_chunks("aspirin")

            self.assertGreaterEqual(evidence[0].score, evidence[1].score)
            self.assertEqual(evidence[0].document.metadata["filename"], "aspirin.pdf")

    def test_lexical_bonus_is_generic_and_bounded(self) -> None:
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.indexed_chunks = [
            Document(
                page_content="Paracetamol may reduce fever and relieve mild pain.",
                metadata={"filename": "usage.pdf", "page_number": 1, "chunk_id": "chunk_001"},
            ),
            Document(
                page_content="Storage instructions say to keep medicines away from moisture.",
                metadata={"filename": "storage.pdf", "page_number": 1, "chunk_id": "chunk_001"},
            ),
        ]

        candidates = [
            RetrievalChunk(
                document=pipeline.indexed_chunks[1],
                score=0.30,
            )
        ]

        expanded = pipeline._add_lexical_candidates("What does paracetamol say about fever?", candidates)
        ranked = sorted(expanded, key=lambda item: item.score, reverse=True)

        self.assertEqual(ranked[0].document.metadata["filename"], "storage.pdf")
        self.assertTrue(
            any(item.document.metadata["filename"] == "usage.pdf" for item in expanded)
        )
        self.assertLessEqual(
            RAGPipeline._lexical_relevance_bonus(
                "What does paracetamol say about fever?",
                "Paracetamol may reduce fever and relieve mild pain.",
            ),
            0.20,
        )

    def test_contradictory_documents_remain_grounded_in_sources(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            safe_pdf = Path(tmp_dir) / "safe.pdf"
            warning_pdf = Path(tmp_dir) / "warning.pdf"
            write_text_pdf(safe_pdf, "Paracetamol guidance says this dose is safe for adults.")
            write_text_pdf(warning_pdf, "Paracetamol guidance says this dose is unsafe for children.")
            pipeline, _, _ = self.build_test_pipeline(
                [safe_pdf, warning_pdf],
                top_k=2,
                similarity_threshold=0.1,
            )

            result = pipeline.answer_question("What does paracetamol guidance say about this dose?")
            source_filenames = {
                chunk["metadata"]["filename"] for chunk in result["source_documents"]
            }

            self.assertTrue(result["llm_called"])
            self.assertEqual(source_filenames, {"safe.pdf", "warning.pdf"})
            self.assertIn("[safe.pdf | p.1 | chunk_001]", result["answer"])
            self.assertIn("[warning.pdf | p.1 | chunk_001]", result["answer"])
