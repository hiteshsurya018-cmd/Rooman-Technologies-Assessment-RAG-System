from __future__ import annotations

import json
import platform
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_pipeline import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DUPLICATE_TEXT_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    REFUSAL_ANSWER,
    RAGPipeline,
)


CASES = [
    {
        "id": "supported_01",
        "question": "What is paracetamol commonly used for?",
        "supported": True,
        "relevant_documents": ["paracetamol_usage.pdf"],
        "required_keyword_groups": [["pain"]],
    },
    {
        "id": "supported_02",
        "question": "What safety risk is mentioned for excess paracetamol?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["liver harm", "liver damage"]],
    },
    {
        "id": "supported_03",
        "question": "Who should seek medical advice before using paracetamol?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["liver disease"], ["heavy alcohol", "alcohol use"]],
    },
    {
        "id": "supported_04",
        "question": "What should a user do if an overdose is suspected?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["medical help", "medical attention"], ["immediately"]],
    },
    {
        "id": "supported_05",
        "question": "Why should someone avoid taking multiple medicines containing paracetamol?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["avoid"], ["multiple medicines", "more than one medicine"]],
    },
    {
        "id": "supported_06",
        "question": "Compare the usage notes and safety notes for paracetamol.",
        "supported": True,
        "relevant_documents": ["paracetamol_usage.pdf", "paracetamol_safety.pdf"],
        "required_keyword_groups": [["pain", "fever"], ["liver", "overdose", "multiple medicines"]],
    },
    {
        "id": "supported_07",
        "question": "What warning is given about liver harm?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["too much", "excess", "overdose"], ["liver harm", "liver damage"]],
    },
    {
        "id": "supported_08",
        "question": "What do the documents say about fever?",
        "supported": True,
        "relevant_documents": ["paracetamol_usage.pdf"],
        "required_keyword_groups": [["reduce fever", "fever"], ["directed", "label"]],
    },
    {
        "id": "supported_09",
        "question": "What should users do before combining medicines?",
        "supported": True,
        "relevant_documents": ["paracetamol_safety.pdf"],
        "required_keyword_groups": [["avoid"], ["contain paracetamol", "containing paracetamol"]],
    },
    {
        "id": "unsupported_01",
        "question": "Does the document mention treatment for broken bones?",
        "supported": False,
        "relevant_documents": [],
        "required_keyword_groups": [],
    },
    {
        "id": "unsupported_02",
        "question": "What is the recommended treatment for malaria?",
        "supported": False,
        "relevant_documents": [],
        "required_keyword_groups": [],
    },
    {
        "id": "unsupported_03",
        "question": "qxzv blorf nnn",
        "supported": False,
        "relevant_documents": [],
        "required_keyword_groups": [],
    },
]


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def cited_filenames(answer: str) -> list[str]:
    return sorted(set(re.findall(r"\[([^\]|]+\.pdf)\s*\|", answer)))


def portable_chunk(chunk: dict) -> dict:
    """Remove machine-specific absolute paths from serialized evaluation evidence."""
    cleaned = {**chunk, "metadata": dict(chunk["metadata"])}
    filename = cleaned["metadata"].get("filename")
    if filename:
        cleaned["metadata"]["source"] = f"sample_docs/{filename}"
    return cleaned


def main() -> None:
    output_path = ROOT / "evaluation" / "actual_model_evaluation.json"
    pdf_paths = sorted((ROOT / "sample_docs").glob("*.pdf"))
    started = perf_counter()
    pipeline = RAGPipeline()
    ingestion = pipeline.ingest_pdfs(pdf_paths)

    results = []
    for case in CASES:
        case_started = perf_counter()
        response = pipeline.answer_question(case["question"])
        answer = response["answer"]
        retrieved_filenames = [
            chunk["metadata"]["filename"] for chunk in response["retrieved_chunks"]
        ]
        cited = cited_filenames(answer)
        expected = set(case["relevant_documents"])
        retrieved = set(retrieved_filenames)
        citations = set(cited)

        if case["supported"]:
            group_matches = [
                any(keyword.lower() in answer.lower() for keyword in group)
                for group in case["required_keyword_groups"]
            ]
            answer_correct = all(group_matches) and REFUSAL_ANSWER.lower() not in answer.lower()
            retrieval_hit = expected.issubset(retrieved)
            citation_valid = bool(citations) and citations.issubset(retrieved) and expected.issubset(citations)
            grounded = answer_correct and retrieval_hit and citation_valid
            refusal_correct = None
        else:
            group_matches = []
            answer_correct = REFUSAL_ANSWER.lower() in answer.lower() and not response["llm_called"]
            retrieval_hit = None
            citation_valid = not citations
            grounded = answer_correct and citation_valid
            refusal_correct = answer_correct

        results.append(
            {
                **case,
                "relevant_passages": [
                    {"filename": filename, "chunk_id": "chunk_001"}
                    for filename in case["relevant_documents"]
                ],
                "answer": answer,
                "answer_correct": answer_correct,
                "keyword_group_matches": group_matches,
                "retrieval_hit": retrieval_hit,
                "grounded": grounded,
                "citation_valid": citation_valid,
                "cited_documents": cited,
                "refusal_correct": refusal_correct,
                "llm_called": response["llm_called"],
                "confidence": response["confidence"],
                "retrieved_chunks": [portable_chunk(chunk) for chunk in response["retrieved_chunks"]],
                "source_documents": [portable_chunk(chunk) for chunk in response["source_documents"]],
                "latency_seconds": round(perf_counter() - case_started, 3),
            }
        )

    supported = [item for item in results if item["supported"]]
    unsupported = [item for item in results if not item["supported"]]
    report = {
        "report_type": "actual-model RAG evaluation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_size": {
            "questions": len(results),
            "supported_questions": len(supported),
            "unsupported_questions": len(unsupported),
            "documents": len(pdf_paths),
            "pages": ingestion["num_pages"],
            "indexed_chunks": ingestion["num_chunks"],
        },
        "models_and_settings": {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "generation_model": DEFAULT_GENERATION_MODEL,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "top_k": DEFAULT_TOP_K,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "duplicate_text_threshold": DEFAULT_DUPLICATE_TEXT_THRESHOLD,
            "generation_max_new_tokens": 256,
            "generation_do_sample": False,
            "generation_temperature": 0.0,
            "device": "cpu",
        },
        "scoring_methodology": {
            "relevance_labels": "Manually assigned expected passage(s) per supported question as filename + chunk_id. Each short PDF yields exactly one passage (chunk_001), so passage-level and document-level recall are identical in this dataset; all expected passages/documents must occur in the top-k retrieved chunks.",
            "retrieval_metric": "Document-level Recall@5 (strict all-relevant-documents hit per question), measured only on supported questions.",
            "answer_correctness": "Deterministic required-keyword-group rubric declared in this script; every group must match and the response must not be a refusal.",
            "grounding": "A supported answer passes when it is correct, all labelled documents were retrieved, and citations name all labelled documents while naming no un-retrieved document.",
            "unsupported_handling": "Pass requires the canonical refusal text and llm_called=false.",
        },
        "aggregate_metrics": {
            "retrieval_document_recall_at_5": ratio(sum(bool(x["retrieval_hit"]) for x in supported), len(supported)),
            "supported_answer_accuracy": ratio(sum(x["answer_correct"] for x in supported), len(supported)),
            "supported_grounding_rate": ratio(sum(x["grounded"] for x in supported), len(supported)),
            "unsupported_refusal_accuracy": ratio(sum(bool(x["refusal_correct"]) for x in unsupported), len(unsupported)),
            "overall_case_pass_rate": ratio(sum(x["grounded"] for x in results), len(results)),
            "total_runtime_seconds": round(perf_counter() - started, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                name: version(name)
                for name in [
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "langchain",
                    "langchain-community",
                    "langchain-huggingface",
                    "faiss-cpu",
                    "pypdf",
                ]
            },
        },
        "limitations": [
            "The benchmark has only two short, single-page documents and twelve questions.",
            "Relevance is labelled at document level, not passage-span level, because each PDF produces one chunk with the default settings.",
            "Keyword scoring is deterministic and reproducible but is not a substitute for blinded human or semantic-judge evaluation.",
            "Citation validity checks provenance labels, not full natural-language entailment of every generated clause.",
            "The confidence value is the application's retrieval-strength heuristic, not calibrated answer probability.",
        ],
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["evaluation_size"], indent=2))
    print(json.dumps(report["aggregate_metrics"], indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
