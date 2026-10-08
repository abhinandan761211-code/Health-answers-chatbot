"""Persistent ChromaDB vector search over MedQuAD and trusted-source pages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from medical_qa import (
    MedicalEntityRecognizer,
    QARecord,
    SearchResult,
    load_medquad,
)


class ChromaKnowledgeBase:
    """Store searchable medical content in a persistent Chroma collection."""

    def __init__(
        self,
        dataset_dir: Path,
        persist_directory: Path,
        collection: Any | None = None,
        embedding_function: Any | None = None,
    ) -> None:
        if collection is None:
            import chromadb
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            persist_directory.mkdir(parents=True, exist_ok=True)
            embedding_function = embedding_function or SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            client = chromadb.PersistentClient(path=str(persist_directory))
            collection = client.get_or_create_collection(
                name="medical_knowledge",
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        if collection is None:
            raise RuntimeError("Unable to initialize the ChromaDB collection.")
        self.collection = collection
        self.records = load_medquad(dataset_dir)
        self.entity_recognizer = MedicalEntityRecognizer(self.records)
        self.index_medquad()

    @property
    def count(self) -> int:
        return self.collection.count()

    def index_medquad(self, batch_size: int = 128) -> int:
        """Ensure all MedQuAD records exist without re-embedding indexed items."""
        existing = self.collection.get(where={"kind": "medquad"}, include=[])
        indexed_ids = set(existing["ids"])
        records_by_id: dict[str, QARecord] = {}
        for record in self.records:
            records_by_id.setdefault(_medquad_id(record), record)

        stale_ids = sorted(indexed_ids - records_by_id.keys())
        for offset in range(0, len(stale_ids), batch_size):
            self.collection.delete(ids=stale_ids[offset : offset + batch_size])

        missing = [
            (identifier, record)
            for identifier, record in records_by_id.items()
            if identifier not in indexed_ids
        ]

        for offset in range(0, len(missing), batch_size):
            batch = missing[offset : offset + batch_size]
            self.collection.upsert(
                ids=[identifier for identifier, _ in batch],
                documents=[_medquad_document(record) for _, record in batch],
                metadatas=[_medquad_metadata(record) for _, record in batch],
            )
        return len(missing)

    def replace_source(
        self,
        source_url: str,
        source_name: str,
        title: str,
        text: str,
        chunk_size: int = 1200,
    ) -> int:
        """Replace a source's old chunks with its latest extracted page text."""
        chunks = _chunk_text(text, chunk_size)
        if not chunks:
            raise ValueError(f"Trusted source contains no indexable text: {source_url}")

        ids = [
            f"source:{hashlib.sha256(source_url.encode('utf-8')).hexdigest()}:{index}"
            for index in range(len(chunks))
        ]
        metadata = [
            {
                "kind": "trusted_source",
                "source_url": source_url,
                "source": source_name,
                "title": title,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]
        self.collection.upsert(ids=ids, documents=chunks, metadatas=metadata)

        previous = self.collection.get(
            where={"source_url": source_url},
            include=[],
        )
        stale_ids = sorted(set(previous["ids"]) - set(ids))
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(chunks)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not question.strip():
            return []
        matches = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]
        results: list[SearchResult] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            if score <= 0:
                continue
            if metadata["kind"] == "medquad":
                record = QARecord(
                    question=metadata["question"],
                    answer=metadata["answer"],
                    focus=metadata["focus"],
                    question_type=metadata["question_type"],
                    source=metadata["source"],
                    url=metadata["url"],
                    focus_category=metadata["focus_category"],
                )
            else:
                record = QARecord(
                    question=metadata["title"] or metadata["source"],
                    answer=document,
                    focus="",
                    question_type="",
                    source=metadata["source"],
                    url=metadata["source_url"],
                    focus_category="",
                )
            results.append(SearchResult(record, score))
        return results

    def recognize_entities(self, text: str) -> dict[str, list[str]]:
        return self.entity_recognizer.recognize_entities(text)


def _medquad_id(record: QARecord) -> str:
    identity = "\0".join(
        (
            record.source,
            record.url,
            record.question,
            record.answer,
            record.focus,
            record.question_type,
            record.focus_category,
            *record.synonyms,
        )
    )
    return f"medquad:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


def _medquad_document(record: QARecord) -> str:
    return f"Question: {record.question}\nAnswer: {record.answer}"


def _medquad_metadata(record: QARecord) -> dict[str, str]:
    return {
        "kind": "medquad",
        "question": record.question,
        "answer": record.answer,
        "focus": record.focus,
        "question_type": record.question_type,
        "source": record.source,
        "url": record.url,
        "focus_category": record.focus_category,
    }


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    normalized = " ".join(text.split())
    return [
        normalized[start : start + chunk_size].strip()
        for start in range(0, len(normalized), chunk_size)
        if normalized[start : start + chunk_size].strip()
    ]
