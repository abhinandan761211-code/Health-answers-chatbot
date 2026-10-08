import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge_base import ChromaKnowledgeBase
from knowledge_updater import (
    KnowledgeUpdater,
    SourcePage,
    load_source_urls,
    validate_source_url,
)


class InMemoryCollection:
    def __init__(self) -> None:
        self.documents: dict[str, str] = {}
        self.metadatas: dict[str, dict[str, object]] = {}

    def count(self) -> int:
        return len(self.documents)

    def get(self, where=None, include=None) -> dict[str, list[str]]:
        del include
        matching_ids = [
            identifier
            for identifier, metadata in self.metadatas.items()
            if where is None
            or all(metadata.get(key) == value for key, value in where.items())
        ]
        return {"ids": matching_ids}

    def upsert(self, ids, documents, metadatas) -> None:
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs in upsert batch")
        for identifier, document, metadata in zip(ids, documents, metadatas):
            self.documents[identifier] = document
            self.metadatas[identifier] = metadata

    def delete(self, ids) -> None:
        for identifier in ids:
            self.documents.pop(identifier, None)
            self.metadatas.pop(identifier, None)

    def query(self, query_texts, n_results, include) -> dict[str, list[list[object]]]:
        del query_texts, include
        ids = list(self.documents)[:n_results]
        return {
            "documents": [[self.documents[identifier] for identifier in ids]],
            "metadatas": [[self.metadatas[identifier] for identifier in ids]],
            "distances": [[0.2 for _ in ids]],
        }


class KnowledgeBaseUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        (self.dataset_dir / "sample.xml").write_text(
            """<Document source="GARD">
              <Focus>Asthma</Focus>
              <QAPairs><QAPair>
                <Question>What are asthma symptoms?</Question>
                <Answer>Asthma may cause wheezing and shortness of breath.</Answer>
              </QAPair></QAPairs>
            </Document>""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_medquad_is_seeded_once_and_source_chunks_are_replaced(self) -> None:
        collection = InMemoryCollection()
        knowledge_base = ChromaKnowledgeBase(
            self.dataset_dir,
            self.root / "chroma",
            collection=collection,
        )
        self.assertEqual(knowledge_base.count, 1)
        self.assertEqual(knowledge_base.index_medquad(), 0)

        source_url = "https://medlineplus.gov/healthtopics.html"
        initial_text = "Trusted health information " * 10
        self.assertEqual(
            knowledge_base.replace_source(
                source_url,
                "medlineplus.gov",
                "Health topics",
                initial_text,
                chunk_size=20,
            ),
            len(initial_text) // 20 + (len(initial_text) % 20 > 0),
        )
        self.assertGreater(knowledge_base.count, 2)

        knowledge_base.replace_source(
            source_url,
            "medlineplus.gov",
            "Health topics",
            "Updated trusted health information.",
            chunk_size=200,
        )
        source_results = [
            metadata
            for metadata in collection.metadatas.values()
            if metadata.get("kind") == "trusted_source"
        ]
        self.assertEqual(len(source_results), 1)
        self.assertEqual(knowledge_base.count, 2)

        results = knowledge_base.search("trusted health information")
        self.assertIn("medlineplus.gov", {result.record.source for result in results})

    def test_medquad_indexing_uses_unique_ids_for_duplicate_questions(self) -> None:
        (self.dataset_dir / "sample.xml").write_text(
            """<Document source="GARD" url="https://example.org/asthma">
              <Focus>Asthma</Focus>
              <FocusAnnotations><Category>Disease</Category></FocusAnnotations>
              <QAPairs>
                <QAPair>
                  <Question qtype="overview">What is asthma?</Question>
                  <Answer>Asthma affects the airways.</Answer>
                </QAPair>
                <QAPair>
                  <Question qtype="symptoms">What is asthma?</Question>
                  <Answer>Asthma may cause wheezing.</Answer>
                </QAPair>
                <QAPair>
                  <Question qtype="overview">What is asthma?</Question>
                  <Answer>Asthma affects the airways.</Answer>
                </QAPair>
              </QAPairs>
            </Document>""",
            encoding="utf-8",
        )
        collection = InMemoryCollection()
        knowledge_base = ChromaKnowledgeBase(
            self.dataset_dir,
            self.root / "chroma",
            collection=collection,
        )
        self.assertEqual(knowledge_base.count, 2)
        self.assertEqual(len(collection.documents), 2)

    def test_medquad_indexing_removes_stale_legacy_ids_only(self) -> None:
        collection = InMemoryCollection()
        collection.upsert(
            ids=["medquad:legacy", "source:keep"],
            documents=["Old MedQuAD row", "Trusted source"],
            metadatas=[
                {"kind": "medquad"},
                {"kind": "trusted_source", "source_url": "https://medlineplus.gov/page"},
            ],
        )
        knowledge_base = ChromaKnowledgeBase(
            self.dataset_dir,
            self.root / "chroma",
            collection=collection,
        )
        self.assertNotIn("medquad:legacy", collection.documents)
        self.assertIn("source:keep", collection.documents)
        self.assertEqual(knowledge_base.count, 2)

    def test_source_refresh_indexes_configured_page_and_reports_failures(self) -> None:
        class DummyKnowledgeBase:
            def __init__(self) -> None:
                self.sources: dict[str, str] = {}

            def replace_source(self, source_url, source_name, title, text):
                self.sources[source_url] = f"{source_name}: {title} - {text}"
                return 2

        source_url = "https://www.cdc.gov/health-topics/index.html"
        knowledge_base = DummyKnowledgeBase()
        updater = KnowledgeUpdater(knowledge_base, [source_url])
        page = SourcePage(source_url, "Health topics", "A trusted medical page.")
        with patch("knowledge_updater.fetch_source_page", return_value=page):
            result = updater.refresh()

        self.assertEqual(result.updated_sources, 1)
        self.assertEqual(result.indexed_chunks, 2)
        self.assertIn(page.text, knowledge_base.sources[source_url])

        with patch(
            "knowledge_updater.fetch_source_page",
            side_effect=RuntimeError("network unavailable"),
        ):
            failed_result = updater.refresh()
        self.assertEqual(failed_result.updated_sources, 0)
        self.assertEqual(len(failed_result.errors), 1)

    def test_source_list_validates_and_deduplicates_urls(self) -> None:
        source_file = self.root / "sources.txt"
        source_file.write_text(
            "# trusted pages\n"
            "https://medlineplus.gov/healthtopics.html\n"
            "https://medlineplus.gov/healthtopics.html\n",
            encoding="utf-8",
        )
        self.assertEqual(
            load_source_urls(source_file),
            ["https://medlineplus.gov/healthtopics.html"],
        )
        for url in (
            "http://medlineplus.gov/",
            "https://medlineplus.gov.attacker.example/",
            "https://example.org/",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_source_url(url)

        source_file.write_text("https://example.org/health\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sources.txt:1"):
            load_source_urls(source_file)


if __name__ == "__main__":
    unittest.main()
