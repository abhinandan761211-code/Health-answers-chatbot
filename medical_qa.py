"""Local MedQuAD loading, retrieval, and lightweight medical entity recognition."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class QARecord:
    question: str
    answer: str
    focus: str
    question_type: str
    source: str
    url: str
    focus_category: str
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    record: QARecord
    score: float


SYMPTOM_TERMS = (
    "abdominal pain", "back pain", "chest pain", "difficulty breathing",
    "shortness of breath", "sore throat", "high blood pressure", "low blood pressure",
    "blurred vision", "blood in urine", "blood in stool", "weight loss",
    "weight gain", "loss of appetite", "hearing loss", "memory loss",
    "muscle weakness", "joint pain", "stomach pain", "runny nose",
    "nasal congestion", "night sweats", "hair loss", "skin rash",
    "nausea", "vomiting", "diarrhea", "constipation", "fever", "cough",
    "fatigue", "tiredness", "headache", "dizziness", "weakness", "pain",
    "swelling", "itching", "rash", "bleeding", "seizures", "tremor",
)

TREATMENT_TERMS = (
    "physical therapy", "radiation therapy", "occupational therapy",
    "chemotherapy", "medication", "medications", "treatment", "treatments",
    "therapy", "therapies", "surgery", "surgical", "vaccine", "vaccination",
    "antibiotic", "antibiotics", "pain relief", "drug", "drugs", "medicine",
    "medicines", "insulin", "dialysis", "transplant", "rehabilitation",
)


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def load_medquad(dataset_dir: Path) -> list[QARecord]:
    """Read answerable MedQuAD question-answer pairs from a dataset directory."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"MedQuAD dataset directory not found: {dataset_dir}")

    records: list[QARecord] = []
    for path in sorted(dataset_dir.rglob("*.xml")):
        root = ET.parse(path).getroot()
        focus = _text(root.find("Focus"))
        source = root.get("source", path.parent.name)
        url = root.get("url", "")
        category = _text(root.find("./FocusAnnotations/Category"))
        synonyms = tuple(
            text for element in root.findall("./FocusAnnotations/Synonyms/Synonym")
            if (text := _text(element))
        )

        for pair in root.findall(".//QAPair"):
            question = _text(pair.find("Question"))
            answer = _text(pair.find("Answer"))
            if not question or not answer:
                continue
            question_element = pair.find("Question")
            records.append(
                QARecord(
                    question=question,
                    answer=answer,
                    focus=focus,
                    question_type=(
                        question_element.get("qtype", "") if question_element is not None else ""
                    ),
                    source=source,
                    url=url,
                    focus_category=category,
                    synonyms=synonyms,
                )
            )

    if not records:
        raise ValueError(f"No answerable MedQuAD question-answer pairs found in {dataset_dir}")
    return records


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


class MedicalRetriever:
    """TF-IDF question retrieval over the answerable MedQuAD records."""

    def __init__(self, records: list[QARecord]) -> None:
        if not records:
            raise ValueError("At least one answerable record is required to build the retriever.")
        self.records = records
        corpus = [
            " ".join(
                (record.question, record.focus, " ".join(record.synonyms), record.answer)
            )
            for record in records
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
            max_features=150_000,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        self._entity_phrases = self._build_entity_phrases(records)

    def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        """Return the highest-scoring answerable records for a question."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not question.strip():
            return []

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        return [
            SearchResult(self.records[index], float(scores[index]))
            for index in ranked[:top_k]
            if scores[index] > 0
        ]

    @staticmethod
    def _build_entity_phrases(records: list[QARecord]) -> dict[tuple[str, ...], set[str]]:
        phrases: dict[tuple[str, ...], set[str]] = {}
        for term in SYMPTOM_TERMS:
            phrases.setdefault(_tokens(term), set()).add("Symptom")
        for term in TREATMENT_TERMS:
            phrases.setdefault(_tokens(term), set()).add("Treatment")
        for record in records:
            category = record.focus_category.casefold()
            # Non-MedlinePlus MedQuAD collections primarily describe diseases.
            is_disease = category == "disease" or (
                not category and record.source.casefold() not in {
                    "mplusdruginfo", "mplusherbsandsupplements", "mplushealthtopics"
                }
            )
            if is_disease:
                for term in (record.focus, *record.synonyms):
                    normalized = _tokens(term)
                    if normalized and not phrases.get(normalized, set()).intersection(
                        {"Symptom", "Treatment"}
                    ):
                        phrases.setdefault(normalized, set()).add("Disease")
        return phrases

    def recognize_entities(self, text: str) -> dict[str, list[str]]:
        """Find known dataset diseases and common symptom/treatment terms."""
        words = _tokens(text)
        found: dict[str, set[str]] = {
            "Disease": set(),
            "Symptom": set(),
            "Treatment": set(),
        }
        max_phrase_length = max((len(phrase) for phrase in self._entity_phrases), default=0)
        for start in range(len(words)):
            for end in range(start + 1, min(len(words), start + max_phrase_length) + 1):
                phrase = words[start:end]
                categories = self._entity_phrases.get(phrase)
                if categories:
                    display = " ".join(phrase)
                    for category in categories:
                        found[category].add(display)
        return {category: sorted(terms) for category, terms in found.items() if terms}
