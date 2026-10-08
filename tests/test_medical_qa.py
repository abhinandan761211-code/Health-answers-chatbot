import tempfile
import unittest
from pathlib import Path

from medical_qa import MedicalRetriever, load_medquad


class MedicalQATest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_dir = Path(self.temp_dir.name)
        (self.dataset_dir / "sample.xml").write_text(
            """<?xml version="1.0"?>
            <Document source="GARD" url="https://example.org/asthma">
              <Focus>Asthma</Focus>
              <FocusAnnotations>
                <Category>Disease</Category>
                <Synonyms><Synonym>bronchial asthma</Synonym></Synonyms>
              </FocusAnnotations>
              <QAPairs>
                <QAPair pid="1">
                  <Question qid="q1" qtype="symptoms">What are asthma symptoms?</Question>
                  <Answer>Asthma may cause wheezing and shortness of breath.</Answer>
                </QAPair>
                <QAPair pid="2">
                  <Question qid="q2" qtype="treatment">How is asthma treated?</Question>
                  <Answer>Treatment may include inhaled medication.</Answer>
                </QAPair>
                <QAPair pid="3">
                  <Question qid="q3">Question with no answer</Question>
                  <Answer></Answer>
                </QAPair>
              </QAPairs>
            </Document>""",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_loader_keeps_answerable_records_and_metadata(self) -> None:
        records = load_medquad(self.dataset_dir)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].focus, "Asthma")
        self.assertEqual(records[0].question_type, "symptoms")
        self.assertEqual(records[0].synonyms, ("bronchial asthma",))

    def test_retrieval_ranks_relevant_answer(self) -> None:
        retriever = MedicalRetriever(load_medquad(self.dataset_dir))
        results = retriever.search("What symptoms happen with asthma?")
        self.assertTrue(results)
        self.assertIn("wheezing", results[0].record.answer)
        self.assertEqual(retriever.search("volcano astronomy"), [])

    def test_entity_recognition_finds_disease_symptom_and_treatment(self) -> None:
        retriever = MedicalRetriever(load_medquad(self.dataset_dir))
        entities = retriever.recognize_entities(
            "Does bronchial asthma cause shortness of breath, and is medication a treatment?"
        )
        self.assertIn("Disease", entities)
        self.assertIn("Symptom", entities)
        self.assertIn("Treatment", entities)
        self.assertIn("asthma", entities["Disease"])
        self.assertIn("shortness of breath", entities["Symptom"])
        self.assertIn("medication", entities["Treatment"])
        self.assertNotIn("cough", entities["Disease"])

    def test_empty_query_and_invalid_limit(self) -> None:
        retriever = MedicalRetriever(load_medquad(self.dataset_dir))
        self.assertEqual(retriever.search("   "), [])
        with self.assertRaises(ValueError):
            retriever.search("asthma", top_k=0)

    def test_missing_dataset_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_medquad(self.dataset_dir / "missing")


if __name__ == "__main__":
    unittest.main()
