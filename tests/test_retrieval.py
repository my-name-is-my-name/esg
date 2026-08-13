from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from esg.models import QueryExtraction, SearchSource, Zone, ZoneElement
from esg.retrieval import NEGATIVE_ANSWER, RetrievalService
from esg.storage import SQLiteStore


class RetrievalExtractionTests(unittest.TestCase):
    def test_retrieved_chunk_zones_are_cached_by_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = RetrievalService.__new__(RetrievalService)
            service.settings = SimpleNamespace(chunk_extraction_max_chars=5000)
            service.store = SQLiteStore(Path(directory) / "test.sqlite3")
            service.extractor = mock.Mock()
            service.extractor.extract_chunk_zones.return_value = [
                Zone(
                    elements=[ZoneElement(kind="rib", start=1, end=15, role="boundary")],
                    components=["skin"],
                    structure="wing",
                )
            ]
            row = {
                "record_id": "record-1",
                "heading_path": ["5", "Описание ремонта"],
                "evidence_text": "Обшивка крыла между нервюрами 1-15.",
            }

            first = service._zones_for_candidate(row)
            second = service._zones_for_candidate(row)

            self.assertEqual((first[1], first[2]), (0, 1))
            self.assertEqual((second[1], second[2]), (1, 0))
            self.assertEqual(second[0][0].element("rib").end, 15)
            service.extractor.extract_chunk_zones.assert_called_once()

    def test_negative_decision_is_canonical_and_has_no_supporting_sources(self) -> None:
        service = RetrievalService.__new__(RetrievalService)
        service.llm = mock.Mock(enabled=True)
        service.llm.json_completion.return_value = {
            "found": False,
            "answer": "Произвольный отрицательный текст.",
            "supporting_source_indexes": [1],
        }
        source = SearchSource(
            document_id="doc-1",
            source_path="report.docx",
            section_heading="Описание ремонта",
            evidence_text="Похожий фрагмент.",
        )

        answer, supporting, found = service._answer("question", QueryExtraction(), [source], [])

        self.assertEqual(answer, NEGATIVE_ANSWER)
        self.assertEqual(supporting, [])
        self.assertFalse(found)


if __name__ == "__main__":
    unittest.main()
