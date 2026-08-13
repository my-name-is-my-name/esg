from __future__ import annotations

import unittest
from types import SimpleNamespace

from esg.api import append_sources, handler_for, source_excerpt, source_windows_path


class FakeStore:
    @staticmethod
    def latest_job() -> dict[str, object] | None:
        return None


class FakeIngestion:
    @staticmethod
    def start(force: bool = False, refresh_sources: bool = False) -> dict[str, object]:
        return {
            "accepted": True, "job_id": "job-1", "status": "running",
            "force": force, "refresh_sources": refresh_sources,
        }

    @staticmethod
    def source_status() -> dict[str, object]:
        return {"status": "not_refreshed", "catalog_available": False, "scheduled": False}


class FakeRetrieval:
    @staticmethod
    def chat(question: str) -> dict[str, object]:
        return {
            "answer": "В проиндексированных документах подтверждающий ремонт не найден.",
            "sources": [],
            "warnings": [],
            "reasoning": ["Выполняю поиск.", "Проверяю источники."],
        }


class FakeApp:
    settings = SimpleNamespace(model_id="esg-repair-search")
    store = FakeStore()
    ingestion = FakeIngestion()
    retrieval = FakeRetrieval()

    @staticmethod
    def health() -> dict[str, object]:
        return {"ok": True}


class ApiContractTests(unittest.TestCase):
    def test_source_windows_path_targets_original_docx(self) -> None:
        path = source_windows_path(
            "20240322/02_Внутренние_данные/report TR.docx",
            "P:/WP013C_RE/ECAR_Stress/Deliverables/01_A320CEO_MSN2947",
        )
        self.assertEqual(
            path,
            "P:\\WP013C_RE\\ECAR_Stress\\Deliverables\\01_A320CEO_MSN2947\\"
            "20240322\\02_Внутренние_данные\\report TR.docx",
        )

    def test_source_windows_path_uses_prefixed_root_mapping(self) -> None:
        path = source_windows_path(
            "__s7a/folder/report.docx",
            "P:/WP013C_RE/ECAR_Stress/Deliverables",
            (("__s7a", "//ru0-archive05/archive/WP_Archives/WP163_S7/ECAR_Stress/Deliverables/02_S7A"),),
        )
        self.assertEqual(
            path,
            "\\\\ru0-archive05\\archive\\WP_Archives\\WP163_S7\\ECAR_Stress\\Deliverables\\02_S7A\\folder\\report.docx",
        )

    def test_source_excerpt_is_single_line_and_bounded(self) -> None:
        self.assertEqual(source_excerpt(" first\n second ", limit=20), "first second")
        self.assertEqual(source_excerpt("123456789", limit=6), "12345…")

    def test_handler_exposes_expected_server_identity(self) -> None:
        handler = handler_for(FakeApp())
        self.assertEqual(handler.server_version, "ESGRepairSearch/0.1")
        self.assertTrue(callable(handler.do_GET))
        self.assertTrue(callable(handler.do_POST))

    def test_answer_without_sources_is_unchanged(self) -> None:
        answer = "В проиндексированных документах подтверждающий ремонт не найден."
        self.assertEqual(append_sources(answer, []), answer)

    def test_negative_answer_never_renders_source_table(self) -> None:
        answer = "В проиндексированных документах подтверждающий ремонт не найден."
        source = {
            "source_path": "folder/report.docx",
            "section_heading": "Описание ремонта",
            "evidence_text": "Похожий, но не подтверждающий фрагмент.",
        }

        self.assertEqual(append_sources(answer, [source]), answer)

    def test_debug_mode_renders_sources_for_negative_answer(self) -> None:
        answer = "В проиндексированных документах подтверждающий ремонт не найден."
        source = {
            "source_path": "folder/report.docx",
            "section_heading": "Описание ремонта",
            "evidence_text": "Кандидат retrieval.",
        }

        rendered = append_sources(answer, [source], show_on_negative=True)

        self.assertIn("Источники:", rendered)
        self.assertIn("Кандидат retrieval.", rendered)

    def test_source_table_is_deduplicated_and_escaped(self) -> None:
        source = {
            "source_path": "folder/report|A.pdf",
            "section_heading": "5. Оценка|ремонта",
            "evidence_text": "Трещина | между стрингерами",
            "zone": {"zone_text": "FR34 | STGR24-28"},
        }
        rendered = append_sources("Да.", [source, source])
        self.assertEqual(rendered.count("report\\|A.pdf"), 1)
        self.assertIn("Трещина \\| между стрингерами", rendered)

    def test_source_table_deduplicates_word_anchor_variants(self) -> None:
        first = {
            "source_path": "20230419/report.docx",
            "section_heading": "Описание ремонта",
            "evidence_text": 'Ремонт. <a id="_Toc100"></a> Заключение.',
        }
        second = {
            "source_path": "20230427/report.docx",
            "section_heading": "Описание ремонта",
            "evidence_text": 'Ремонт. <a id="_Toc200"></a> Заключение.',
        }
        rendered = append_sources("Да.", [first, second], "P:/root")
        self.assertEqual(rendered.count("Описание ремонта"), 1)


if __name__ == "__main__":
    unittest.main()
