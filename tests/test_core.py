from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esg.chunking import parse_markdown
from esg.ingest import is_chapter_five, sha256_file
from esg.matching import best_zone_match, compare_zones, ranges_compatible
from esg.models import NumericRange, Zone
from esg.retrieval import deduplicate_evidence, query_summary
from esg.storage import SQLiteStore


class ChunkingTests(unittest.TestCase):
    def test_heading_hierarchy_is_preserved(self) -> None:
        sections = parse_markdown(
            "# Отчет\nВведение.\n## 5. Оценка\nТекст раздела.\n### 5.1 Описание\nОписание ремонта.",
            "doc-1",
        )
        self.assertEqual([item.heading for item in sections], ["Отчет", "5. Оценка", "5.1 Описание"])
        self.assertEqual(sections[-1].heading_path, ["Отчет", "5. Оценка", "5.1 Описание"])


class MatchingTests(unittest.TestCase):
    def test_intersection_is_inclusive(self) -> None:
        query = NumericRange(start=24, end=28)
        self.assertTrue(ranges_compatible(query, 28, 31))
        self.assertTrue(ranges_compatible(query, 25, 27))
        self.assertFalse(ranges_compatible(query, 29, 31))

    def test_frame_mismatch_is_conflict(self) -> None:
        query = Zone(elements=[
            {"kind": "frame", "start": 34, "end": 34},
            {"kind": "stringer", "start": 24, "end": 28},
        ])
        record = Zone(elements=[
            {"kind": "frame", "start": 35, "end": 35},
            {"kind": "stringer", "start": 24, "end": 28},
        ])
        self.assertEqual(compare_zones(query, record).status, "CONFLICT")

    def test_missing_extraction_is_unknown(self) -> None:
        query = Zone(elements=[{"kind": "frame", "start": 34, "end": 34}])
        self.assertEqual(compare_zones(query, Zone()).status, "UNKNOWN")

    def test_rib_and_stringer_intersections_must_both_match(self) -> None:
        query = Zone(elements=[
            {"kind": "rib", "start": 4, "end": 8},
            {"kind": "stringer", "start": 4, "end": 8},
        ])
        matching = Zone(elements=[
            {"kind": "rib", "start": 6, "end": 10},
            {"kind": "stringer", "start": 6, "end": 10},
        ])
        conflicting = Zone(elements=[
            {"kind": "rib", "start": 9, "end": 10},
            {"kind": "stringer", "start": 6, "end": 10},
        ])
        result = compare_zones(query, matching)
        self.assertEqual(result.status, "MATCH")
        self.assertIn("rib: пересечение 6-8", result.details)
        self.assertEqual(compare_zones(query, conflicting).status, "CONFLICT")

    def test_numbered_wing_devices_are_compared_as_intervals(self) -> None:
        query = Zone(elements=[{"kind": "spoiler", "start": 3, "end": 3}])
        self.assertEqual(
            compare_zones(query, Zone(elements=[{"kind": "spoiler", "start": 2, "end": 4}])).status,
            "MATCH",
        )
        self.assertEqual(
            compare_zones(query, Zone(elements=[{"kind": "spoiler", "start": 4, "end": 4}])).status,
            "CONFLICT",
        )
        self.assertEqual(
            compare_zones(query, Zone(elements=[{"kind": "spoiler"}])).status,
            "UNKNOWN",
        )

    def test_target_component_is_distinct_from_location_reference(self) -> None:
        query = Zone(
            components=["rib"],
            elements=[{"kind": "rib", "start": 7, "end": 7, "role": "target"}],
        )
        same_target = Zone(
            components=["rib"],
            elements=[{"kind": "rib", "start": 7, "end": 7, "role": "target"}],
        )
        skin_at_rib = Zone(
            components=["skin"],
            elements=[{"kind": "rib", "start": 7, "end": 7, "role": "reference"}],
        )
        missing_role = Zone(
            components=["rib"],
            elements=[{"kind": "rib", "start": 7, "end": 7}],
        )
        self.assertEqual(compare_zones(query, same_target).status, "MATCH")
        self.assertEqual(compare_zones(query, skin_at_rib).status, "CONFLICT")
        self.assertEqual(compare_zones(query, missing_role).status, "UNKNOWN")

    def test_best_zone_match_does_not_merge_distinct_zones(self) -> None:
        query = Zone(elements=[
            {"kind": "rib", "start": 7, "end": 7},
            {"kind": "stringer", "start": 4, "end": 8},
        ])
        zones = [
            Zone(elements=[{"kind": "rib", "start": 7, "end": 7}]),
            Zone(elements=[{"kind": "stringer", "start": 4, "end": 8}]),
            Zone(elements=[
                {"kind": "rib", "start": 7, "end": 7},
                {"kind": "stringer", "start": 4, "end": 8},
            ]),
        ]
        result = best_zone_match(query, zones)
        self.assertEqual(result.status, "MATCH")
        self.assertEqual(result.zone_index, 2)

    def test_best_unknown_zone_prefers_most_complete_candidate(self) -> None:
        query = Zone(
            elements=[
                {"kind": "rib", "start": 1, "end": 15},
                {"kind": "stringer", "start": 4, "end": 8},
            ],
            components=["skin"],
        )
        zones = [
            Zone(elements=[{"kind": "stringer", "start": 2, "end": 10}]),
            Zone(elements=[
                {"kind": "rib", "start": 1, "end": 15},
                {"kind": "stringer", "start": 2, "end": 10},
            ]),
        ]

        result = best_zone_match(query, zones)

        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.zone_index, 1)

    def test_explicit_aircraft_area_conflicts_are_rejected(self) -> None:
        self.assertEqual(
            compare_zones(Zone(structure="крыло", system="NLG"), Zone(structure="крыло", system="MLG")).status,
            "CONFLICT",
        )

    def test_exact_evidence_deduplication_ignores_word_anchors(self) -> None:
        candidates = [
            {"source_path": "old.docx", "evidence_text": 'Text <a id="_Toc1"></a> end'},
            {"source_path": "new.docx", "evidence_text": 'Text <a id="_Toc2"></a> end'},
        ]
        self.assertEqual(deduplicate_evidence(candidates), [candidates[0]])

    def test_query_summary_exposes_extracted_zone(self) -> None:
        from esg.models import QueryExtraction

        query = QueryExtraction(
            defect_type="трещина",
            zone={
                "structure": "крыло",
                "side": "правая",
                "elements": [
                    {"kind": "stringer", "start": 4, "end": 8},
                    {"kind": "rib", "start": 1, "end": 15},
                ],
            },
        )
        summary = query_summary(query)
        self.assertIn("стрингеры=4-8", summary)
        self.assertIn("нервюры=1-15", summary)

    def test_chapter_five_is_selected_without_title_hardcode(self) -> None:
        sections = parse_markdown(
            "# 5 **Оценка накопленной повреждаемости**\nТекст.\n## **Описание ремонта**\nРемонт.",
            "doc",
        )
        self.assertTrue(all(is_chapter_five(section) for section in sections))


class StorageTests(unittest.TestCase):
    def test_replace_search_and_delete_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "test.sqlite3")
            document = {
                "document_id": "doc-1",
                "source_path": "sample.md",
                "content_hash": "abc",
                "size": 10,
                "mtime": 1.0,
                "status": "indexed",
                "error": "",
                "indexed_at": "2026-01-01T00:00:00Z",
            }
            record = {
                "record_id": "record-1",
                "document_id": "doc-1",
                "record_type": "repair_evidence",
                "section_id": "section-1",
                "section_heading": "Описание",
                "heading_path_json": '["Описание"]',
                "section_role": "repair_description",
                "defect_type": "трещина",
                "repair_description": "установлена накладка",
                "frame_start": 34,
                "frame_end": 34,
                "stringer_start": 24,
                "stringer_end": 28,
                "component": "обшивка",
                "side": "",
                "zone_text": "между стрингерами 24 и 28",
                "evidence_text": "Трещина отремонтирована накладкой.",
                "search_text": "трещина обшивка стрингерами 24 28",
                "extraction_status": "ok",
                "source_path": "sample.md",
                "filename_token_key": "|sample|tr|",
                "structure": "фюзеляж",
                "system": "",
                "region": "",
                "surface": "",
                "components": ["обшивка"],
                "elements": [
                    {"kind": "frame", "start": 34, "end": 34, "qualifier": ""},
                    {"kind": "stringer", "start": 24, "end": 28, "qualifier": ""},
                    {"kind": "rib", "start": 6, "end": 10, "qualifier": "RH"},
                ],
            }
            store.replace_document(document, [record])
            hits = store.lexical_search("трещина обшивка", 5)
            self.assertEqual(hits[0]["record_id"], "record-1")
            self.assertEqual(len(store.lexical_search("трещина", 5, ("tr",))), 1)
            self.assertEqual(len(store.lexical_search("трещина", 5, ("sr", "tr"))), 1)
            self.assertEqual(store.lexical_search("трещина", 5, ("tc",)), [])
            self.assertEqual(store.counts()["repair_records"], 1)
            structural = store.structural_search(
                [
                    {"kind": "stringer", "start": 26, "end": 30, "qualifier": ""},
                    {"kind": "rib", "start": 7, "end": 7, "qualifier": "RH"},
                ],
                ("tr",),
                ("repair_description",),
            )
            self.assertEqual([item["record_id"] for item in structural], ["record-1"])
            self.assertEqual(store.counts()["zone_elements"], 3)
            removed = store.migrate_raw_document("doc-1", 3)
            self.assertEqual(removed, ["record-1"])
            self.assertEqual(store.counts()["repair_records"], 0)
            self.assertEqual(store.documents()[0]["index_version"], 3)
            store.delete_document("doc-1")
            self.assertEqual(store.counts()["records"], 0)

    def test_existing_records_receive_filename_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.sqlite3"
            store = SQLiteStore(path)
            with store.connect() as connection:
                connection.execute(
                    """INSERT INTO documents
                    (document_id, source_path, content_hash, size, mtime, status, error, indexed_at)
                    VALUES ('doc', 'folder/report_TR_A.docx', 'abc', 1, 1, 'indexed', '', 'now')"""
                )
                connection.execute(
                    """INSERT INTO records (
                    record_id, document_id, record_type, section_id, section_heading,
                    heading_path_json, section_role, defect_type, repair_description,
                    component, side, zone_text, evidence_text, search_text,
                    extraction_status, source_path
                    ) VALUES (
                    'record', 'doc', 'raw_section', 'section', 'heading', '[]', 'other', '', '',
                    '', '', '', 'text', 'text', 'ok', 'folder/report_TR_A.docx'
                    )"""
                )
            migrated = SQLiteStore(path)
            self.assertEqual(migrated.record("record")["filename_tokens"], ["report", "tr", "a"])

    def test_document_aliases_are_replaced_without_duplicate_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "test.sqlite3")
            document = {
                "document_id": "doc", "source_path": "new/report.docx", "content_hash": "x",
                "size": 1, "mtime": 2.0, "status": "indexed", "error": "", "indexed_at": "now",
            }
            store.replace_document(document, [])
            store.replace_aliases("doc", [
                {"source_path": "old/report.docx", "source_name_key": "report.docx", "canonical_document_id": "doc", "size": 1, "mtime": 1.0},
                {"source_path": "new/report.docx", "source_name_key": "report.docx", "canonical_document_id": "doc", "size": 1, "mtime": 2.0},
            ])
            self.assertEqual(len(store.aliases("doc")), 2)
            self.assertEqual(store.counts()["source_aliases"], 2)


class FileHashTests(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
