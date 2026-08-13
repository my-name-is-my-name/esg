from __future__ import annotations

import tempfile
import unittest
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from esg.config import Settings, filename_matches, filename_tokens
from esg.ingest import IngestionService, SourceFile, group_source_files, group_sources, select_canonical
from esg.models import DocumentRepairExtraction, Repair, Zone
from esg.storage import SQLiteStore


class SourceSelectionTests(unittest.TestCase):
    @staticmethod
    def _service(root: Path, tokens: tuple[str, ...] = ("tr", "sr")) -> IngestionService:
        primary = root / "primary"
        archive = root / "archive"
        primary.mkdir(exist_ok=True)
        archive.mkdir(exist_ok=True)
        settings = Settings(
            input_dir=primary,
            archive_input_dir=archive,
            markdown_dir=root / "markdown",
            runtime_dir=root / "runtime",
            input_extensions=(".docx",),
            input_filename_tokens=tokens,
        )
        return IngestionService(
            settings,
            store=SQLiteStore(settings.db_path),
            embeddings=object(),
            vector_index=object(),
        )

    def test_only_configured_docx_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docx = root / "M-120.A320.01.TR_A.docx"
            pdf = root / "report.pdf"
            lock = root / "~$report.docx"
            other = root / "M-120.A320.01.TRC_A.docx"
            for path in (docx, pdf, lock, other):
                path.write_bytes(b"test")
            service = IngestionService(
                SimpleNamespace(input_extensions=(".docx",), input_filename_tokens=("tr",)),
                store=object(),
                embeddings=object(),
                vector_index=object(),
            )
            self.assertTrue(service._is_source(docx))
            self.assertFalse(service._is_source(pdf))
            self.assertFalse(service._is_source(lock))
            self.assertFalse(service._is_source(other))

    def test_filename_tokens_use_separators_without_domain_rules(self) -> None:
        self.assertEqual(filename_tokens("MP-120.02-TR_B_II.docx"), ("mp", "120", "02", "tr", "b", "ii"))

    def test_filename_profiles_are_alternatives(self) -> None:
        profiles = ("tr", "sr")
        self.assertTrue(filename_matches("MP-120.02-TR_B.docx", profiles))
        self.assertTrue(filename_matches("MP-120.02-SR_B.docx", profiles))
        self.assertFalse(filename_matches("MP-120.02-AN_B.docx", profiles))

    def test_same_filename_in_different_folders_is_one_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old" / "Report_TR.docx"
            new = root / "new" / "report_tr.DOCX"
            old.parent.mkdir()
            new.parent.mkdir()
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            groups = group_sources([old, new])
            self.assertEqual(list(groups), ["report_tr.docx"])
            self.assertEqual(select_canonical(groups["report_tr.docx"], root), new)

    def test_same_filename_is_deduplicated_across_source_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary" / "Report_TR.docx"
            archive = root / "archive" / "report_tr.docx"
            primary.parent.mkdir()
            archive.parent.mkdir()
            primary.write_bytes(b"primary")
            archive.write_bytes(b"archive")
            groups = group_source_files([
                SourceFile(primary, "Report_TR.docx"),
                SourceFile(archive, "__s7a/report_tr.docx"),
            ])
            self.assertEqual(len(groups), 1)

    def test_manual_refresh_persists_filtered_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self._service(root)
            (root / "primary" / "report_TR.docx").write_bytes(b"tr")
            (root / "primary" / "report_AN.docx").write_bytes(b"an")
            (root / "archive" / "report_SR.docx").write_bytes(b"sr")

            refreshed = service._sources_for_job(refresh_sources=True)

            self.assertEqual(
                [item.relative for item in refreshed],
                ["__s7a/report_SR.docx", "report_TR.docx"],
            )
            status = service.source_status()
            self.assertTrue(status["catalog_available"])
            self.assertEqual(status["scanned"], 2)
            self.assertFalse(status["scheduled"])

    def test_cached_run_does_not_walk_source_trees(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self._service(root)
            source = root / "primary" / "report_TR.docx"
            source.write_bytes(b"tr")
            service._sources_for_job(refresh_sources=True)

            with patch("esg.ingest.os.walk", side_effect=AssertionError("unexpected walk")):
                cached = service._sources_for_job(refresh_sources=False)

            self.assertEqual(len(cached), 1)
            self.assertEqual(cached[0].path, source)

    def test_changed_source_configuration_requires_manual_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self._service(root)
            (root / "primary" / "report_TR.docx").write_bytes(b"tr")
            service._sources_for_job(refresh_sources=True)
            changed = IngestionService(
                Settings(
                    input_dir=root / "primary",
                    archive_input_dir=root / "archive",
                    markdown_dir=root / "markdown",
                    runtime_dir=root / "runtime",
                    input_extensions=(".docx",),
                    input_filename_tokens=("tr",),
                ),
                service.store,
                object(),
                object(),
            )

            with self.assertRaisesRegex(RuntimeError, "refresh_sources=true"):
                changed._sources_for_job(refresh_sources=False)

    def test_failed_refresh_preserves_last_good_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = self._service(root)
            (root / "primary" / "report_TR.docx").write_bytes(b"tr")
            service._sources_for_job(refresh_sources=True)

            with patch.object(service, "_discover_sources", side_effect=OSError("archive unavailable")):
                with self.assertRaises(OSError):
                    service._sources_for_job(refresh_sources=True)

            self.assertEqual(len(service._sources_for_job(refresh_sources=False)), 1)

    def test_full_pipeline_indexes_one_document_record_without_writing_source(self) -> None:
        class Embeddings:
            @staticmethod
            def embed(texts):
                return [[0.1, 0.2] for _ in texts]

        class Vectors:
            def __init__(self):
                self.rows = []

            def replace_document(self, document_id, rows):
                self.rows = rows

            @staticmethod
            def delete_document(document_id):
                del document_id

        class Extractor:
            @staticmethod
            def extract_document_repairs(sections):
                del sections
                return DocumentRepairExtraction(repairs=[Repair(
                    repair_id="AC-1",
                    evidence_text="Обшивка между нервюрами 1-15 и стрингерами 2-10.",
                    zones=[Zone(
                        components=["skin"], structure="wing",
                        elements=[
                            {"kind": "rib", "start": 1, "end": 15, "role": "boundary"},
                            {"kind": "stringer", "start": 2, "end": 10, "role": "boundary"},
                        ],
                    )],
                )])

        document = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Описание ремонта</w:t></w:r></w:p>
<w:p><w:r><w:t>Обшивка между нервюрами 1-15 и стрингерами 2-10.</w:t></w:r></w:p>
</w:body></w:document>"""
        styles = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
</w:styles>"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "input"
            source_root.mkdir()
            source = source_root / "report_TR.docx"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("word/document.xml", document)
                archive.writestr("word/styles.xml", styles)
            before = source.read_bytes()
            settings = Settings(
                input_dir=source_root,
                archive_input_dir=None,
                markdown_dir=root / "markdown",
                runtime_dir=root / "runtime",
                input_extensions=(".docx",),
                input_filename_tokens=("tr",),
            )
            store = SQLiteStore(settings.db_path)
            vectors = Vectors()
            service = IngestionService(settings, store, Embeddings(), vectors, Extractor())

            result = service.run(refresh_sources=True)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(store.document_registry_summary()["statuses"]["indexed"], 1)
            records = store.repair_document_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["repairs"][0]["repair_id"], "AC-1")
            self.assertEqual(len(vectors.rows), 1)


if __name__ == "__main__":
    unittest.main()
