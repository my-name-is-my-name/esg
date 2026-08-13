from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from esg.mineru import MinerUConverter


class MinerUOutputTests(unittest.TestCase):
    def test_converted_markdown_preserves_relative_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(markdown_dir=Path(directory))
            converter = MinerUConverter(settings)
            path = converter.markdown_path("aircraft/reports/report.pdf", ".pdf")
            self.assertEqual(path, Path(directory) / "aircraft/reports/report.pdf.md")

    def test_existing_markdown_keeps_md_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(markdown_dir=Path(directory))
            converter = MinerUConverter(settings)
            path = converter.markdown_path("prepared/report.md", ".md")
            self.assertEqual(path, Path(directory) / "prepared/report.md")

    def test_delete_removes_only_derived_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "zone/report.docx.md"
            target.parent.mkdir(parents=True)
            target.write_text("converted", encoding="utf-8")
            unrelated = root / "keep.md"
            unrelated.write_text("keep", encoding="utf-8")
            converter = MinerUConverter(SimpleNamespace(markdown_dir=root))
            converter.delete_markdown("zone/report.docx")
            self.assertFalse(target.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
