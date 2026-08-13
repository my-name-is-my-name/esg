from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from esg.config import Settings


class InputSafetyTests(unittest.TestCase):
    def test_input_directory_is_never_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            settings = Settings(input_dir=missing, markdown_dir=Path(directory) / "md", runtime_dir=Path(directory) / "runtime")
            with self.assertRaises(RuntimeError):
                settings.ensure_dirs()
            self.assertFalse(missing.exists())

    def test_input_must_not_overlap_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            settings = Settings(input_dir=input_dir, markdown_dir=input_dir / "md", runtime_dir=root / "runtime")
            with self.assertRaises(RuntimeError):
                settings.ensure_dirs()

    def test_separate_read_and_write_directories_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            settings = Settings(input_dir=input_dir, markdown_dir=root / "md", runtime_dir=root / "runtime")
            settings.ensure_dirs()
            self.assertTrue(settings.markdown_dir.is_dir())
            self.assertTrue(settings.runtime_dir.is_dir())

    def test_archive_input_must_exist_and_remains_unmodified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            primary.mkdir()
            missing_archive = root / "archive"
            settings = Settings(
                input_dir=primary,
                archive_input_dir=missing_archive,
                markdown_dir=root / "md",
                runtime_dir=root / "runtime",
            )
            with self.assertRaises(RuntimeError):
                settings.ensure_dirs()
            self.assertFalse(missing_archive.exists())


if __name__ == "__main__":
    unittest.main()
