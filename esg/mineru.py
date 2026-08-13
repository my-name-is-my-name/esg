from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from esg.config import Settings
from esg.docx import docx_to_markdown


MARKDOWN_EXTENSIONS = {".md", ".markdown"}
MINERU_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg"}


class MinerUConverter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def convert(
        self,
        source: Path,
        document_id: str,
        relative_path: str,
        reuse_existing: bool = False,
    ) -> Path:
        target = self.markdown_path(relative_path, source.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        if reuse_existing and target.is_file():
            return target
        if source.suffix.lower() == ".docx" and getattr(self.settings, "direct_docx_enabled", True):
            markdown = docx_to_markdown(source)
            if not markdown.strip():
                raise ValueError("DOCX contains no extractable text")
            target.write_text(markdown, encoding="utf-8")
            return target
        if source.suffix.lower() in MARKDOWN_EXTENSIONS:
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            return target
        if source.suffix.lower() not in MINERU_EXTENSIONS:
            raise ValueError(f"Unsupported source extension: {source.suffix}")
        executable = shutil.which("mineru")
        if not executable:
            raise RuntimeError("mineru executable not found")
        output_root = self.settings.converted_dir / document_id
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True)
        command = [executable, "-p", str(source), "-o", str(output_root)]
        if self.settings.mineru_backend:
            command.extend(["--backend", self.settings.mineru_backend])
        if self.settings.mineru_method:
            command.extend(["--method", self.settings.mineru_method])
        if self.settings.mineru_lang:
            command.extend(["--lang", self.settings.mineru_lang])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.settings.mineru_timeout_seconds,
            check=False,
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout or "MinerU failed").strip())
        candidates = sorted(output_root.rglob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("MinerU produced no Markdown")
        shutil.copy2(candidates[0], target)
        return target

    def markdown_path(self, relative_path: str, source_suffix: str) -> Path:
        relative = Path(relative_path)
        if source_suffix in MARKDOWN_EXTENSIONS:
            return self.settings.markdown_dir / relative
        return self.settings.markdown_dir / Path(f"{relative.as_posix()}.md")

    def delete_markdown(self, relative_path: str) -> None:
        suffix = Path(relative_path).suffix.lower()
        target = self.markdown_path(relative_path, suffix)
        if target.exists():
            target.unlink()
        parent = target.parent
        while parent != self.settings.markdown_dir and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
