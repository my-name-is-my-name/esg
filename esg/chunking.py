from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class Section:
    section_id: str
    heading: str
    heading_path: list[str]
    text: str
    ordinal: int

    @property
    def search_text(self) -> str:
        context = " > ".join(self.heading_path)
        return f"{context}\n{self.text}".strip()


def parse_markdown(markdown: str, document_id: str, max_chars: int = 12000) -> list[Section]:
    path: list[tuple[int, str]] = []
    current_heading = "Document"
    current_path = [current_heading]
    current_lines: list[str] = []
    sections: list[Section] = []

    def flush() -> None:
        nonlocal current_lines
        text = "\n".join(current_lines).strip()
        current_lines = []
        if not text:
            return
        parts = split_text(text, max_chars=max_chars)
        for part_index, part in enumerate(parts, start=1):
            ordinal = len(sections) + 1
            heading = current_heading if part_index == 1 else f"{current_heading} [{part_index}]"
            raw_id = f"{document_id}|{ordinal}|{heading}|{hashlib.sha256(part.encode()).hexdigest()}"
            sections.append(
                Section(
                    section_id=hashlib.sha256(raw_id.encode()).hexdigest()[:32],
                    heading=heading,
                    heading_path=list(current_path),
                    text=part,
                    ordinal=ordinal,
                )
            )

    for raw_line in markdown.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        match = HEADING_RE.match(raw_line.strip())
        if not match:
            current_lines.append(raw_line)
            continue
        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        while path and path[-1][0] >= level:
            path.pop()
        path.append((level, title))
        current_heading = title
        current_path = [item[1] for item in path]
    flush()
    return sections


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    output: list[str] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and length + len(paragraph) + 2 > max_chars:
            output.append("\n\n".join(current))
            current = []
            length = 0
        if len(paragraph) > max_chars:
            if current:
                output.append("\n\n".join(current))
                current = []
                length = 0
            output.extend(paragraph[pos : pos + max_chars] for pos in range(0, len(paragraph), max_chars))
            continue
        current.append(paragraph)
        length += len(paragraph) + 2
    if current:
        output.append("\n\n".join(current))
    return output
