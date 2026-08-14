from __future__ import annotations

import re

from esg.chunking import split_text


_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.IGNORECASE)
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^]]*]\([^)]*\)")
_FIGURE_REFERENCE_RE = re.compile(
    r"\(\s*(?:см\.?\s*)?(?:на\s+)?рисун(?:ок|ке|ки|ках)\b[^)]*\)",
    re.IGNORECASE,
)
_FIGURE_REFERENCE_SENTENCE_RE = re.compile(
    r"[^.!?\n]*(?:представлен\w*|показан\w*|приведен\w*)\s+(?:на|в)\s+рисунк\w*[^.!?\n]*[.!?]?",
    re.IGNORECASE,
)
_FIGURE_NUMBER_RE = re.compile(r"\bРисунок\s+\d+(?:\.\d+)+(?:\s*[–—-]\s*Рисунок\s+\d+(?:\.\d+)*)?", re.IGNORECASE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(?:рисунок|рис\.)\s+\d", re.IGNORECASE)
_TABLE_CAPTION_RE = re.compile(r"^\s*таблица\s+\d", re.IGNORECASE)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?-{3,}:?$")
_RELEVANT_TABLE_HEADER_RE = re.compile(
    r"\b(?:область\s+ремонта|зона\s+ремонта|расположение)\b", re.IGNORECASE
)
_ELEMENT_PATTERNS = {
    "frame": r"(?:шпангоут\w*|frames?)",
    "stringer": r"(?:стрингер\w*|stringers?)",
    "rib": r"(?:нервюр\w*|ribs?)",
    "flap": r"(?:закрыл\w*|flaps?)",
    "slat": r"(?:предкрыл\w*|slats?)",
    "spoiler": r"(?:спойлер\w*|spoilers?)",
}
_NUMBER = r"(?P<start>\d{1,4})(?:\s*(?:[-–—]|по|до)\s*(?P<end>\d{1,4}))?"
_MAX_INTERVAL_WIDTH = 200


def repair_section_chunks(text: str, max_chars: int) -> list[str]:
    cleaned = clean_repair_section(text)
    if not cleaned:
        return []
    return split_text(cleaned, max(1, max_chars))


def interval_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for kind, pattern in _ELEMENT_PATTERNS.items():
        expressions = [
            re.compile(rf"\b{pattern}\b[^\d\n]{{0,40}}{_NUMBER}", re.IGNORECASE),
            re.compile(rf"{_NUMBER}[^\w\n]{{0,20}}\b{pattern}\b", re.IGNORECASE),
        ]
        for expression in expressions:
            for match in expression.finditer(text):
                start = int(match.group("start"))
                end = int(match.group("end") or start)
                if end < start:
                    start, end = end, start
                if end - start > _MAX_INTERVAL_WIDTH:
                    continue
                tokens.extend(f"inv_{kind}_{value:04d}" for value in range(start, end + 1))
    return list(dict.fromkeys(tokens))


def append_interval_tokens(text: str) -> str:
    tokens = interval_tokens(text)
    if not tokens:
        return text
    return f"{' '.join(tokens)}\n{text}"


def clean_repair_section(text: str) -> str:
    """Remove non-searchable illustration data while preserving repair-location tables."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _ANCHOR_RE.sub("", normalized)
    normalized = _HTML_IMAGE_RE.sub("", normalized)
    normalized = _MARKDOWN_IMAGE_RE.sub("", normalized)
    blocks = re.split(r"\n\s*\n", normalized)
    output: list[str] = []
    drop_figure_payload = False

    for raw_block in blocks:
        block = raw_block.strip()
        if not block:
            continue
        if _FIGURE_CAPTION_RE.match(block):
            drop_figure_payload = True
            continue
        if _is_markdown_table(block):
            drop_figure_payload = False
            table_text = _relevant_table_text(block)
            if table_text:
                output.append(table_text)
            continue
        if _TABLE_CAPTION_RE.match(block):
            drop_figure_payload = False
            continue
        if drop_figure_payload:
            # Direct OOXML conversion emits drawing labels as the paragraph
            # immediately following its caption.
            drop_figure_payload = False
            continue

        cleaned = _FIGURE_REFERENCE_SENTENCE_RE.sub("", block)
        cleaned = _FIGURE_REFERENCE_RE.sub("", cleaned)
        cleaned = _FIGURE_NUMBER_RE.sub("", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = cleaned.strip(" \t,;")
        if cleaned:
            output.append(cleaned)

    return "\n\n".join(_deduplicate_adjacent(output)).strip()


def _is_markdown_table(block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) >= 2 and all(_TABLE_LINE_RE.match(line) for line in lines)


def _relevant_table_text(block: str) -> str:
    rows = [_table_cells(line) for line in block.splitlines() if _TABLE_LINE_RE.match(line)]
    rows = [row for row in rows if row and not all(_TABLE_SEPARATOR_CELL_RE.match(cell) for cell in row)]
    header_index = next(
        (index for index, row in enumerate(rows) if _RELEVANT_TABLE_HEADER_RE.search(" ".join(row))),
        -1,
    )
    if header_index < 0:
        return ""
    headers = rows[header_index]
    rendered: list[str] = []
    for row in rows[header_index + 1 :]:
        fields = []
        for index, value in enumerate(row):
            value = value.strip()
            if not value:
                continue
            header = headers[index].strip() if index < len(headers) else ""
            fields.append(f"{header}: {value}" if header else value)
        if fields:
            rendered.append("; ".join(fields) + ".")
    return "\n".join(_deduplicate_adjacent(rendered))


def _table_cells(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [
        re.sub(r"\s+", " ", cell.replace("\\|", "|").replace("<br>", " ")).strip()
        for cell in cells
    ]


def _deduplicate_adjacent(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not output or value != output[-1]:
            output.append(value)
    return output
