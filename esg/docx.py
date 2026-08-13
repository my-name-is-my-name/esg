from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_markdown(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        styles = _read_styles(archive)
    body = document.find(f"{W}body")
    if body is None:
        return ""
    blocks: list[str] = []
    for child in body:
        if child.tag == f"{W}p":
            block = _paragraph_markdown(child, styles)
        elif child.tag == f"{W}tbl":
            block = _table_markdown(child, styles)
        else:
            block = ""
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def _read_styles(archive: zipfile.ZipFile) -> dict[str, int]:
    try:
        root = ET.fromstring(archive.read("word/styles.xml"))
    except KeyError:
        return {}
    raw: dict[str, tuple[str, int | None, str]] = {}
    for style in root.findall(f"{W}style"):
        if style.get(f"{W}type") != "paragraph":
            continue
        style_id = style.get(f"{W}styleId") or ""
        name_node = style.find(f"{W}name")
        outline_node = style.find(f"{W}pPr/{W}outlineLvl")
        based_on = style.find(f"{W}basedOn")
        raw[style_id] = (
            (name_node.get(f"{W}val") if name_node is not None else "") or "",
            int(outline_node.get(f"{W}val")) if outline_node is not None else None,
            (based_on.get(f"{W}val") if based_on is not None else "") or "",
        )

    resolved: dict[str, int] = {}
    for style_id in raw:
        level = _heading_level(style_id, raw, set())
        if level:
            resolved[style_id] = level
    return resolved


def _heading_level(
    style_id: str,
    styles: dict[str, tuple[str, int | None, str]],
    visited: set[str],
) -> int:
    if not style_id or style_id in visited or style_id not in styles:
        return 0
    visited.add(style_id)
    name, outline, based_on = styles[style_id]
    if outline is not None and 0 <= outline <= 5:
        return outline + 1
    match = re.search(r"(?:heading|заголовок)\s*([1-6])", f"{style_id} {name}", re.IGNORECASE)
    if match:
        return int(match.group(1))
    return _heading_level(based_on, styles, visited)


def _paragraph_markdown(paragraph: ET.Element, styles: dict[str, int]) -> str:
    text = _element_text(paragraph).strip()
    if not text:
        return ""
    properties = paragraph.find(f"{W}pPr")
    style_id = ""
    outline: int | None = None
    is_list = False
    if properties is not None:
        style = properties.find(f"{W}pStyle")
        style_id = (style.get(f"{W}val") if style is not None else "") or ""
        outline_node = properties.find(f"{W}outlineLvl")
        if outline_node is not None:
            outline = int(outline_node.get(f"{W}val"))
        is_list = properties.find(f"{W}numPr") is not None
    level = outline + 1 if outline is not None and 0 <= outline <= 5 else styles.get(style_id, 0)
    if level:
        return f"{'#' * level} {text}"
    return f"- {text}" if is_list else text


def _table_markdown(table: ET.Element, styles: dict[str, int]) -> str:
    rows: list[list[str]] = []
    for row in table.findall(f"{W}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{W}tc"):
            parts = [
                _paragraph_markdown(paragraph, styles).lstrip("#- ")
                for paragraph in cell.findall(f"{W}p")
            ]
            text = "<br>".join(part for part in parts if part)
            cells.append(text.replace("|", "\\|"))
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in normalized[1:])
    return "\n".join(lines)


def _element_text(element: ET.Element) -> str:
    output: list[str] = []
    for node in element.iter():
        if node.tag == f"{W}t" and node.text:
            output.append(node.text)
        elif node.tag == f"{W}tab":
            output.append("\t")
        elif node.tag in {f"{W}br", f"{W}cr"}:
            output.append("\n")
    return "".join(output)
