from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from esg.docx import docx_to_markdown


DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>5 Оценка ремонта</w:t></w:r></w:p>
<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Описание ремонта</w:t></w:r></w:p>
<w:p><w:r><w:t>Обшивка между нервюрами 1-15</w:t></w:r><w:r><w:tab/><w:t>и стрингерами 2-10.</w:t></w:r></w:p>
<w:p><w:pPr><w:numPr><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>Первый пункт</w:t></w:r></w:p>
<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Зона</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Ремонт</w:t></w:r></w:p></w:tc></w:tr>
<w:tr><w:tc><w:p><w:r><w:t>RH</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Накладка</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
<w:sectPr/></w:body></w:document>"""

STYLES = """<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Заголовок 2"/></w:style>
</w:styles>"""


class DirectDocxTests(unittest.TestCase):
    def test_preserves_headings_paragraphs_lists_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("word/document.xml", DOCUMENT)
                archive.writestr("word/styles.xml", STYLES)

            markdown = docx_to_markdown(path)

        self.assertIn("# 5 Оценка ремонта", markdown)
        self.assertIn("## Описание ремонта", markdown)
        self.assertIn("Обшивка между нервюрами 1-15\tи стрингерами 2-10.", markdown)
        self.assertIn("- Первый пункт", markdown)
        self.assertIn("| Зона | Ремонт |", markdown)
        self.assertIn("| RH | Накладка |", markdown)


if __name__ == "__main__":
    unittest.main()
