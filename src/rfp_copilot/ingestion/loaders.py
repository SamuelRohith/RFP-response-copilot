"""
loaders.py — document loading layer for the RFP Response Co-pilot.

Design principle: every loader returns the SAME shape (LoadedDocument),
regardless of source format. Nothing downstream should ever need to know
whether the original file was a PDF or a DOCX.
"""

from dataclasses import dataclass, field
import pymupdf as fitz  # PyMuPDF
import docx  # python-docx


@dataclass
class LoadedDocument:
    text: str                      # full raw extracted text, pages joined
    pages: list[str]                # per-page (PDF) or per-section (DOCX) text
    metadata: dict = field(default_factory=dict)


def load_pdf(path: str) -> LoadedDocument:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        # "text" mode is a reasonable default; PyMuPDF also supports
        # "blocks" / "dict" modes that preserve layout/position info,
        # which you may need later for table-heavy RFPs.
        pages.append(page.get_text("text"))
    doc.close()

    return LoadedDocument(
        text="\n".join(pages),
        pages=pages,
        metadata={
            "source_type": "pdf",
            "filename": path.split("/")[-1],
            "page_count": len(pages),
        },
    )


def load_docx(path: str) -> LoadedDocument:
    """Uses python-docx's higher-level paragraph API (rather than raw XML
    iteration) so we can read each paragraph's style name. Any paragraph
    styled as a heading ("Heading 1", "Heading 2", ...) is prefixed with
    "## " in the output text — a lightweight, format-agnostic way to carry
    that structural signal through to the chunker, since plain extracted
    text otherwise throws heading information away entirely."""
    d = docx.Document(path)
    sections = []
    heading_count = 0

    for para in d.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "") if para.style else ""
        if style_name.startswith("Heading"):
            sections.append(f"## {text}")
            heading_count += 1
        else:
            sections.append(text)

    # Tables need separate handling — python-docx exposes them at doc level,
    # not interleaved with body paragraphs via the .paragraphs API
    table_texts = []
    for table in d.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells)
            table_texts.append(row_text)

    all_text = "\n".join(sections)
    if table_texts:
        all_text += "\n\n[TABLES]\n" + "\n".join(table_texts)

    return LoadedDocument(
        text=all_text,
        pages=sections,  # DOCX has no native "pages" — paragraphs stand in
        metadata={
            "source_type": "docx",
            "filename": path.split("/")[-1],
            "paragraph_count": len(sections),
            "table_count": len(d.tables),
            "heading_count": heading_count,
        },
    )


def load_document(path: str) -> LoadedDocument:
    """Single entry point — dispatches by file extension."""
    if path.lower().endswith(".pdf"):
        return load_pdf(path)
    elif path.lower().endswith(".docx"):
        return load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    import sys

    for test_path in sys.argv[1:]:
        result = load_document(test_path)
        print(f"\n=== {test_path} ===")
        print(f"metadata: {result.metadata}")
        print(f"total chars: {len(result.text)}")
        print(f"first 400 chars:\n{result.text[:400]}")
