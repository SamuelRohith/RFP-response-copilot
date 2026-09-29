"""
chunkers.py — chunking layer for the RFP Response Co-pilot.

Strategy: structure-aware primary split (on detected numbered
sections/headings), with a recursive character-based fallback for any
single section that's still too large for the embedding model's limit.

This preserves the document's own organization (which downstream
extraction and retrieval both benefit from) instead of chunking blind.
"""

import re
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches heading patterns seen across both sample RFPs:
#   "1. Introduction and Background"
#   "SECTION 3 — TECHNICAL AND FUNCTIONAL REQUIREMENTS"
HEADING_PATTERN = re.compile(
    r"^(?:\d+\.\s+[A-Z][^\n]{2,80}|SECTION\s+\d+\s*[—\-–]\s*[^\n]{2,80}|##\s+[^\n]{2,80})$",
    re.MULTILINE,
)


@dataclass
class Chunk:
    text: str
    chunk_id: str
    section_heading: str | None
    metadata: dict = field(default_factory=dict)


def split_by_sections(text: str) -> list[tuple[str | None, str]]:
    """
    Splits text into (heading, section_text) pairs using detected headings
    as boundaries. Text before the first heading (if any) gets heading=None.
    """
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return [(None, text)]

    sections = []
    # content before the first heading
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

    for i, match in enumerate(matches):
        heading = match.group().strip()
        heading = re.sub(r"^##\s+", "", heading)  # strip our markdown-style marker for clean display
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        sections.append((heading, section_text))

    return sections


def chunk_document(
    text: str,
    doc_id: str,
    max_chunk_chars: int = 1500,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    """
    Primary split: by detected section headings.
    Fallback split: any section still longer than max_chunk_chars gets
    recursively split, with overlap, so no chunk blows the embedding limit.
    """
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_chars,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    sections = split_by_sections(text)
    chunks: list[Chunk] = []
    chunk_counter = 0

    for heading, section_text in sections:
        if not section_text:
            continue

        if len(section_text) <= max_chunk_chars:
            pieces = [section_text]
        else:
            pieces = fallback_splitter.split_text(section_text)

        for piece in pieces:
            chunk_counter += 1
            chunks.append(
                Chunk(
                    text=piece,
                    chunk_id=f"{doc_id}-chunk-{chunk_counter:03d}",
                    section_heading=heading,
                    metadata={
                        "doc_id": doc_id,
                        "char_count": len(piece),
                        "was_split_further": len(pieces) > 1,
                    },
                )
            )

    return chunks


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.loaders import load_document
    from ingestion.cleaners import clean_text

    for test_path in sys.argv[1:]:
        doc = load_document(test_path)
        cleaned = clean_text(doc.text)
        doc_id = os.path.basename(test_path)
        chunks = chunk_document(cleaned, doc_id)

        print(f"\n=== {test_path} ===")
        print(f"total chunks: {len(chunks)}")
        for c in chunks:
            print(f"  [{c.chunk_id}] heading={c.section_heading!r} chars={c.metadata['char_count']} split_further={c.metadata['was_split_further']}")
