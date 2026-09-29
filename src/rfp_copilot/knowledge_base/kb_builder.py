"""
kb_builder.py — builds the knowledge base from past proposals, product
documentation, and case studies.

Reuses the exact same loading/cleaning/chunking pipeline as the RFP
ingestion path (Topics 2-4) — the mechanics are identical. What's
different is the metadata attached to each chunk: doc_type, and
(optionally) outcome/date, which is what lets Week 6's retrieval step
be selective about what it pulls in.

Expected folder layout under data/knowledge_base/:
    data/knowledge_base/
        past_proposals/
        product_docs/
        case_studies/

Each subfolder name becomes the doc_type tag automatically. If your
files aren't organized this way yet, doc_type falls back to "unspecified"
and you can fix it later without re-running ingestion logic.
"""

import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ingestion.loaders import load_document
from ingestion.cleaners import clean_text
from ingestion.chunkers import chunk_document, Chunk

KB_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "knowledge_base")
DOC_TYPE_FOLDERS = {"past_proposals", "product_docs", "case_studies"}


@dataclass
class KBChunk:
    chunk: Chunk
    doc_type: str
    source_filename: str
    extra_metadata: dict = field(default_factory=dict)


def infer_doc_type(file_path: str, kb_root: str) -> str:
    """doc_type = the immediate subfolder name under the KB root, if it
    matches a known category; otherwise 'unspecified' rather than guessing."""
    rel = os.path.relpath(file_path, kb_root)
    parts = rel.split(os.sep)
    if len(parts) > 1 and parts[0] in DOC_TYPE_FOLDERS:
        return parts[0]
    return "unspecified"


def build_knowledge_base(kb_root: str = KB_ROOT, max_chunk_chars: int = 1500) -> list[KBChunk]:
    """Walks the knowledge base folder, runs every PDF/DOCX through the
    same load -> clean -> chunk pipeline as RFP ingestion, and tags each
    resulting chunk with its doc_type and source filename."""
    kb_chunks: list[KBChunk] = []

    if not os.path.isdir(kb_root):
        print(f"WARNING: knowledge base root not found: {kb_root}")
        return kb_chunks

    for dirpath, _, filenames in os.walk(kb_root):
        for filename in filenames:
            if not filename.endswith((".pdf", ".docx")):
                continue
            file_path = os.path.join(dirpath, filename)
            doc_type = infer_doc_type(file_path, kb_root)

            try:
                doc = load_document(file_path)
            except Exception as e:
                print(f"WARNING: failed to load {file_path}: {e}")
                continue

            cleaned = clean_text(doc.text)
            chunks = chunk_document(cleaned, doc_id=filename, max_chunk_chars=max_chunk_chars)

            for c in chunks:
                kb_chunks.append(
                    KBChunk(
                        chunk=c,
                        doc_type=doc_type,
                        source_filename=filename,
                    )
                )

    return kb_chunks


if __name__ == "__main__":
    kb_chunks = build_knowledge_base()
    print(f"Total knowledge base chunks: {len(kb_chunks)}")

    by_type: dict[str, int] = {}
    for kc in kb_chunks:
        by_type[kc.doc_type] = by_type.get(kc.doc_type, 0) + 1
    print("By doc_type:", by_type)

    for kc in kb_chunks[:5]:
        print(f"  [{kc.chunk.chunk_id}] doc_type={kc.doc_type} source={kc.source_filename} heading={kc.chunk.section_heading!r}")
