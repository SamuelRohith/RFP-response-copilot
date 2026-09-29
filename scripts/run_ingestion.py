"""Run the RFP workflow: load -> clean -> chunk -> Gemini extraction -> JSON -> Chroma."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rfp_copilot.embeddings.embedder import get_sentence_transformer_embedder
from rfp_copilot.embeddings.vector_store import store_rfp_chunks
from rfp_copilot.extraction.extractor import extract_from_chunks, save_structured_requirements
from rfp_copilot.ingestion.chunkers import chunk_document
from rfp_copilot.ingestion.cleaners import clean_text
from rfp_copilot.ingestion.loaders import load_document

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample_rfps"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"


def run_rfp(rfp_path: Path, use_dummy_embeddings: bool = False) -> Path:
    """Run one RFP end-to-end and return its structured JSON output path."""
    document = load_document(str(rfp_path))
    cleaned = clean_text(document.text)
    chunks = chunk_document(cleaned, doc_id=rfp_path.name)
    print(f"Loaded {rfp_path.name}: {len(document.text)} raw chars -> {len(chunks)} chunks")

    extraction = extract_from_chunks(chunks)
    output_path = OUTPUT_DIR / f"{rfp_path.stem}_structured_requirements.json"
    save_structured_requirements(extraction, output_path)
    print(f"Structured requirements written to: {output_path}")

    if use_dummy_embeddings:
        from rfp_copilot.embeddings.embedder import get_dummy_embedder

        embed = get_dummy_embedder()
        print("Using dummy embeddings: Chroma retrieval is only a plumbing check.")
    else:
        embed = get_sentence_transformer_embedder()
    store_rfp_chunks(chunks, rfp_path.name, embed)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract RFP requirements with Gemini and store chunks in Chroma.")
    parser.add_argument("rfp", nargs="?", type=Path, help="Path to one PDF or DOCX RFP. Defaults to the first sample.")
    parser.add_argument("--dummy-embeddings", action="store_true", help="Avoid downloading the sentence-transformer model; Chroma results are not meaningful.")
    args = parser.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    rfp_path = args.rfp or next(iter(sorted(SAMPLE_DIR.glob("*.pdf"))), None)
    if rfp_path is None or not rfp_path.is_file():
        parser.error("Provide a valid PDF/DOCX RFP path, or add a sample to data/sample_rfps.")
    if rfp_path.suffix.lower() not in {".pdf", ".docx"}:
        parser.error("RFP must be a .pdf or .docx file.")
    run_rfp(rfp_path.resolve(), args.dummy_embeddings)


if __name__ == "__main__":
    main()
