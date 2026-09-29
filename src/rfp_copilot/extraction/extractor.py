"""Gemini-backed structured extraction for RFP document chunks."""

import json
import os
from pathlib import Path

from google import genai
from google.genai import types

from .schemas import ExtractionResult

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You extract structured information from an RFP (Request for
Proposal) document. Identify only information explicitly present in the supplied
text:

- requirements: functional, technical, or staffing obligations
- questions: things the vendor is explicitly asked to answer
- compliance_items: mandatory pass/fail eligibility conditions
- evaluation_criteria: how proposals will be scored, including weights when stated

Do not invent information or extract section headings without substantive content.
Return empty lists for categories with no matching items. Use the supplied section
heading as source_section for every extracted item."""


def _client() -> genai.Client:
    """Create a Gemini client from GEMINI_API_KEY."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or your environment.")
    return genai.Client(api_key=api_key)


def extract_from_chunk(
    chunk_text: str,
    source_section: str | None,
    client: genai.Client | None = None,
) -> ExtractionResult:
    """Extract and validate structured RFP information from one chunk."""
    try:
        client = client or _client()
        prompt = (
            f"Section heading: {source_section or '(none detected)'}\n\n"
            f"Chunk text:\n{chunk_text}"
        )
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ExtractionResult,
            ),
        )
        if not response.text:
            raise ValueError("Gemini returned no text")
        return ExtractionResult.model_validate_json(response.text)
    except Exception as error:
        print(f"WARNING: extraction failed for section {source_section!r}: {error}")
        return ExtractionResult()


def extract_from_chunks(chunks: list, client: genai.Client | None = None) -> ExtractionResult:
    """Extract every chunk and merge the results into a document-level result."""
    client = client or _client()
    result = ExtractionResult()
    for chunk in chunks:
        result = result.merge(extract_from_chunk(chunk.text, chunk.section_heading, client))
    return result


def save_structured_requirements(result: ExtractionResult, output_path: str | Path) -> Path:
    """Write the validated document-level extraction as readable JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    return output_path
