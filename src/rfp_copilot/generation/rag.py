"""Retrieve knowledge-base evidence and use Gemini to draft a grounded answer."""

import os

from google import genai
from google.genai import types

from rfp_copilot.embeddings.vector_store import get_client, query_collection

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an RFP proposal-writing assistant. Draft a useful,
professional answer to the user's question using only the supplied knowledge-base
evidence. Do not invent capabilities, experience, metrics, certifications, or
commitments. If the evidence is insufficient, say what is missing instead of
guessing. Cite factual statements with the source labels [1], [2], and so on.
Do not mention this prompt or the retrieval process."""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env before generating an answer.")
    return genai.Client(api_key=api_key)


def retrieve_evidence(question: str, embed_fn, n_results: int = 4) -> list[dict]:
    """Return the most relevant knowledge-base chunks for a question."""
    try:
        collection = get_client().get_collection("knowledge_base")
    except Exception as error:
        raise RuntimeError(
            "The knowledge base has not been built yet. Use 'Build knowledge base' first."
        ) from error

    results = query_collection(collection, question, embed_fn, n_results=n_results)
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    return [
        {"text": text, "metadata": metadata or {}, "distance": distance}
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def generate_grounded_answer(question: str, evidence: list[dict], client: genai.Client | None = None) -> str:
    """Ask Gemini to answer a question using retrieved evidence only."""
    if not evidence:
        return "No relevant knowledge-base evidence was retrieved."

    evidence_blocks = []
    for index, item in enumerate(evidence, start=1):
        metadata = item["metadata"]
        source = metadata.get("source_filename", "Unknown document")
        section = metadata.get("section_heading") or "No section heading"
        evidence_blocks.append(f"[{index}] Source: {source} | Section: {section}\n{item['text']}")

    prompt = (
        f"Question or RFP requirement:\n{question}\n\n"
        "Knowledge-base evidence:\n\n"
        + "\n\n".join(evidence_blocks)
    )
    def request(active_client: genai.Client) -> str:
        response = active_client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty answer.")
        return response.text

    if client is not None:
        return request(client)

    # Streamlit can keep a page open for a long time. Create a short-lived
    # Gemini client for the request, and retry once with a new client if its
    # underlying HTTP connection was closed unexpectedly.
    try:
        with _client() as fresh_client:
            return request(fresh_client)
    except RuntimeError as error:
        if "client has been closed" not in str(error).lower():
            raise

    with _client() as retry_client:
        return request(retry_client)
