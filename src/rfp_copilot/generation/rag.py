"""Retrieve knowledge-base evidence and use Gemini to draft a grounded answer."""

import os

from google import genai
from google.genai import types

from rfp_copilot.embeddings.vector_store import get_client, query_collection

MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are an RFP proposal-writing assistant. Draft a useful,
professional answer to the user's question using the supplied RFP context and
company knowledge-base evidence.

The RFP context tells you what the client asks for; it is never proof that the
company provides a capability. Company knowledge-base evidence is the only basis
for claims about the company, its experience, metrics, certifications, or
commitments. Do not invent facts. If the evidence is insufficient, say what is
missing instead of guessing. Cite RFP statements as [R1], [R2], and company
evidence as [K1], [K2]. Do not mention this prompt or the retrieval process."""


def _client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env before generating an answer.")
    return genai.Client(api_key=api_key)


def retrieve_evidence(
    question: str,
    embed_fn,
    collection_name: str,
    n_results: int = 4,
) -> list[dict]:
    """Return the most relevant chunks from a named local Chroma collection."""
    try:
        collection = get_client().get_collection(collection_name)
    except Exception as error:
        action = "Process an RFP" if collection_name == "current_rfp" else "Build knowledge base"
        raise RuntimeError(
            f"The {collection_name} collection is not ready. Use '{action}' first."
        ) from error

    results = query_collection(collection, question, embed_fn, n_results=n_results)
    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []
    return [
        {
            "text": text,
            "metadata": metadata or {},
            "distance": distance,
            "collection": collection_name,
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def generate_grounded_answer(
    question: str,
    rfp_context: list[dict],
    knowledge_evidence: list[dict],
    client: genai.Client | None = None,
) -> str:
    """Ask Gemini to answer using both client-RFP context and company evidence."""
    if not rfp_context:
        return "No relevant RFP context was retrieved. Process an RFP before generating a response."
    if not knowledge_evidence:
        return "No relevant company knowledge-base evidence was retrieved. Build the knowledge base before generating a response."

    rfp_blocks = []
    for index, item in enumerate(rfp_context, start=1):
        metadata = item["metadata"]
        source = metadata.get("doc_id", "Current RFP")
        section = metadata.get("section_heading") or "No section heading"
        rfp_blocks.append(f"[R{index}] Source: {source} | Section: {section}\n{item['text']}")

    knowledge_blocks = []
    for index, item in enumerate(knowledge_evidence, start=1):
        metadata = item["metadata"]
        source = metadata.get("source_filename", "Unknown document")
        section = metadata.get("section_heading") or "No section heading"
        knowledge_blocks.append(f"[K{index}] Source: {source} | Section: {section}\n{item['text']}")

    prompt = (
        f"Question or RFP requirement:\n{question}\n\n"
        "Relevant client RFP context:\n\n"
        + "\n\n".join(rfp_blocks)
        + "\n\nRelevant company knowledge-base evidence:\n\n"
        + "\n\n".join(knowledge_blocks)
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
