"""
vector_store.py — wraps ChromaDB for the RFP Response Co-pilot.

Two collections, deliberately kept separate:
  - "knowledge_base"  : past proposals, product docs, case studies
                        (built once, persists across many RFPs)
  - "current_rfp"     : the RFP currently being responded to
                        (rebuilt fresh per RFP — cleared before each new one)

Week 6's RAG pipeline will query "knowledge_base" using text derived from
"current_rfp"'s extracted requirements. Keeping them separate means that
query can never accidentally retrieve the RFP's own text as if it were
prior knowledge.
"""

import os
import chromadb

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "vector_store")


def get_client(persist_path: str = DB_PATH) -> chromadb.ClientAPI:
    os.makedirs(persist_path, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def store_kb_chunks(kb_chunks: list, embed_fn, client: chromadb.ClientAPI | None = None, collection_name: str = "knowledge_base"):
    """Embeds and stores KBChunk objects (from kb_builder.py) into Chroma,
    carrying doc_type and source_filename through as metadata."""
    client = client or get_client()
    collection = client.get_or_create_collection(name=collection_name)

    texts = [kc.chunk.text for kc in kb_chunks]
    if not texts:
        print("No KB chunks to store.")
        return collection

    embeddings = embed_fn(texts)
    collection.upsert(
        ids=[kc.chunk.chunk_id for kc in kb_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "doc_type": kc.doc_type,
                "source_filename": kc.source_filename,
                "section_heading": kc.chunk.section_heading or "",
            }
            for kc in kb_chunks
        ],
    )
    print(f"Stored {len(kb_chunks)} chunks in '{collection_name}' collection.")
    return collection


def store_rfp_chunks(chunks: list, doc_id: str, embed_fn, client: chromadb.ClientAPI | None = None, collection_name: str = "current_rfp"):
    """Embeds and stores Chunk objects (from chunkers.py) for the RFP
    currently being processed. Clears the collection first, since only
    one RFP should be 'current' at a time."""
    client = client or get_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # collection didn't exist yet — fine
    collection = client.get_or_create_collection(name=collection_name)

    texts = [c.text for c in chunks]
    if not texts:
        print("No RFP chunks to store.")
        return collection

    embeddings = embed_fn(texts)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "doc_id": doc_id,
                "section_heading": c.section_heading or "",
            }
            for c in chunks
        ],
    )
    print(f"Stored {len(chunks)} chunks in '{collection_name}' collection.")
    return collection


def query_collection(collection, query_text: str, embed_fn, n_results: int = 3, where: dict | None = None):
    """Basic similarity search — the building block Week 6's retrieval
    step will call, with the query being derived from an extracted
    requirement rather than typed by a user."""
    query_embedding = embed_fn([query_text])[0]
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.loaders import load_document
    from ingestion.cleaners import clean_text
    from ingestion.chunkers import chunk_document
    from knowledge_base.kb_builder import build_knowledge_base
    from embeddings.embedder import get_dummy_embedder

    print("NOTE: using the dummy embedder (no semantic meaning) — this test verifies")
    print("Chroma storage/retrieval plumbing only, not real retrieval quality.\n")

    embed_fn = get_dummy_embedder()
    client = get_client()

    # Store the knowledge base
    kb_chunks = build_knowledge_base()
    store_kb_chunks(kb_chunks, embed_fn, client=client)

    # Store one sample RFP as "current"
    rfp_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sample_rfps", "sample1_it_support_rfp.pdf")
    doc = load_document(rfp_path)
    cleaned = clean_text(doc.text)
    chunks = chunk_document(cleaned, doc_id="sample1_it_support_rfp.pdf")
    store_rfp_chunks(chunks, "sample1_it_support_rfp.pdf", embed_fn, client=client)

    # Query the KB collection
    kb_collection = client.get_collection("knowledge_base")
    results = query_collection(kb_collection, "HIPAA healthcare experience", embed_fn, n_results=2)
    print("\nQuery results (dummy embedder — distances are not semantically meaningful):")
    for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  doc_type={meta['doc_type']} source={meta['source_filename']}")
        print(f"  text preview: {doc_text[:100]}...")
