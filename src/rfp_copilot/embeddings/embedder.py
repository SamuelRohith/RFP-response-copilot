"""
embedder.py — turns chunk text into vectors.

Pluggable by design: swap the embedding function without touching
vector_store.py or anything upstream. Two implementations are provided:

  - get_sentence_transformer_embedder()  <- USE THIS on your machine.
    Uses `all-MiniLM-L6-v2` via sentence-transformers: free, runs locally,
    no API key needed, good baseline quality. Downloads model weights from
    Hugging Face on first run (needs normal internet access).

  - get_dummy_embedder()  <- FOR OFFLINE PLUMBING TESTS ONLY.
    A deterministic hash-based "embedding" with no semantic meaning
    whatsoever. It exists purely so the rest of the pipeline (Chroma
    storage, retrieval wiring) can be exercised and verified in an
    environment with no model-download access. Never use this for
    anything you expect real retrieval quality from.
"""

import hashlib
import struct


def get_dummy_embedder(dim: int = 384):
    """Returns a callable: list[str] -> list[list[float]].
    Deterministic (same text always -> same vector) but semantically
    meaningless — for testing storage/retrieval plumbing only."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Repeat/truncate the hash bytes to fill `dim` floats in [0, 1)
            raw = (digest * ((dim // len(digest)) + 1))[: dim * 4]
            floats = [b / 255.0 for b in raw[:dim]]
            vectors.append(floats)
        return vectors

    return embed


def get_sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Real embedder for actual use. Requires: pip install sentence-transformers"""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    return embed
