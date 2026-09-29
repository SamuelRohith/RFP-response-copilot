import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rfp_copilot.embeddings.embedder import get_sentence_transformer_embedder
from rfp_copilot.embeddings.vector_store import (
    get_client,
    query_collection,
)

embed = get_sentence_transformer_embedder()
client = get_client()
collection = client.get_collection("current_rfp")

question = "What are the HIPPAA and the security requirements?"

results = query_collection(
    collection,
    question,
    embed,
    n_results=3,
)

for text, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0],
):
    print("\n--- Match ---")
    print("Section:", metadata["section_heading"])
    print("Distance:", distance)
    print(text)