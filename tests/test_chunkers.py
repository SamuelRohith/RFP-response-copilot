from rfp_copilot.ingestion.chunkers import chunk_document_blocks
from rfp_copilot.ingestion.loaders import DocumentBlock


def test_block_chunks_retain_source_metadata():
    blocks = [
        DocumentBlock(text="## Security", section="Security", block_index=0),
        DocumentBlock(text="Encrypt all data at rest and in transit.", page=2, section="Security", block_index=1),
    ]
    chunks = chunk_document_blocks(blocks, "rfp.pdf")
    assert len(chunks) == 1
    assert chunks[0].section_heading == "Security"
    assert chunks[0].metadata["page"] == 2
    assert chunks[0].metadata["block_index"] == 1
    assert chunks[0].metadata["end_char"] > chunks[0].metadata["start_char"]
