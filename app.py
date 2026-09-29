"""Simple browser interface for the RFP Response Co-pilot."""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from rfp_copilot.embeddings.embedder import get_sentence_transformer_embedder
from rfp_copilot.embeddings.vector_store import store_kb_chunks, store_rfp_chunks
from rfp_copilot.extraction.extractor import extract_from_chunks, save_structured_requirements
from rfp_copilot.generation.rag import generate_grounded_answer, retrieve_evidence
from rfp_copilot.ingestion.chunkers import chunk_document
from rfp_copilot.ingestion.cleaners import clean_text
from rfp_copilot.ingestion.loaders import load_document
from rfp_copilot.knowledge_base.kb_builder import build_knowledge_base

OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

st.set_page_config(page_title="RFP Response Co-pilot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def embedding_model():
    return get_sentence_transformer_embedder()


def ingest_rfp(uploaded_file) -> Path:
    """Run the existing RFP ingestion workflow for an uploaded document."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
        temporary_file.write(uploaded_file.getbuffer())
        temporary_path = Path(temporary_file.name)
    try:
        document = load_document(str(temporary_path))
        chunks = chunk_document(clean_text(document.text), doc_id=uploaded_file.name)
        extraction = extract_from_chunks(chunks)
        output_path = OUTPUT_DIR / f"{Path(uploaded_file.name).stem}_structured_requirements.json"
        save_structured_requirements(extraction, output_path)
        store_rfp_chunks(chunks, uploaded_file.name, embedding_model())
        return output_path
    finally:
        temporary_path.unlink(missing_ok=True)


st.title("RFP Response Co-pilot")
st.caption("Upload an RFP, retrieve relevant company knowledge, and generate a Gemini-grounded response.")

with st.sidebar:
    st.header("1. Prepare documents")
    uploaded_rfp = st.file_uploader("Upload an RFP", type=["pdf", "docx"])
    if st.button("Process RFP", use_container_width=True, disabled=uploaded_rfp is None):
        with st.spinner("Extracting requirements and storing RFP chunks..."):
            try:
                output = ingest_rfp(uploaded_rfp)
                st.success(f"RFP processed. Structured requirements saved to {output.relative_to(PROJECT_ROOT)}.")
            except Exception as error:
                st.error(f"Could not process the RFP: {error}")

    st.divider()
    st.write("Put past proposals, product documents, and case studies in `data/knowledge_base/`, then build the searchable knowledge base.")
    if st.button("Build knowledge base", use_container_width=True):
        with st.spinner("Loading and embedding knowledge-base documents..."):
            try:
                chunks = build_knowledge_base()
                if not chunks:
                    st.warning("No PDF or DOCX files were found in data/knowledge_base/.")
                else:
                    store_kb_chunks(chunks, embedding_model())
                    st.success(f"Knowledge base built with {len(chunks)} chunks.")
            except Exception as error:
                st.error(f"Could not build the knowledge base: {error}")

st.header("2. Draft a grounded response")
question = st.text_area(
    "Enter an RFP requirement or question",
    placeholder="Example: Describe our approach to protecting sensitive data and meeting healthcare compliance requirements.",
    height=110,
)
top_k = st.slider("Supporting chunks to retrieve", min_value=1, max_value=8, value=4)

if st.button("Generate response with Gemini", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence and drafting a grounded response..."):
        try:
            evidence = retrieve_evidence(question.strip(), embedding_model(), top_k)
            answer = generate_grounded_answer(question.strip(), evidence)
            st.subheader("Generated response")
            st.write(answer)
            with st.expander("View retrieved evidence"):
                for index, item in enumerate(evidence, start=1):
                    metadata = item["metadata"]
                    st.markdown(
                        f"**[{index}] {metadata.get('source_filename', 'Unknown document')}**  \n"
                        f"{metadata.get('section_heading') or 'No section heading'}"
                    )
                    st.write(item["text"])
        except Exception as error:
            st.error(f"Could not generate a response: {error}")
