# RFP Response Co-pilot

A small local RAG application for drafting grounded RFP responses. It can process an RFP, search your company knowledge base, and use Gemini to generate a response backed only by retrieved evidence.

## What it does

```text
RFP PDF/DOCX
  -> text extraction and cleaning
  -> section-aware chunks
  -> Gemini structured extraction
  -> JSON requirements output + ChromaDB RFP storage

Your question or RFP requirement
  -> local semantic search of the current RFP and company knowledge
  -> relevant RFP context + company-evidence chunks
  -> Gemini grounded response with source labels
```

The structured RFP extraction identifies requirements, questions, compliance items, and evaluation criteria. The RAG response stage searches both the `current_rfp` collection and the `knowledge_base` collection. This gives Gemini the client's relevant requirement/context and the company's supporting evidence. RFP text is never treated as proof of a company capability.

## Setup

Use Python 3.11 or newer. From the project folder, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create your local `.env` file from the safe template:

```powershell
Copy-Item .env.example .env
```

Then add your Gemini API key to `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

`.env` is ignored by Git. Never put a real key in `.env.example`.

## Use the interface

After completing setup, double-click `Start RFP Co-pilot.bat` to open the interface. If you prefer, start the app from the project folder:

```powershell
streamlit run app.py
```

Then use the browser page:

1. Add internal source material to one or more of these folders:
   - `data/knowledge_base/past_proposals/`
   - `data/knowledge_base/product_docs/`
   - `data/knowledge_base/case_studies/`
2. Choose **Build knowledge base**.
3. Upload an RFP (`.pdf` or `.docx`) and choose **Process RFP**. This writes structured requirements to `data/outputs/` and stores the RFP chunks in ChromaDB.
4. Enter an RFP requirement or proposal question and select **Generate response with Gemini**.

Your RFPs and company knowledge-base documents are intentionally not included in GitHub. Each person using the project must add their own local documents before building the knowledge base or processing an RFP.

The app retrieves the most relevant RFP and knowledge-base chunks with Sentence Transformers and ChromaDB, passes both sets to Gemini, and displays a cited draft plus the supporting evidence. RFP citations use `[R1]`, `[R2]`; company-evidence citations use `[K1]`, `[K2]`. Gemini is instructed not to invent claims; if the evidence is incomplete, it should say so.

## Existing command-line workflow

The original ingestion script remains available:

```powershell
python scripts/run_ingestion.py path\\to\\your_rfp.docx
```

For an offline storage-only test, use `--dummy-embeddings`. Those embeddings are not semantically meaningful, so do not use them for RAG answers.

## Notes

- The first normal run downloads the local `all-MiniLM-L6-v2` embedding model.
- RFPs, knowledge-base documents, generated files under `data/outputs/`, and the local ChromaDB store are ignored by Git.
- Review every generated response before using it in a proposal.
