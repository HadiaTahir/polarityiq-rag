# PolarityIQ — Family Office Intelligence RAG System

A natural language query interface over a dataset of 266 real-world family office records. Built as part of the PolarityIQ Differentiator evaluation.

**Live demo:** [polarityiq-rag.streamlit.app](https://polarityiq-rag.streamlit.app/)

---

## What This Does

Ask questions in plain English and get structured, grounded answers drawn directly from the dataset:

- _"Which family offices focus on AI with check sizes above $10M?"_
- _"Show me all SFOs in the Middle East that made direct investments recently"_
- _"Which European family offices co-invest frequently in clean energy?"_
- _"Which family offices have AUM above $50B and invest in technology?"_

The system retrieves the most relevant records using vector similarity search, then generates a structured intelligence report via GPT-4o — grounded strictly in the dataset, no hallucination.

---

## Architecture

```
User query
    │
    ▼
Embed query (text-embedding-3-small)
    │
    ▼
ChromaDB vector search (cosine similarity, top-K records)
    │   ↑ AUM threshold queries use metadata pre-filter
    │   ↑ e.g. {"aum_billions": {"$gt": 50.0}}
    ▼
Assemble context (top-K document chunks)
    │
    ▼
GPT-4o (temperature 0.1, strict grounding prompt)
    │
    ▼
Structured intelligence report + expandable source cards
```

---

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Vector database | ChromaDB (PersistentClient) | Zero infrastructure cost, native Python, persists to disk |
| Embedding model | OpenAI text-embedding-3-small | 1536 dims, ~$0.008 to embed full dataset, negligible quality gap vs large |
| LLM | GPT-4o (temp 0.1) | Near-deterministic, respects grounding prompt reliably |
| Frontend | Streamlit | Zero-config deployment to Streamlit Cloud |
| Dataset | Excel (.xlsx) | 266 family office records, 31 intelligence fields |

---

## Chunking Strategy

**Row-level chunking** — each family office record becomes exactly one document chunk with all 31 fields concatenated as structured text:

```
Family Office: Walton Enterprises LLC
Type: SFO
AUM: $225B
Location: Bentonville, AR, USA
Investment Thesis: Retail, Real Estate, Public Equities...
Sector Focus: K-12 Education, Freshwater Conservation
...all 31 fields
```

Each chunk is ~400–600 tokens, well within the 8192-token embedding limit. Row-level chunking keeps retrieval explainable: every result maps to exactly one real entity.

---

## AUM-Aware Retrieval

Queries containing AUM thresholds (e.g. _"above $50B"_, _"over $100M"_) trigger a ChromaDB metadata pre-filter before semantic search. This ensures large-AUM records are included regardless of semantic similarity score.

```python
# Detected: "AUM above $50B" → {"aum_billions": {"$gt": 50.0}}
# Detected: "AUM below $10B" → {"aum_billions": {"$lt": 10.0}}
# No AUM phrase → pure semantic search, no filter applied
```

AUM strings like `$225B`, `$500M`, `$1.2T` are parsed to a numeric `aum_billions` float at ingestion time and stored as metadata alongside the document vector.

---

## Dataset

- **266 records** across USA, Europe, Middle East, Asia-Pacific, Latin America, Africa, Australia
- **31 intelligence fields** per record: identity, contact, investment profile, sector focus, co-invest frequency, recency signals, data provenance
- **Sources:** fundingstack.com, swfinstitute.org, familyofficehub.io, SEC EDGAR, company annual reports, Bloomberg/Reuters, LinkedIn public profiles
- **Validation:** three-tier system — Verified (2+ sources, ~60%), Partial (~30%), Estimated (~10%)

---

## Repository Structure

```
polarityiq-rag/
├── app.py                                  # Streamlit app — query interface
├── ingest.py                               # Standalone ingestion script (optional)
├── FamilyOffice_Intelligence_Dataset.xlsx  # 266-record dataset (Task 1 output)
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Excludes .env, chroma_db/, __pycache__
└── README.md
```

> **Note:** The `chroma_db/` directory is excluded from the repository. On first run, `app.py` automatically builds the vector database from the Excel file (~60–90 seconds). Subsequent loads use the cached collection.

---

## Local Setup

**Prerequisites:** Python 3.10+, an OpenAI API key

```bash
# 1. Clone the repo
git clone https://github.com/HadiaTahir/polarityiq-rag.git
cd polarityiq-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Run the app
streamlit run app.py
```

On first run the app will build the ChromaDB collection from the Excel file. A progress bar tracks the embedding batches. Once complete, the query interface loads automatically.

---

## Streamlit Cloud Deployment

API key is stored in Streamlit Cloud **Secrets** (not in the repository):

```toml
# .streamlit/secrets.toml (Streamlit Cloud dashboard)
OPENAI_API_KEY = "sk-..."
```

The app reads it via `st.secrets.get("OPENAI_API_KEY")` with a `.env` fallback for local development.

---

## Known Limitations

| Limitation | Impact | Upgrade Path |
|------------|--------|--------------|
| No hybrid search | AUM/country hard-filters rely on metadata pre-filter, not SQL | Migrate to Supabase pgvector |
| Email coverage ~15% | ~40 of 266 records have verified emails | Apollo / People Data Labs enrichment run |
| No query rewriting | Ambiguous queries not rephrased before embedding | GPT-4o pre-processing step |
| No re-ranking | Results ranked by embedding similarity only | Cross-encoder re-ranker |
| ChromaDB cold start | ~60–90s rebuild on first Streamlit Cloud run | Progress bar communicates this; use managed vector DB in production |

---

## Requirements

```
openai>=1.0.0
chromadb>=0.4.0
pandas>=2.0.0
openpyxl>=3.1.0
streamlit>=1.28.0
python-dotenv>=1.0.0
httpx>=0.24.0
```

---

*Built by Hadia Tahir · PolarityIQ Differentiator Task 2 · 2026*
