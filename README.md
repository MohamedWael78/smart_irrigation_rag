# \U0001f331 AquaMind AI — Smart Irrigation Agentic RAG System
Built by: Mohamed Wael (AI Engineer)

## \U0001f680 Overview
An AI agent for smart irrigation design combining domain-specific document
retrieval, live agronomic data, and agentic tool use.

## \U0001f9e0 Architecture
- **Ingestion**: PDFs (FAO-56, manufacturer manuals) → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000/150) → page-level metadata preserved
- **Storage**: HuggingFace `all-MiniLM-L6-v2` embeddings → ChromaDB
- **Retrieval**: Hybrid `EnsembleRetriever` (60% vector / 40% BM25) → FlashRank rerank → top-k
- **Orchestration**: LangChain tool-calling agent, with conversation memory across turns
- **LLM**: Llama 3.1 8B via Groq, streamed token-by-token
- **UI**: Streamlit with a custom-component citation layer

## \U0001f6e0\ufe0f Agent Tools
| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Hybrid + reranked semantic search over ingested documents, with page-level citations |
| `calculate_drip_irrigation` | Hydraulic calculator: flow rate + water volume |
| `get_reference_evapotranspiration` | Live FAO-56 Penman-Monteith ET0 forecast (Open-Meteo, no API key) for a field's coordinates |
| `lookup_crop_coefficient` | Structured FAO-56 Table 12 Kc lookup (deterministic, bypasses RAG noise for a purely tabular fact) |

## \u2728 What changed vs. the original version
- **Fixed a chunking bug**: `chunk_overlap` was equal to `chunk_size` (100% duplication) — now a sane 15%.
- **Conversation memory**: the agent now receives `chat_history`, so follow-up questions retain context (previously every turn was stateless).
- **Streaming**: answers render token-by-token via `astream_events`, plus live "using tool: X" status while the agent works.
- **Page-level citations**: citation cards show source file, exact page number, and a snippet — not just a filename list.
- **Two new tools**: live ET0 (real weather-driven irrigation demand) and a structured Kc lookup table.
- **Design tokens**: colors/fonts centralized as CSS variables; added a monospace utility face for numeric readouts; added a restrained animated "flow line" signature element.
- **Eval harness** (`eval.py`): RAGAS-based faithfulness/relevancy/precision/recall scoring, extendable with your own Q&A pairs.
- **Optional LangSmith tracing** for observability, enabled by setting `LANGCHAIN_API_KEY` in secrets.

## \U0001f4bb How to Run Locally
1. Install requirements: `pip install -r requirements.txt`
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set your `GROQ_API_KEY`
3. Put your source PDFs in `./data`
4. Ingest data: `python ingest.py`
5. Run the app: `streamlit run app.py`

## \U0001f9ea Running the evaluation suite
```
export GROQ_API_KEY=your-key
python eval.py
```
Edit `TEST_SET` in `eval.py` with real question/ground-truth pairs from your
own documents as you go — treat it as a regression suite you re-run whenever
you touch chunking, retrieval weights, or the rerank cutoff.

## \U0001f5fa\ufe0f Suggested next steps (not yet implemented)
- Persist chat sessions to SQLite/Postgres instead of `st.session_state` for multi-user support
- User feedback capture (\U0001f44d/\U0001f44e per answer) to build a retrieval-tuning dataset
- Full React/Next.js frontend if you outgrow Streamlit's styling ceiling
- Semantic response caching for repeated/near-duplicate queries
