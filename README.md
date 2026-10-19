# Self-RAG Studio

Document question answering you can check. Ask questions about PDFs, Word and PowerPoint files, spreadsheets and
source code. Every answer is cited, scored by the system itself, and verified **sentence by sentence** against the
evidence it came from. Unsupported sentences are flagged (and can be repaired), and figures are shown when they help.

> **Not the Asai et al. "Self-RAG".** That method fine-tunes a model to emit reflection tokens. This project uses
> prompting-based self-evaluation, a reflect-and-retry loop, and an independent verifier on top of ordinary models.

## Features

| Area | What it does |
|---|---|
| Retrieval | Parent/child chunks, TF-IDF + dense search, optional chunk-level **BM25 + dense fusion (RRF)**, cross-encoder rerank, follow-up question rewriting |
| Self-checking | LLM self-evaluation (relevance, grounding, quality), retry with a rewritten query, **cascaded sentence verification** (embeddings first, LLM only when unsure) and **sentence-level repair** |
| Documents | PDF (tables, figures, optional vision pass for formulas and charts), Word, PowerPoint, Excel/CSV (exact pandas calculations), code files, notebooks, zip projects |
| Answers | Source citations (page, table, file and lines), related figures, "figure only" requests, Markdown tables, code and LaTeX |
| Study tools | Quiz mode, automatic summary / entities / tags, direct Chat with the model (clearly marked as not from your documents) |
| Evaluation Lab | Benchmark, ablation ladder A0–A5 with confidence intervals, calibration (ECE, AURC), LaTeX/CSV export |

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/<your-username>/selfrag-studio.git
cd selfrag-studio
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # then put your Groq key in .env
python app.py                 # open http://127.0.0.1:5000
```

You need a free API key from <https://console.groq.com>. The first start downloads the embedding and reranker models.

## Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required |
| `GEN_MODEL` / `JUDGE_MODEL` | Answer model and (stronger, different) judge model |
| `VISION_MODEL` | Optional vision model for "Deep analysis"; auto-detected if empty |
| `MAX_VISION_PAGES`, `VISION_ALL_PAGES` | Limit how many pages go to the vision model |
| `EMBED_MODEL` | Changing it requires re-indexing every document |

Check <https://console.groq.com/docs/models> for the models currently available to your account.

## Tests

The tests run offline (a fake embedder and fake LLM live in `tests/stubs`), so no API key is needed:

```bash
python tests/test_research.py
python tests/test_app.py
python tests/test_multimodal.py
python tests/test_routes_multimodal.py
python tests/test_vision.py
```

They verify wiring and logic. They do not measure answer quality with a real model; use the Lab for that.

## Project layout

```
app.py              Flask routes and the streaming pipeline
indexer.py          chunking and vector indexing        ingest.py   file parsing (PDF, Word, sheets, code...)
retriever.py        retrieval and reranking             research.py hybrid retrieval, verification, ablation
generator.py        prompts and generation              evaluator.py self-evaluation
multimodal.py       sources and figures                 tabular.py  safe spreadsheet calculations
enrich.py, quiz.py  summaries, tags, quizzes            conversation.py memory and follow-ups
static/, templates/ web interface                       docs/       change log and research plan
```

## Privacy

Documents and chats stay on your machine. Text sent for answering, evaluation and (optionally) vision goes to Groq's API.
The `.env` file, databases, vector store, uploads and extracted figures are excluded from git by `.gitignore`.

## Research

See `docs/RESEARCH.md` for the experimental protocol and `docs/CHANGES.md` for the change log.

## License

MIT. See `LICENSE`.
