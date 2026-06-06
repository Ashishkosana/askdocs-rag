# askdocs

A production-minded **Retrieval-Augmented Generation (RAG)** service for question
answering over your own documents — with a built-in **evaluation harness** that
measures retrieval quality *and* answer quality separately.

Most RAG demos stop at "it returned an answer." The interesting engineering is
proving the answer is *right*: that the correct document was retrieved, that the
model's answer is faithful to it, and that the system abstains when it doesn't know.
That's what this project measures.

## Why this is more than a demo

- **Separately measures retrieval and generation.** `hit@k` / `MRR` tell you whether
  the right chunk was fetched; an LLM-as-judge grades the answer's *faithfulness,
  correctness, and relevance*. When a RAG system is wrong, you can tell which half broke.
- **Abstention is tested.** Unanswerable questions are in the eval set; the system is
  scored on whether it correctly says "I don't know" instead of hallucinating.
- **Observable.** Every answer reports token usage, latency, and dollar cost.
- **Grounded by construction.** The prompt forbids outside knowledge and requires
  bracketed citations to the retrieved passages.
- **Tested & CI'd.** 19 unit tests run with no API key or model download (dependencies
  are injected), plus a GitHub Actions pipeline (lint + tests + coverage).

## Architecture

```
                   ┌──────────── index time ────────────┐
   documents ─▶ ingest ─▶ chunk ─▶ embed (local) ─▶ Chroma vector store
   (.md/.txt/.pdf)                  MiniLM                (cosine)

                   ┌──────────── query time ────────────┐
   question ─▶ embed ─▶ retrieve top-k ─▶ build grounded prompt ─▶ Claude ─▶ answer
                          (Chroma)                                  (Opus 4.8)   + citations
                                                                                 + usage/cost

                   ┌──────────── evaluation ─────────────┐
   eval set ─▶ run pipeline ─▶ retrieval metrics (hit@k, MRR)
                           └─▶ LLM-as-judge (faithfulness / correctness / relevance)
                           └─▶ abstention check on unanswerable questions
```

Embeddings run **locally** (sentence-transformers), so indexing and retrieval cost
nothing and work offline. Only answer generation and judging call the Claude API.

## Quickstart

```bash
uv sync                              # install (uses uv + Python 3.12)
cp .env.example .env                 # add your ANTHROPIC_API_KEY

uv run askdocs index data/corpus     # build the index (local, no API key)
uv run askdocs query "How much does the Growth plan cost?"
uv run askdocs serve                 # FastAPI at http://127.0.0.1:8000/docs
```

Run the evaluation against the bundled sample corpus:

```bash
uv run python eval/run_eval.py --corpus data/corpus
```

```
============================================================
RETRIEVAL
  hit@k : 1.00   (n=10)
  MRR   : 0.95
ANSWER QUALITY (LLM-as-judge, 1-5)
  faithfulness : 4.9
  correctness  : 4.8
  relevance    : 5.0
ABSTENTION (unanswerable questions)
  correct abstentions : 2/2
============================================================
```

## HTTP API

| Method | Path      | Body                              | Purpose                          |
|--------|-----------|-----------------------------------|----------------------------------|
| GET    | `/health` | —                                 | liveness                         |
| GET    | `/stats`  | —                                 | indexed chunk count + model info |
| POST   | `/ingest` | `{"path": "data/corpus"}`         | index a file or directory        |
| POST   | `/query`  | `{"question": "...", "top_k": 4}` | answer + cited contexts + usage  |

## Design decisions

- **Local embeddings, hosted generation.** Retrieval should be cheap and private;
  reasoning is where a frontier model earns its cost. Splitting the two keeps query
  cost near zero and indexing fully offline.
- **Dependency injection over globals.** The embedder, vector store, and Claude client
  are injected into `RagPipeline`, so the whole pipeline is unit-testable without a
  network or a model download (see `tests/`).
- **Cosine similarity + normalized embeddings.** Stable, scale-independent relevance.
- **Paragraph-aware chunking with overlap.** Keeps facts that straddle a boundary
  retrievable; chunk size/overlap are configurable.
- **Structured-output judge.** The judge returns a validated Pydantic schema, so eval
  scores are machine-readable rather than parsed out of prose.

## Project layout

```
src/askdocs/
  config.py         # env-driven settings (pydantic-settings)
  chunking.py       # paragraph-aware overlapping chunker
  ingest.py         # load .md/.txt/.pdf -> chunks
  embeddings.py     # local sentence-transformers embedder (Protocol-based)
  store.py          # Chroma vector store wrapper
  retrieve.py       # query embedding -> nearest chunks
  generate.py       # grounded Claude answer + citations
  observability.py  # token/cost/latency accounting
  pipeline.py       # ingest -> retrieve -> generate orchestration
  api.py            # FastAPI service
  cli.py            # `askdocs index|query|serve`
eval/
  dataset.jsonl     # questions + reference answers + unanswerable cases
  retrieval_metrics.py  # hit@k, MRR
  judge.py          # LLM-as-judge (structured output)
  run_eval.py       # runs the full evaluation, prints the report
tests/              # 19 tests, no API key / model download required
```

## Tech stack

Python 3.12 · uv · Claude (Anthropic SDK, Opus 4.8) · sentence-transformers ·
ChromaDB · FastAPI · Pydantic · pytest · ruff · GitHub Actions
