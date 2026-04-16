# Agentic Financial Analyst

Production-style backend system for grounded financial question answering, combining deterministic analytics, retrieval, agent orchestration, and optional provider-backed answer generation.

- deterministic ETF analytics
- local document retrieval
- an inspectable orchestration layer
- a minimal FastAPI API

## Why This Project Exists

This project is designed to demonstrate practical backend engineering for AI-assisted financial analysis without hiding the logic behind heavyweight frameworks or opaque chains.

The focus is on:
- clear deterministic tools
- grounded retrieval
- readable orchestration
- testable service boundaries

## Current Status

Implemented now:
- ETF market data retrieval utility
- deterministic analytics for returns, volatility, momentum, and drawdown
- local-first text ingestion and chunking
- deterministic lexical retrieval
- provider-agnostic orchestration layer with deterministic fallback behavior
- optional provider-backed summary generation
- FastAPI `health` and `ask` endpoints
- lightweight local evaluation utilities

Not implemented yet:
- embeddings or vector database retrieval
- richer evaluation datasets
- deployment infrastructure beyond a simple local Docker setup

## Architecture Overview

The current request flow is:

1. A client sends a question, price data, and local text chunks to `POST /ask`.
2. The API validates the payload with Pydantic and converts `price_data` into a pandas `DataFrame`.
3. The agent orchestration layer retrieves relevant context with the baseline lexical retriever.
4. Deterministic analytics are computed from the price series.
5. The agent produces a deterministic grounded summary and can optionally upgrade that summary through a provider-backed LLM call when configured.
6. A structured grounded response is returned with supporting signals, risks, retrieved context, and tool trace.

## Key Skills Demonstrated

- Backend system design (FastAPI, modular architecture)
- Building agent-style orchestration without frameworks
- Retrieval-Augmented Generation (RAG) fundamentals
- Deterministic financial analytics
- API design and validation with Pydantic
- Testing (unit + integration)
- Evaluation of AI systems

## Current Features

- `app/tools/market_data.py`
  Fetches and normalizes ETF OHLCV history.
- `app/tools/analytics.py`
  Computes simple returns, log returns, rolling volatility, momentum, drawdown, and max drawdown.
- `app/rag/ingest.py`
  Loads local `.txt` and `.md` documents and builds simple overlapping chunks.
- `app/rag/retriever.py`
  Ranks chunks with deterministic lexical overlap scoring.
- `app/llm/agent.py`
  Orchestrates retrieval plus deterministic analytics into a structured answer.
- `app/llm/client.py`
  Optionally calls an OpenAI-compatible endpoint for grounded answer generation and falls back to deterministic output when no API key is configured.
- `app/llm/evaluator.py`
  Evaluates agent runs with lightweight local checks.
- `app/api/routes.py`
  Exposes `GET /health` and `POST /ask`.

## Project Structure

```text
app/
  api/
    routes.py
  llm/
    agent.py
    evaluator.py
    prompts.py
  rag/
    ingest.py
    retriever.py
  tools/
    analytics.py
    market_data.py
  main.py
tests/
  test_agent.py
  test_analytics.py
  test_api.py
  test_evaluator.py
  test_market_data.py
  test_rag_ingest.py
  test_rag_retriever.py
```

## Setup

Use Python 3.11.

### Create an environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup

Copy `.env.example` to `.env` if you want a local environment file:

```bash
cp .env.example .env
```

Current runtime configuration is intentionally minimal. No live provider key is required for the implemented phases.

Optional provider-backed answer generation can be enabled with:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
```

## Run Locally

Start the API with Uvicorn:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Run Tests

Run the full local test suite:

```bash
python -m unittest discover -s tests
```

## Run with Docker

Build the image:

```bash
docker build -t financial-agent .


## API Usage

### Health Check

Request:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "ok"
}
```

### Ask Endpoint

Request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What signals matter for this ETF?",
    "price_data": [
      {"date": "2024-01-01", "close": 100.0, "symbol": "SPY"},
      {"date": "2024-01-02", "close": 102.0, "symbol": "SPY"},
      {"date": "2024-01-03", "close": 101.0, "symbol": "SPY"},
      {"date": "2024-01-04", "close": 104.0, "symbol": "SPY"},
      {"date": "2024-01-05", "close": 106.0, "symbol": "SPY"}
    ],
    "chunks": [
      {
        "chunk_id": "chunk-1",
        "source": "doc1.txt",
        "text": "ETF momentum improved as inflows increased this week.",
        "metadata": {"file_name": "doc1.txt"}
      },
      {
        "chunk_id": "chunk-2",
        "source": "doc2.txt",
        "text": "Drawdown risk remains manageable for diversified ETFs.",
        "metadata": {"file_name": "doc2.txt"}
      }
    ],
    "top_k": 2
  }'
```

Response shape:

```json
{
  "question": "What signals matter for this ETF?",
  "summary": "Latest close is 106.00. ...",
  "supporting_signals": ["..."],
  "risks": ["..."],
  "retrieved_context": [
    {
      "chunk_id": "chunk-1",
      "source": "doc1.txt",
      "score": 0.5,
      "text": "ETF momentum improved as inflows increased this week."
    }
  ],
  "tool_trace": [
    {
      "step": "validate_question",
      "status": "completed",
      "details": "Validated non-empty user question."
    }
  ]
}
```

## Evaluation Overview

`app/llm/evaluator.py` provides a small local evaluation layer for the current agent. It checks:
- whether the orchestration completed successfully
- whether a summary is present
- whether a tool trace exists
- how many supporting signals, risks, and retrieved context items were produced
- whether expected tool steps matched, when provided

This is intentionally a practical utility for regression checks, not a benchmarking framework.

## Current Limitations

- Retrieval is simple lexical overlap, not embedding-based semantic search.
- Provider-backed answer generation is optional and only runs when environment configuration is supplied.
- The API expects callers to provide price data and chunks directly.
- The current answer synthesis is deterministic and intentionally conservative.
- No persistence layer or production deployment configuration is included beyond the simple local container.

## Next Steps

- add a local vector store or embedding-backed retrieval layer
- integrate an LLM provider behind the existing orchestration boundary
- add richer evaluation cases and datasets
- expand API schemas and error reporting as the service surface grows

## Development Approach

This project was developed with the assistance of generative AI tools for iterative coding and refinement.

The system design, architecture, and implementation decisions were driven intentionally, with a focus on clarity, testability, and avoiding unnecessary abstraction.

AGENTS.md - https://github.com/Eldor-Utabekov/agentic-financial-analyst/blob/main/AGENTS.md
