# PLANS.md

## Project name

Agentic Financial Analyst

## Project goal

Build a production-style GenAI backend that answers financial analysis questions by combining:
- ETF market data
- calculated financial metrics
- retrieved financial text context
- LLM tool orchestration
- FastAPI endpoints

This project is designed to demonstrate:
- agentic AI patterns
- LLM orchestration
- RAG
- Python backend engineering
- production-style structure and evaluation

---

## Target user flow

A user asks a question such as:

> Which ETFs look promising this week and why?

The system should:
1. understand the question,
2. decide which tools are needed,
3. fetch market data,
4. calculate metrics,
5. retrieve relevant financial context,
6. synthesize a grounded answer,
7. return a structured response.

---

## Current architecture direction

Planned modules:

- `app/core/`
  - config
  - logging

- `app/tools/`
  - market data retrieval
  - feature engineering
  - risk metrics

- `app/rag/`
  - document ingestion
  - chunking
  - embeddings
  - retrieval

- `app/llm/`
  - prompts
  - orchestration
  - evaluation helpers

- `app/api/`
  - FastAPI routes
  - request/response schemas

- `tests/`
  - unit and integration tests

---

## Build phases

### Phase 1 — Core financial tools
Implement:
- ETF price retrieval
- return calculation
- rolling volatility
- momentum
- drawdown
- basic risk metrics

Deliverables:
- `app/tools/market_data.py`
- `app/tools/analytics.py`
- initial tests

Status: not started

---

### Phase 2 — Retrieval layer
Implement:
- local document loader
- chunking
- embeddings
- vector store
- retriever

Deliverables:
- `app/rag/ingest.py`
- `app/rag/retriever.py`
- `app/rag/vector_store.py`

Status: not started

---

### Phase 3 — Agent orchestration
Implement:
- question handling
- tool selection
- multi-step execution
- grounded final answer generation

Deliverables:
- `app/llm/agent.py`
- `app/llm/prompts.py`

Status: not started

---

### Phase 4 — API layer
Implement:
- FastAPI app
- health endpoint
- ask endpoint
- request/response schemas

Deliverables:
- `app/main.py`
- `app/api/routes.py`

Status: not started

---

### Phase 5 — Evaluation
Implement:
- example benchmark questions
- output validation
- tool usage logging
- retrieval usage checks

Deliverables:
- `app/llm/evaluator.py`
- evaluation dataset

Status: not started

---

### Phase 6 — Packaging and polish
Implement:
- `.env.example`
- `requirements.txt`
- `Dockerfile`
- polished `README.md`

Status: not started

---

## Non-goals for initial version

Do not prioritize these yet:
- frontend UI
- live cloud deployment
- multi-agent design
- streaming responses
- advanced observability stack
- large-scale production infra

---

## Engineering principles

- Build incrementally.
- Keep scope tight.
- Prefer correctness over novelty.
- Prefer modular code over notebook logic.
- Keep every phase demonstrable.
- Each phase should end in a clean git commit.

---

## Suggested commit flow

- `Initial project setup`
- `Add ETF market data retrieval tool`
- `Add analytics and risk metrics`
- `Implement local RAG pipeline`
- `Add tool-using financial agent`
- `Expose FastAPI endpoints`
- `Add evaluation workflow`
- `Add Docker and project documentation`

---

## Open decisions

To finalize later:
- exact LLM provider
- exact embedding provider
- FAISS vs Chroma
- live news source vs local sample corpus

For now, prefer the simplest working option.