# Agentic Financial Analyst

Production-style backend system for grounded financial question answering, combining deterministic analytics, retrieval, agent orchestration, and optional provider-backed LLM generation.

- deterministic ETF analytics  
- local document retrieval (RAG baseline)  
- inspectable agent orchestration  
- optional LLM integration with safe fallback  
- minimal FastAPI service  

---

## Why This Project Exists

This project demonstrates how to build **AI-powered systems in a controlled, production-oriented way**, without relying on opaque frameworks.

Instead of “black-box AI”, it emphasizes:
- deterministic computation first  
- retrieval grounding  
- transparent orchestration  
- strict validation and testability  

This reflects real-world GenAI system design: **LLMs as optional components, not the core logic.**

---

Given:
- ETF price data  
- related text context  

the system:
- computes financial signals (returns, volatility, momentum, drawdown)  
- retrieves relevant context  
- generates a grounded answer (deterministic or LLM-backed)  

All steps are transparent, testable, and inspectable.

---

## Current Status

### Implemented
- ETF market data retrieval utility  
- deterministic analytics (returns, volatility, momentum, drawdown)  
- local-first document ingestion and chunking  
- lexical retrieval baseline (RAG)  
- agent orchestration layer (fully inspectable)  
- optional provider-backed LLM answer generation  
- deterministic fallback when LLM is unavailable  
- FastAPI API (`/health`, `/ask`)  
- unit + integration tests  
- lightweight evaluation framework  

### Not implemented (by design)
- embedding-based retrieval / vector DB  
- large-scale evaluation datasets  
- production deployment infrastructure  

---

## Architecture Overview

Request flow:

1. Client sends:
   - question  
   - price data  
   - local text chunks  

2. API layer:
   - validates input via Pydantic  
   - converts price data → pandas DataFrame  

3. Retrieval:
   - lexical scoring of chunks  
   - top-k selection  

4. Deterministic analytics:
   - returns  
   - momentum  
   - volatility  
   - drawdown  

5. Answer generation:
   - deterministic grounded summary  
   - optionally upgraded via LLM (if configured)  
   - safe fallback if provider fails  

6. Response:
   - summary  
   - supporting signals  
   - risks  
   - retrieved context  
   - tool trace (full transparency)  

---

## Key Skills Demonstrated

**Backend & Engineering**
- FastAPI service design  
- Pydantic validation  
- modular architecture  
- Docker-ready service  

**Data Science**
- financial time-series analysis  
- feature computation (returns, volatility, momentum)  
- signal interpretation  

**GenAI / NLP**
- Retrieval-Augmented Generation (RAG)  
- prompt construction  
- grounded answer generation  
- safe LLM fallback design  
- evaluation of AI outputs  

**Systems Thinking**
- deterministic-first design  
- inspectable agent orchestration  
- failure-safe LLM integration  

---

## Project Structure

```text
app/
  api/
    routes.py
  llm/
    agent.py
    client.py
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

---

## Setup

Use Python 3.11.

### Create environment

```bash
python -m venv .venv
```

**Windows**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Configuration

Optional LLM integration:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_BASE_URL=https://api.openai.com/v1/chat/completions
```

If not provided → system runs fully deterministic.

---

## Run Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## Run Tests

```bash
python -m unittest discover -s tests
```

---

## Quick Manual Test

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What signals matter for this ETF?",
    "price_data": [
      {"date": "2024-01-01", "close": 100.0},
      {"date": "2024-01-02", "close": 102.0},
      {"date": "2024-01-03", "close": 101.0},
      {"date": "2024-01-04", "close": 104.0},
      {"date": "2024-01-05", "close": 106.0}
    ],
    "chunks": [
      {
        "chunk_id": "chunk-1",
        "source": "doc1.txt",
        "text": "ETF momentum improved this week.",
        "metadata": {}
      }
    ],
    "top_k": 2
  }'
```

If response contains:
- summary  
- signals  
- tool_trace  

→ everything works correctly.

---

## Docker

Build:

```bash
docker build -t financial-agent .
```

Run:

```bash
docker run -p 8000:8000 financial-agent
```

---

## Evaluation

`app/llm/evaluator.py` provides local evaluation utilities:

Checks:
- successful execution  
- summary presence  
- tool trace integrity  
- signal and risk generation  
- expected orchestration steps  

This is a **practical regression tool**, not a benchmarking framework.

---

## Current Limitations

- retrieval = lexical only (no embeddings)  
- no vector database  
- API requires pre-supplied data  
- deterministic summaries are intentionally simple  
- no production infra (CI/CD, cloud deployment)  

---

## Next Steps

- embedding-based retrieval (FAISS / vector DB)  
- LLM abstraction layer  
- richer evaluation datasets  
- cloud deployment (GCP / AWS)  
- streaming responses  

---

## Development Approach

This project was built with **assistance from generative AI tools** for iteration and speed.

All key decisions (architecture, validation, system boundaries) were made intentionally with focus on:
- clarity  
- correctness  
- minimalism  
- avoiding overengineering  

AGENTS.md  
https://github.com/Eldor-Utabekov/agentic-financial-analyst/blob/main/AGENTS.md