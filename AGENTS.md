# AGENTS.md

## Project

This repository contains a portfolio project called **Agentic Financial Analyst**.

The goal is to build a production-style Python backend that answers financial analysis questions by combining:
- structured ETF market data,
- calculated financial and risk metrics,
- retrieved unstructured financial context,
- LLM-based tool use,
- a FastAPI interface.

Prioritize clarity, correctness, modularity, and recruiter-readable implementation over unnecessary complexity.

---

## Build order

Unless explicitly instructed otherwise, build in this order:

1. Core financial tools
2. Analytics and risk metrics
3. RAG ingestion and retrieval
4. Agent orchestration
5. FastAPI endpoints
6. Evaluation utilities
7. Containerization and polish

Do not skip ahead before earlier layers work.

---

## Rules

- Use Python 3.11-compatible code.
- Keep dependencies minimal.
- Prefer simple, maintainable implementations.
- Do not add frontend code unless explicitly requested.
- Do not add cloud deployment unless explicitly requested.
- Do not add multi-agent complexity unless explicitly requested.
- Do not hardcode secrets or API keys.
- Use environment variables for credentials and model settings.
- Do not fabricate financial conclusions; outputs must be grounded in tool results or retrieved context.

---

## Preferred structure

- `app/core/` for config and logging
- `app/tools/` for deterministic utilities
- `app/rag/` for ingestion and retrieval
- `app/llm/` for prompts and orchestration
- `app/api/` for FastAPI routes and schemas
- `tests/` for automated tests
- `data/` for local sample data
- `notebooks/` only for exploration

Keep business logic out of notebooks.

---

## Coding style

- Use small, focused functions.
- Add docstrings to public functions.
- Use descriptive names.
- Prefer explicit code over hidden magic.
- Add type hints where useful.
- Avoid unnecessary abstraction.
- Reuse existing modules before creating new ones.
- Avoid touching unrelated files.

---

## Dependency policy

Before adding a dependency:
- prefer the standard library when practical,
- prefer existing project dependencies when practical,
- avoid large frameworks unless central to the project.

If adding a dependency, explain why in the final summary.

---

## Task workflow

When working on a task:
1. Inspect relevant files first.
2. Make the smallest clean change that solves the task.
3. Keep changes logically grouped.
4. Add or update tests when practical.
5. Run relevant verification commands.
6. Summarize what changed and any remaining risks.

Do not rewrite large parts of the codebase unless explicitly asked.

---

## Verification

Do not claim code works unless it was actually verified.

After meaningful changes:
- run relevant tests if they exist,
- run the smallest useful verification command,
- report what was checked and what was not checked.

---

## API expectations

When adding API code:
- use Pydantic models,
- keep routes thin,
- keep business logic outside route files,
- return structured JSON,
- include basic error handling.

---

## LLM expectations

When adding LLM features:
- separate deterministic tools from orchestration,
- keep prompts in dedicated modules when practical,
- ground outputs in tool results or retrieved context,
- prefer inspectable workflows over opaque chains.

---

## RAG expectations

When adding retrieval:
- keep ingestion separate from query-time retrieval,
- keep chunking simple and inspectable,
- store source metadata,
- make retrieved context traceable in outputs.

---

## Definition of done

A task is done when:
- the requested change is implemented,
- the code matches project structure,
- affected behavior is verified as much as practical,
- tests were added or updated when appropriate,
- the final summary explains changes, checks, and limitations.