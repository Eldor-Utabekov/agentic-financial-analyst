"""Application entrypoint for the FastAPI service."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Agentic Financial Analyst")
app.include_router(router)
