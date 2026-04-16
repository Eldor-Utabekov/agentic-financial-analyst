"""Thin FastAPI routes for the financial analysis agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from app.llm.agent import AgentResponse
from app.llm.agent import answer_financial_question

router = APIRouter()


class AskRequest(BaseModel):
    """Request payload for the financial question endpoint."""

    question: str
    price_data: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    top_k: int = Field(default=3, gt=0)


class RetrievedContextResponse(BaseModel):
    """Retrieved context item returned by the API."""

    chunk_id: str
    source: str
    score: float
    text: str


class ToolTraceStepResponse(BaseModel):
    """Tool trace step returned by the API."""

    step: str
    status: str
    details: str


class AskResponse(BaseModel):
    """Structured response payload for the financial question endpoint."""

    model_config = ConfigDict(from_attributes=True)

    question: str
    summary: str
    supporting_signals: list[str]
    risks: list[str]
    retrieved_context: list[RetrievedContextResponse]
    tool_trace: list[ToolTraceStepResponse]


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple health response."""
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Parse a request, call the agent, and return a structured response."""
    try:
        response = answer_financial_question(
            question=request.question,
            price_data=pd.DataFrame(request.price_data),
            chunks=request.chunks,
            top_k=request.top_k,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _build_ask_response(response)


def _build_ask_response(response: AgentResponse) -> AskResponse:
    """Convert the internal dataclass response into an API response model."""
    return AskResponse.model_validate(asdict(response))
