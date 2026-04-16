"""Minimal, inspectable agent orchestration for financial analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from app.llm.client import generate_grounded_answer_result
from app.rag.ingest import Chunk
from app.rag.retriever import ScoredChunk
from app.rag.retriever import retrieve_top_k
from app.tools.analytics import calculate_drawdown
from app.tools.analytics import calculate_max_drawdown
from app.tools.analytics import calculate_momentum
from app.tools.analytics import calculate_rolling_volatility
from app.tools.analytics import calculate_simple_returns

REQUIRED_PRICE_COLUMNS: Final[set[str]] = {"close"}


@dataclass(frozen=True, slots=True)
class RetrievedContextItem:
    """Retrieved context item included in the final response."""

    chunk_id: str
    source: str
    score: float
    text: str


@dataclass(frozen=True, slots=True)
class ToolTraceStep:
    """Single orchestration step recorded for inspection."""

    step: str
    status: str
    details: str


@dataclass(frozen=True, slots=True)
class AgentResponse:
    """Structured response for a grounded financial analysis answer."""

    question: str
    summary: str
    supporting_signals: list[str]
    risks: list[str]
    retrieved_context: list[RetrievedContextItem]
    tool_trace: list[ToolTraceStep]


def answer_financial_question(
    question: str,
    price_data: pd.DataFrame,
    chunks: list[Chunk],
    top_k: int = 3,
) -> AgentResponse:
    """Answer a financial question using deterministic signals and local retrieval."""
    validated_question = _validate_question(question)
    validated_top_k = _validate_top_k(top_k)
    validated_price_data = _validate_price_data(price_data)
    price_series = extract_price_series(validated_price_data)
    retrieved_chunks = retrieve_top_k(validated_question, chunks, k=validated_top_k)
    supporting_signals, risks = summarize_financial_signals(price_series)
    if not retrieved_chunks:
        risks = [*risks, "No relevant local context was retrieved for this question."]

    retrieved_context = [build_retrieved_context_item(chunk) for chunk in retrieved_chunks]
    deterministic_summary = build_summary(price_series, supporting_signals, retrieved_context)
    summary_result = generate_grounded_answer_result(
        question=validated_question,
        supporting_signals=supporting_signals,
        retrieved_context_texts=[item.text for item in retrieved_context],
        fallback_summary=deterministic_summary,
    )
    tool_trace = build_tool_trace(
        price_observation_count=len(price_series),
        requested_top_k=validated_top_k,
        retrieved_count=len(retrieved_context),
        signal_count=len(supporting_signals),
        summary_source=summary_result.source,
    )

    return build_agent_response(
        question=validated_question,
        summary=summary_result.summary,
        supporting_signals=supporting_signals,
        risks=risks,
        retrieved_context=retrieved_context,
        tool_trace=tool_trace,
    )


def extract_price_series(price_data: pd.DataFrame) -> pd.Series:
    """Extract a close-price series from normalized price data."""
    sorted_price_data = price_data.sort_values("date") if "date" in price_data.columns else price_data.copy()
    return sorted_price_data["close"].astype(float).reset_index(drop=True)


def summarize_financial_signals(prices: pd.Series) -> tuple[list[str], list[str]]:
    """Summarize deterministic financial signals from a price series."""
    supporting_signals: list[str] = []
    risks: list[str] = []

    simple_returns = calculate_simple_returns(prices).dropna()
    if not simple_returns.empty:
        latest_return = float(simple_returns.iloc[-1])
        supporting_signals.append(f"Latest simple return: {_format_percentage(latest_return)}.")
        if latest_return < 0:
            risks.append("The most recent observed return was negative.")

    if len(prices) > 1:
        momentum_window = min(20, len(prices) - 1)
        momentum_series = calculate_momentum(prices, window=momentum_window).dropna()
        if not momentum_series.empty:
            latest_momentum = float(momentum_series.iloc[-1])
            supporting_signals.append(
                f"{momentum_window}-period momentum: {_format_percentage(latest_momentum)}."
            )
            if latest_momentum < 0:
                risks.append("Momentum is negative over the available lookback window.")

    if len(simple_returns) >= 2:
        volatility_window = min(20, len(simple_returns))
        volatility_series = calculate_rolling_volatility(
            simple_returns,
            window=volatility_window,
            annualize=True,
        ).dropna()
        if not volatility_series.empty:
            latest_volatility = float(volatility_series.iloc[-1])
            supporting_signals.append(f"Annualized rolling volatility: {_format_percentage(latest_volatility)}.")
            if latest_volatility > 0.30:
                risks.append("Recent annualized volatility is elevated.")

    latest_drawdown = float(calculate_drawdown(prices).iloc[-1])
    max_drawdown = calculate_max_drawdown(prices)
    supporting_signals.append(f"Current drawdown: {_format_percentage(latest_drawdown)}.")
    supporting_signals.append(f"Max drawdown over the sample: {_format_percentage(max_drawdown)}.")
    if max_drawdown < -0.10:
        risks.append("Historical drawdown exceeded 10% over the provided sample.")

    if not risks:
        risks.append("No material deterministic risk flag was triggered by the current thresholds.")

    return supporting_signals, risks


def build_tool_trace(
    price_observation_count: int,
    requested_top_k: int,
    retrieved_count: int,
    signal_count: int,
    summary_source: str,
) -> list[ToolTraceStep]:
    """Build a simple trace of orchestration steps."""
    return [
        ToolTraceStep(step="validate_question", status="completed", details="Validated non-empty user question."),
        ToolTraceStep(
            step="extract_price_series",
            status="completed",
            details=f"Extracted close-price series with {price_observation_count} observation(s).",
        ),
        ToolTraceStep(
            step="retrieve_context",
            status="completed",
            details=f"Requested top {requested_top_k} chunk(s) and retrieved {retrieved_count} relevant match(es).",
        ),
        ToolTraceStep(
            step="summarize_financial_signals",
            status="completed",
            details=f"Derived {signal_count} deterministic financial signal(s).",
        ),
        ToolTraceStep(
            step="build_response",
            status="completed",
            details=f"Built grounded structured response using {summary_source}.",
        ),
    ]


def build_agent_response(
    question: str,
    summary: str,
    supporting_signals: list[str],
    risks: list[str],
    retrieved_context: list[RetrievedContextItem],
    tool_trace: list[ToolTraceStep],
) -> AgentResponse:
    """Build the final typed response object."""
    return AgentResponse(
        question=question,
        summary=summary,
        supporting_signals=supporting_signals,
        risks=risks,
        retrieved_context=retrieved_context,
        tool_trace=tool_trace,
    )


def build_retrieved_context_item(chunk: ScoredChunk) -> RetrievedContextItem:
    """Convert a scored chunk into response context."""
    return RetrievedContextItem(
        chunk_id=chunk["chunk_id"],
        source=chunk["source"],
        score=chunk["score"],
        text=chunk["text"],
    )


def build_summary(
    prices: pd.Series,
    supporting_signals: list[str],
    retrieved_context: list[RetrievedContextItem],
) -> str:
    """Build a concise grounded summary from signals and retrieved context."""
    latest_price = float(prices.iloc[-1])
    signal_clause = _build_signal_clause(supporting_signals)
    context_clause = (
        f"Retrieved {len(retrieved_context)} relevant local context item(s) from {_format_context_sources(retrieved_context)}."
        if retrieved_context
        else "No directly relevant local context was retrieved."
    )
    return f"Latest close is {latest_price:.2f}. {signal_clause} {context_clause}"


def _validate_question(question: str) -> str:
    """Validate a user question passed to the agent."""
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty.")
    return normalized_question


def _validate_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Validate required price data input for orchestration."""
    if not isinstance(price_data, pd.DataFrame):
        raise TypeError("price_data must be a pandas DataFrame.")
    if price_data.empty:
        raise ValueError("price_data must not be empty.")

    missing_columns = REQUIRED_PRICE_COLUMNS - set(price_data.columns)
    if missing_columns:
        missing_columns_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"price_data is missing required columns: {missing_columns_str}.")

    if not pd.api.types.is_numeric_dtype(price_data["close"]):
        raise TypeError("price_data close column must contain numeric values.")
    if price_data["close"].isna().any():
        raise ValueError("price_data close column must not contain missing values.")
    if (price_data["close"] <= 0).any():
        raise ValueError("price_data close column must contain only positive values.")

    return price_data.copy()


def _validate_top_k(top_k: int) -> int:
    """Validate top-k retrieval configuration for orchestration."""
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    return top_k


def _build_signal_clause(supporting_signals: list[str]) -> str:
    """Build a concise signal summary for the final response text."""
    if not supporting_signals:
        return "No deterministic signal was available."
    if len(supporting_signals) == 1:
        return supporting_signals[0]
    return f"{supporting_signals[0]} {supporting_signals[1]}"


def _format_context_sources(retrieved_context: list[RetrievedContextItem]) -> str:
    """Format the sources behind retrieved context for summary text."""
    unique_sources = list(dict.fromkeys(item.source for item in retrieved_context))
    if len(unique_sources) == 1:
        return unique_sources[0]
    if len(unique_sources) == 2:
        return f"{unique_sources[0]} and {unique_sources[1]}"
    return f"{', '.join(unique_sources[:2])}, and others"


def _format_percentage(value: float) -> str:
    """Format a numeric ratio as a percentage string."""
    return f"{value:.2%}"
