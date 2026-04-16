"""Lightweight evaluation helpers for the local financial agent."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.llm.agent import answer_financial_question
from app.rag.ingest import Chunk


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Single local evaluation case for the agent."""

    question: str
    price_data: pd.DataFrame
    chunks: list[Chunk]
    top_k: int = 3
    expected_tool_steps: list[str] | None = None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Structured result for a single agent evaluation."""

    question: str
    success: bool
    tool_trace_present: bool
    retrieved_context_count: int
    supporting_signal_count: int
    risk_count: int
    summary_present: bool
    expected_tools_matched: bool | None
    notes: list[str]


def evaluate_single_case(
    question: str,
    price_data: pd.DataFrame,
    chunks: list[Chunk],
    top_k: int = 3,
    expected_tool_steps: list[str] | None = None,
) -> EvaluationResult:
    """Evaluate a single question against the local agent workflow."""
    notes: list[str] = []
    normalized_question = _normalize_evaluation_question(question)

    try:
        response = answer_financial_question(
            question=question,
            price_data=price_data,
            chunks=chunks,
            top_k=top_k,
        )
    except (TypeError, ValueError) as exc:
        notes.append(str(exc))
        return EvaluationResult(
            question=normalized_question,
            success=False,
            tool_trace_present=False,
            retrieved_context_count=0,
            supporting_signal_count=0,
            risk_count=0,
            summary_present=False,
            expected_tools_matched=False if expected_tool_steps is not None else None,
            notes=notes,
        )

    tool_trace_present = bool(response.tool_trace)
    summary_present = bool(response.summary.strip())
    supporting_signal_count = len(response.supporting_signals)
    risk_count = len(response.risks)
    retrieved_context_count = len(response.retrieved_context)
    expected_tools_matched = _check_expected_tool_steps(response.tool_trace, expected_tool_steps)

    if not summary_present:
        notes.append("Summary was empty.")
    if not tool_trace_present:
        notes.append("Tool trace was missing.")
    if supporting_signal_count == 0:
        notes.append("No supporting signals were produced.")
    if risk_count == 0:
        notes.append("No risks were produced.")
    if expected_tools_matched is False:
        notes.append("Tool trace did not match expected steps.")

    success = not notes

    return EvaluationResult(
        question=normalized_question,
        success=success,
        tool_trace_present=tool_trace_present,
        retrieved_context_count=retrieved_context_count,
        supporting_signal_count=supporting_signal_count,
        risk_count=risk_count,
        summary_present=summary_present,
        expected_tools_matched=expected_tools_matched,
        notes=notes,
    )


def evaluate_batch(cases: list[EvaluationCase]) -> list[EvaluationResult]:
    """Evaluate a batch of local agent cases."""
    if not isinstance(cases, list):
        raise TypeError("cases must be a list.")

    results: list[EvaluationResult] = []
    for case in cases:
        if not isinstance(case, EvaluationCase):
            raise TypeError("Each case must be an EvaluationCase.")
        results.append(
            evaluate_single_case(
                question=case.question,
                price_data=case.price_data,
                chunks=case.chunks,
                top_k=case.top_k,
                expected_tool_steps=case.expected_tool_steps,
            )
        )

    return results


def _check_expected_tool_steps(
    tool_trace: list[object],
    expected_tool_steps: list[str] | None,
) -> bool | None:
    """Check whether observed tool trace steps match expected names."""
    if expected_tool_steps is None:
        return None

    observed_steps = [getattr(step, "step", None) for step in tool_trace]
    return observed_steps == expected_tool_steps


def _normalize_evaluation_question(question: object) -> str:
    """Normalize the recorded evaluation question to a string."""
    if isinstance(question, str):
        return question
    return str(question)
