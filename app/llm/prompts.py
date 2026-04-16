"""Lightweight prompt helpers for future grounded LLM integration."""

from __future__ import annotations

SYSTEM_GROUNDING_PROMPT = (
    "You are a financial analysis assistant. Base every claim on tool outputs and retrieved context. "
    "Do not make unsupported claims, and keep explanations concise and grounded."
)


def build_grounded_answer_prompt(question: str, signals: list[str], retrieved_context: list[str]) -> str:
    """Build a small grounded prompt for a future provider integration."""
    signals_section = "\n".join(f"- {signal}" for signal in signals) or "- No deterministic signals available."
    context_section = "\n".join(f"- {item}" for item in retrieved_context) or "- No retrieved context available."
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    
    return (
        f"Question:\n{question}\n\n"
        f"Deterministic signals:\n{signals_section}\n\n"
        f"Retrieved context:\n{context_section}\n\n"
        "Write a concise answer that stays within the evidence above."
    )


def build_no_claims_prompt() -> str:
    """Return a reminder prompt emphasizing grounded output."""
    return "Only include claims that are directly supported by retrieved context or deterministic tool outputs."
