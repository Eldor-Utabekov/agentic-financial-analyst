"""Deterministic baseline lexical retriever for local chunks."""

from __future__ import annotations

import re
from typing import TypedDict

from app.rag.ingest import Chunk


TOKEN_PATTERN = re.compile(r"\b\w+\b")


class ScoredChunk(Chunk):
    """Retrieved chunk with an attached lexical relevance score."""

    score: float


def score_chunk(query: str, chunk_text: str) -> float:
    """Score a chunk using simple lexical overlap with the query."""
    validated_query = _validate_text(query, field_name="query")
    validated_chunk_text = _validate_text(chunk_text, field_name="chunk_text")

    query_tokens = set(_tokenize(validated_query))
    chunk_tokens = _tokenize(validated_chunk_text)
    if not query_tokens or not chunk_tokens:
        return 0.0

    overlap_count = sum(1 for token in chunk_tokens if token in query_tokens)
    return overlap_count / len(query_tokens)


def retrieve_top_k(query: str, chunks: list[Chunk], k: int = 3) -> list[ScoredChunk]:
    """Retrieve the top-k chunks ranked by lexical overlap score."""
    validated_query = _validate_text(query, field_name="query")
    validated_chunks = _validate_chunks(chunks)
    validated_k = _validate_k(k)

    scored_chunks: list[ScoredChunk] = []
    for chunk in validated_chunks:
        score = score_chunk(validated_query, chunk["text"])
        if score <= 0.0:
            continue

        scored_chunk: ScoredChunk = {
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
            "metadata": dict(chunk["metadata"]),
            "score": score,
        }
        scored_chunks.append(scored_chunk)

    ranked_chunks = sorted(
        scored_chunks,
        key=lambda chunk: (-chunk["score"], chunk["chunk_id"]),
    )

    return ranked_chunks[:validated_k]


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase lexical tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def _validate_text(value: str, field_name: str) -> str:
    """Validate required text inputs."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_value


def _validate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Validate the input chunk list used for retrieval."""
    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list.")

    required_keys = {"chunk_id", "source", "text", "metadata"}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise TypeError("Each chunk must be a dictionary.")
        missing_keys = required_keys - set(chunk.keys())
        if missing_keys:
            missing_keys_str = ", ".join(sorted(missing_keys))
            raise ValueError(f"Each chunk must include keys: {missing_keys_str}.")
        if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"].strip():
            raise ValueError("chunk_id must be a non-empty string.")
        if not isinstance(chunk["source"], str) or not chunk["source"].strip():
            raise ValueError("source must be a non-empty string.")
        if not isinstance(chunk["text"], str) or not chunk["text"].strip():
            raise ValueError("text must be a non-empty string.")
        if not isinstance(chunk["metadata"], dict):
            raise TypeError("metadata must be a dictionary.")

    return chunks


def _validate_k(k: int) -> int:
    """Validate top-k configuration."""
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    return k
