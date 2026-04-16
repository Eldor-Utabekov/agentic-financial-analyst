"""Local-first document ingestion and chunking utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class Chunk(TypedDict):
    """Simple chunk representation for local retrieval."""

    chunk_id: str
    source: str
    text: str
    metadata: dict[str, object]


class LoadedDocument(TypedDict):
    """Loaded source document representation."""

    source: str
    text: str
    metadata: dict[str, object]


SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_text_documents(directory: str | Path) -> list[LoadedDocument]:
    """Load supported local text documents from a directory."""
    directory_path = _validate_directory(directory)

    documents: list[LoadedDocument] = []
    for file_path in sorted(directory_path.iterdir(), key=lambda path: path.name.lower()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        raw_text = file_path.read_text(encoding="utf-8")
        text = raw_text.strip()
        documents.append(
            LoadedDocument(
                source=str(file_path),
                text=text,
                metadata={
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                    "is_empty": text == "",
                },
            )
        )

    return documents


def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split a single document into simple overlapping text chunks."""
    validated_text = _validate_text(text, field_name="text")
    validated_source = _validate_text(source, field_name="source")
    validated_chunk_size = _validate_chunk_size(chunk_size)
    validated_chunk_overlap = _validate_chunk_overlap(chunk_overlap, validated_chunk_size)

    chunks: list[Chunk] = []
    step = validated_chunk_size - validated_chunk_overlap
    normalized_text = " ".join(validated_text.split())

    for chunk_index, start in enumerate(range(0, len(normalized_text), step)):
        chunk_text = normalized_text[start : start + validated_chunk_size].strip()
        if not chunk_text:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{validated_source}::chunk-{chunk_index}",
                source=validated_source,
                text=chunk_text,
                metadata={
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": start + len(chunk_text),
                },
            )
        )

    return chunks


def build_chunks_from_directory(
    directory: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Load supported documents from a directory and chunk them."""
    documents = load_text_documents(directory)

    chunks: list[Chunk] = []
    for document in documents:
        if not document["text"]:
            continue

        document_chunks = chunk_document(
            text=document["text"],
            source=document["source"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for chunk in document_chunks:
            chunk["metadata"] = {**document["metadata"], **chunk["metadata"]}
        chunks.extend(document_chunks)

    return chunks


def _validate_directory(directory: str | Path) -> Path:
    """Validate and normalize a directory input."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")
    return directory_path


def _validate_text(value: str, field_name: str) -> str:
    """Validate required text inputs."""
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized_value


def _validate_chunk_size(chunk_size: int) -> int:
    """Validate chunk size configuration."""
    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    return chunk_size


def _validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> int:
    """Validate chunk overlap configuration."""
    if not isinstance(chunk_overlap, int):
        raise TypeError("chunk_overlap must be an integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    return chunk_overlap
