from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.rag.ingest import build_chunks_from_directory
from app.rag.ingest import chunk_document
from app.rag.ingest import load_text_documents


class LoadTextDocumentsTests(unittest.TestCase):
    def test_load_text_documents_loads_supported_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            (temp_path / "alpha.txt").write_text("Alpha content", encoding="utf-8")
            (temp_path / "beta.md").write_text("Beta content", encoding="utf-8")
            (temp_path / "ignore.csv").write_text("ignored", encoding="utf-8")

            documents = load_text_documents(temp_path)

            self.assertEqual(len(documents), 2)
            self.assertEqual([Path(document["source"]).name for document in documents], ["alpha.txt", "beta.md"])
            self.assertEqual(documents[0]["metadata"]["extension"], ".txt")
            self.assertEqual(documents[1]["metadata"]["extension"], ".md")

    def test_load_text_documents_includes_empty_supported_files_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            (temp_path / "empty.txt").write_text("   \n", encoding="utf-8")

            documents = load_text_documents(temp_path)

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["text"], "")
            self.assertEqual(documents[0]["metadata"]["is_empty"], True)

    def test_load_text_documents_rejects_missing_directory(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_text_documents("missing-directory")

    def test_load_text_documents_rejects_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            file_path = Path(temp_directory) / "alpha.txt"
            file_path.write_text("Alpha content", encoding="utf-8")

            with self.assertRaises(NotADirectoryError):
                load_text_documents(file_path)


class ChunkDocumentTests(unittest.TestCase):
    def test_chunk_document_splits_text_with_overlap(self) -> None:
        chunks = chunk_document(
            text="abcdefghijklmnopqrstuvwxyz",
            source="sample.txt",
            chunk_size=10,
            chunk_overlap=2,
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["abcdefghij", "ijklmnopqr", "qrstuvwxyz", "yz"])
        self.assertEqual(chunks[0]["chunk_id"], "sample.txt::chunk-0")
        self.assertEqual(chunks[1]["metadata"]["start_char"], 8)

    def test_chunk_document_normalizes_whitespace(self) -> None:
        chunks = chunk_document(
            text="Alpha   beta\n\n gamma",
            source="sample.txt",
            chunk_size=50,
            chunk_overlap=0,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Alpha beta gamma")

    def test_chunk_document_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_overlap must be smaller than chunk_size"):
            chunk_document("Alpha beta", "sample.txt", chunk_size=10, chunk_overlap=10)

        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            chunk_document("   ", "sample.txt")


class BuildChunksFromDirectoryTests(unittest.TestCase):
    def test_build_chunks_from_directory_returns_chunks_for_supported_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            (temp_path / "alpha.txt").write_text("Alpha beta gamma delta", encoding="utf-8")
            (temp_path / "beta.md").write_text("Gamma delta epsilon zeta", encoding="utf-8")

            chunks = build_chunks_from_directory(temp_path, chunk_size=12, chunk_overlap=2)

            self.assertGreaterEqual(len(chunks), 2)
            self.assertTrue(all({"chunk_id", "source", "text", "metadata"} <= set(chunk.keys()) for chunk in chunks))
            self.assertEqual([chunk["metadata"]["chunk_index"] for chunk in chunks[:2]], [0, 1])

    def test_build_chunks_from_directory_preserves_document_metadata_and_skips_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            temp_path = Path(temp_directory)
            (temp_path / "alpha.txt").write_text("Alpha beta gamma delta", encoding="utf-8")
            (temp_path / "empty.md").write_text("   ", encoding="utf-8")

            chunks = build_chunks_from_directory(temp_path, chunk_size=12, chunk_overlap=2)

            self.assertTrue(all(Path(chunk["source"]).name != "empty.md" for chunk in chunks))
            self.assertTrue(all(chunk["metadata"]["file_name"] == "alpha.txt" for chunk in chunks))
            self.assertTrue(all(chunk["metadata"]["extension"] == ".txt" for chunk in chunks))
            self.assertTrue(all(chunk["metadata"]["is_empty"] is False for chunk in chunks))
