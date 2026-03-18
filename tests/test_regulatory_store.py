"""
Tests for the regulatory_store module.

Covers: text splitting, ingestion, persistence, and retrieval.
Note: These tests mock the Gemini embedding calls to avoid API costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.regulatory_store import RegulatoryStore, _split_text


# ---------------------------------------------------------------------------
# Tests: _split_text
# ---------------------------------------------------------------------------

class TestSplitText:
    def test_basic_splitting(self):
        text = "\n".join([f"Line {i}: " + "x" * 50 for i in range(30)])
        chunks = _split_text(text, chunk_size=200, chunk_overlap=50, source="test")
        assert len(chunks) > 1
        for c in chunks:
            assert "text" in c
            assert "metadata" in c
            assert c["metadata"]["source"] == "test"

    def test_short_text_single_chunk(self):
        text = "This is a short regulation text."
        chunks = _split_text(text, chunk_size=800, chunk_overlap=200, source="reg")
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_empty_text(self):
        chunks = _split_text("", chunk_size=800, chunk_overlap=200, source="empty")
        # Single chunk with empty text
        assert len(chunks) == 1

    def test_overlap_present(self):
        # Create text that will definitely span multiple chunks
        lines = [f"Line {i}: regulation text about safety " * 3 for i in range(50)]
        text = "\n".join(lines)
        chunks = _split_text(text, chunk_size=300, chunk_overlap=100, source="test")
        # Check that consecutive chunks share some overlap text
        if len(chunks) >= 2:
            chunk0_lines = set(chunks[0]["text"].split("\n"))
            chunk1_lines = set(chunks[1]["text"].split("\n"))
            overlap = chunk0_lines & chunk1_lines
            assert len(overlap) > 0, "Expected overlap between consecutive chunks"

    def test_metadata_has_chunk_index(self):
        text = "\n".join(["Line " + "x" * 100 for _ in range(20)])
        chunks = _split_text(text, chunk_size=200, chunk_overlap=50, source="s")
        for i, c in enumerate(chunks):
            assert c["metadata"]["chunk_index"] == i

    def test_metadata_has_start_line(self):
        text = "Line A\nLine B\nLine C\nLine D\nLine E"
        chunks = _split_text(text, chunk_size=100, chunk_overlap=10, source="s")
        for c in chunks:
            assert "start_line" in c["metadata"]


# ---------------------------------------------------------------------------
# Tests: RegulatoryStore (with mocked embeddings)
# ---------------------------------------------------------------------------

class TestRegulatoryStore:
    """
    Tests that exercise RegulatoryStore logic without making real API calls.
    Embedding calls are mocked with a simple deterministic function.
    """

    @pytest.fixture
    def store_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "vectorstore"
        d.mkdir()
        return d

    @pytest.fixture
    def sample_regulation(self, tmp_path: Path) -> Path:
        text = (
            "1910.178(l)(4) An evaluation of each powered industrial truck "
            "operator's performance shall be conducted at least once every "
            "three years.\n\n"
            "1910.178(q)(6) Industrial trucks shall be examined before being "
            "placed in service, and shall not be placed in service if the "
            "examination shows any condition adversely affecting the safety "
            "of the vehicle.\n"
        )
        p = tmp_path / "osha_1910_178.txt"
        p.write_text(text, encoding="utf-8")
        return p

    @patch("src.regulatory_store.RegulatoryStore._get_embeddings")
    def test_ingest_text(self, mock_embeddings, store_dir: Path):
        """Ingest raw text and check document count."""
        mock_emb = MagicMock()
        # Return fixed-dimension embeddings
        mock_emb.embed_documents.return_value = [[0.1] * 8]
        mock_emb.embed_query.return_value = [0.1] * 8
        mock_embeddings.return_value = mock_emb

        store = RegulatoryStore(persist_dir=store_dir)
        n = store.ingest_text("Test regulation text.", source_label="TEST")
        assert n >= 1
        assert store.document_count >= 1

    @patch("src.regulatory_store.RegulatoryStore._get_embeddings")
    def test_ingest_file(
        self, mock_embeddings, store_dir: Path, sample_regulation: Path
    ):
        """Ingest a file and check chunks are created."""
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 8] * 10
        mock_emb.embed_query.return_value = [0.1] * 8
        mock_embeddings.return_value = mock_emb

        store = RegulatoryStore(persist_dir=store_dir)
        n = store.ingest_file(sample_regulation)
        assert n >= 1

    @patch("src.regulatory_store.RegulatoryStore._get_embeddings")
    def test_ingest_directory(
        self, mock_embeddings, store_dir: Path, sample_regulation: Path
    ):
        """Ingest all files from a directory."""
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1] * 8] * 10
        mock_emb.embed_query.return_value = [0.1] * 8
        mock_embeddings.return_value = mock_emb

        store = RegulatoryStore(persist_dir=store_dir)
        n = store.ingest_directory(sample_regulation.parent)
        assert n >= 1

    def test_document_count_initially_zero(self, store_dir: Path):
        store = RegulatoryStore(persist_dir=store_dir)
        assert store.document_count == 0

    def test_load_empty_store(self, store_dir: Path):
        store = RegulatoryStore(persist_dir=store_dir)
        loaded = store.load()
        assert loaded is False  # No index file exists

    def test_search_empty_store(self, store_dir: Path):
        store = RegulatoryStore(persist_dir=store_dir)
        results = store.search("forklift training")
        assert results == []
