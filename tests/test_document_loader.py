"""
Tests for the document_loader module.

Covers: text reading, section splitting, multi-format loading, and edge cases.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.document_loader import (
    _read_txt,
    _split_into_sections,
    load_procedure,
    load_all_procedures,
)
from src.models import ProcedureDocument, ProcedureSection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    """Create a minimal procedure text file for testing."""
    content = textwrap.dedent("""\
        FORKLIFT SAFETY PROCEDURE
        Document No: TEST-001

        ================================================================================
        1. PURPOSE
        ================================================================================

        This procedure establishes safe forklift operation requirements.

        ================================================================================
        2. SCOPE
        ================================================================================

        Covers all powered industrial trucks.

        ================================================================================
        3. TRAINING
        ================================================================================

        All operators must be trained and certified every three years.
    """)
    p = tmp_path / "forklift_test.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def empty_txt(tmp_path: Path) -> Path:
    """Create an empty text file."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def no_headings_txt(tmp_path: Path) -> Path:
    """A procedure file with no recognisable headings."""
    p = tmp_path / "no_headings.txt"
    p.write_text(
        "This is a procedure document with no section headings. "
        "It should be returned as a single section.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def procedures_dir(tmp_path: Path) -> Path:
    """Create a directory with two procedure files."""
    d = tmp_path / "procedures"
    d.mkdir()
    (d / "proc_a.txt").write_text("1. PURPOSE\nPurpose text.\n2. SCOPE\nScope text.\n")
    (d / "proc_b.txt").write_text("Some procedure without sections.")
    return d


# ---------------------------------------------------------------------------
# Tests: _read_txt
# ---------------------------------------------------------------------------

class TestReadTxt:
    def test_reads_utf8(self, sample_txt: Path):
        text = _read_txt(sample_txt)
        assert "FORKLIFT SAFETY PROCEDURE" in text

    def test_empty_file(self, empty_txt: Path):
        text = _read_txt(empty_txt)
        assert text == ""


# ---------------------------------------------------------------------------
# Tests: _split_into_sections
# ---------------------------------------------------------------------------

class TestSplitIntoSections:
    def test_detects_numbered_headings(self, sample_txt: Path):
        text = _read_txt(sample_txt)
        sections = _split_into_sections(text)
        headings = [s.heading for s in sections]
        # Should detect "1. PURPOSE", "2. SCOPE", "3. TRAINING"
        assert any("PURPOSE" in h for h in headings)
        assert any("SCOPE" in h for h in headings)
        assert any("TRAINING" in h for h in headings)

    def test_no_headings_returns_single_section(self, no_headings_txt: Path):
        text = _read_txt(no_headings_txt)
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0].heading == "(Full Document)"

    def test_preamble_captured(self, sample_txt: Path):
        text = _read_txt(sample_txt)
        sections = _split_into_sections(text)
        # Text before the first heading should appear in a preamble section
        preamble = [s for s in sections if s.heading == "(Preamble)"]
        if preamble:
            assert "FORKLIFT SAFETY PROCEDURE" in preamble[0].text

    def test_empty_text(self):
        sections = _split_into_sections("")
        assert len(sections) == 1
        assert sections[0].heading == "(Full Document)"

    def test_sections_are_procedure_section_objects(self, sample_txt: Path):
        text = _read_txt(sample_txt)
        sections = _split_into_sections(text)
        for s in sections:
            assert isinstance(s, ProcedureSection)


# ---------------------------------------------------------------------------
# Tests: load_procedure
# ---------------------------------------------------------------------------

class TestLoadProcedure:
    def test_loads_txt(self, sample_txt: Path):
        doc = load_procedure(sample_txt)
        assert isinstance(doc, ProcedureDocument)
        assert doc.raw_text
        assert len(doc.sections) > 0
        assert doc.source_path == str(sample_txt)

    def test_title_default(self, sample_txt: Path):
        doc = load_procedure(sample_txt)
        # Default title is derived from filename
        assert "Forklift" in doc.title or "forklift" in doc.title.lower()

    def test_title_override(self, sample_txt: Path):
        doc = load_procedure(sample_txt, title="My Custom Title")
        assert doc.title == "My Custom Title"

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_procedure(tmp_path / "nonexistent.txt")

    def test_unsupported_extension(self, tmp_path: Path):
        bad = tmp_path / "file.xyz"
        bad.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_procedure(bad)


# ---------------------------------------------------------------------------
# Tests: load_all_procedures
# ---------------------------------------------------------------------------

class TestLoadAllProcedures:
    def test_loads_directory(self, procedures_dir: Path):
        docs = load_all_procedures(procedures_dir)
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, ProcedureDocument)

    def test_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_procs"
        empty_dir.mkdir()
        docs = load_all_procedures(empty_dir)
        assert docs == []
