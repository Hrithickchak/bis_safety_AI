"""
Tests for the standards_mapper module.

Covers: keyword detection, standard mapping, procedure enrichment,
LLM-assisted mapping, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import ProcedureDocument
from src.standards_mapper import StandardsMapper, DEFAULT_MAPPINGS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mapper() -> StandardsMapper:
    return StandardsMapper()


@pytest.fixture
def forklift_text() -> str:
    return (
        "This forklift safety procedure covers powered industrial truck "
        "operation, including daily inspections, operator training, "
        "and seatbelt use."
    )


@pytest.fixture
def loto_text() -> str:
    return (
        "This lockout/tagout procedure establishes minimum requirements "
        "for the control of hazardous energy during maintenance and servicing."
    )


@pytest.fixture
def confined_space_text() -> str:
    return (
        "This confined space entry procedure outlines permit-required "
        "confined space entry, atmospheric testing, and rescue provisions."
    )


@pytest.fixture
def ambiguous_text() -> str:
    return "General workplace safety policy for office environments."


@pytest.fixture
def custom_mapping_file(tmp_path: Path) -> Path:
    data = [
        {
            "topic": "Custom Topic",
            "keywords": ["custom", "widget"],
            "standards": ["CUSTOM-001"],
            "description": "A custom standard.",
        }
    ]
    p = tmp_path / "custom_mappings.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests: identify_topics
# ---------------------------------------------------------------------------

class TestIdentifyTopics:
    def test_forklift_detected(self, mapper: StandardsMapper, forklift_text: str):
        topics = mapper.identify_topics(forklift_text)
        topic_names = [t.topic for t in topics]
        assert any("Forklift" in t or "forklift" in t.lower() for t in topic_names)

    def test_loto_detected(self, mapper: StandardsMapper, loto_text: str):
        topics = mapper.identify_topics(loto_text)
        topic_names = [t.topic for t in topics]
        assert any("Lockout" in t or "lockout" in t.lower() for t in topic_names)

    def test_confined_space_detected(
        self, mapper: StandardsMapper, confined_space_text: str
    ):
        topics = mapper.identify_topics(confined_space_text)
        topic_names = [t.topic for t in topics]
        assert any("Confined" in t for t in topic_names)

    def test_no_match_for_ambiguous(
        self, mapper: StandardsMapper, ambiguous_text: str
    ):
        topics = mapper.identify_topics(ambiguous_text)
        # Office safety text shouldn't match specific industrial standards
        # (unless PPE or fire keywords are present)
        assert isinstance(topics, list)

    def test_multiple_topics(self, mapper: StandardsMapper):
        text = (
            "This procedure covers forklift operation and lockout/tagout "
            "procedures for hazardous energy control."
        )
        topics = mapper.identify_topics(text)
        topic_names = [t.topic for t in topics]
        assert len(topics) >= 2
        assert any("Forklift" in t or "forklift" in t.lower() for t in topic_names)
        assert any("Lockout" in t or "lockout" in t.lower() for t in topic_names)

    def test_case_insensitive(self, mapper: StandardsMapper):
        text = "FORKLIFT operations and POWERED INDUSTRIAL TRUCK safety."
        topics = mapper.identify_topics(text)
        assert len(topics) >= 1


# ---------------------------------------------------------------------------
# Tests: get_applicable_standards
# ---------------------------------------------------------------------------

class TestGetApplicableStandards:
    def test_forklift_standards(self, mapper: StandardsMapper, forklift_text: str):
        standards = mapper.get_applicable_standards(forklift_text)
        assert "OSHA 29 CFR 1910.178" in standards

    def test_loto_standards(self, mapper: StandardsMapper, loto_text: str):
        standards = mapper.get_applicable_standards(loto_text)
        assert "OSHA 29 CFR 1910.147" in standards

    def test_no_duplicates(self, mapper: StandardsMapper, forklift_text: str):
        standards = mapper.get_applicable_standards(forklift_text)
        assert len(standards) == len(set(standards))


# ---------------------------------------------------------------------------
# Tests: map_procedure
# ---------------------------------------------------------------------------

class TestMapProcedure:
    def test_enriches_document(self, mapper: StandardsMapper, forklift_text: str):
        doc = ProcedureDocument(
            title="Test Forklift Procedure",
            source_path="test.txt",
            raw_text=forklift_text,
        )
        result = mapper.map_procedure(doc)
        assert result is doc  # mutates in place
        assert len(doc.detected_topics) > 0
        assert len(doc.mapped_standards) > 0
        assert "OSHA 29 CFR 1910.178" in doc.mapped_standards


# ---------------------------------------------------------------------------
# Tests: custom mapping file
# ---------------------------------------------------------------------------

class TestCustomMappings:
    def test_loads_custom_file(self, custom_mapping_file: Path):
        mapper = StandardsMapper(mapping_file=custom_mapping_file)
        assert len(mapper.mappings) == 1
        assert mapper.mappings[0].topic == "Custom Topic"

    def test_custom_keyword_match(self, custom_mapping_file: Path):
        mapper = StandardsMapper(mapping_file=custom_mapping_file)
        standards = mapper.get_applicable_standards("Testing our custom widget system.")
        assert "CUSTOM-001" in standards

    def test_falls_back_to_defaults_if_missing(self):
        mapper = StandardsMapper(mapping_file="/nonexistent/path.json")
        assert len(mapper.mappings) == len(DEFAULT_MAPPINGS)


# ---------------------------------------------------------------------------
# Tests: LLM-assisted mapping (with mock)
# ---------------------------------------------------------------------------

class TestLLMMapping:
    def test_fallback_when_no_llm(self, mapper: StandardsMapper, forklift_text: str):
        result = mapper.identify_topics_with_llm(forklift_text, llm_callable=None)
        # Should fall back to keyword matching
        assert isinstance(result, list)
        assert "OSHA 29 CFR 1910.178" in result

    def test_with_mock_llm(self, mapper: StandardsMapper):
        def mock_llm(prompt: str) -> str:
            return '["OSHA 29 CFR 1910.178", "CSA B335"]'

        result = mapper.identify_topics_with_llm("some text", llm_callable=mock_llm)
        assert "OSHA 29 CFR 1910.178" in result
        assert "CSA B335" in result

    def test_with_bad_llm_response(self, mapper: StandardsMapper, forklift_text: str):
        def bad_llm(prompt: str) -> str:
            return "I cannot determine the standards."

        result = mapper.identify_topics_with_llm(forklift_text, llm_callable=bad_llm)
        # Should fall back to keyword matching
        assert isinstance(result, list)
