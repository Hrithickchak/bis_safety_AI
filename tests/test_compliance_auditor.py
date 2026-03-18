"""
Tests for the compliance_auditor module.

Covers: finding parsing, report building, summary generation, and edge cases.
LLM calls are mocked to avoid API costs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.compliance_auditor import ComplianceAuditor, run_audit
from src.models import (
    ComplianceFinding,
    ComplianceReport,
    ComplianceStatus,
    IssueCategory,
    IssueSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_LLM_RESPONSE = json.dumps([
    {
        "issue_id": "FORK-001",
        "status": "PASS",
        "severity": "INFO",
        "category": "Best Practice Gap",
        "requirement_summary": "Operator training required before operation",
        "regulatory_reference": ["OSHA 29 CFR 1910.178(l)(1)(ii)"],
        "document_quote": "All new forklift operators must complete a training program",
        "regulation_quote": "the employer shall ensure that each operator has successfully completed the training",
        "description": "The procedure requires training before operation, consistent with OSHA requirements.",
        "recommendation": ""
    },
    {
        "issue_id": "FORK-002",
        "status": "FAIL",
        "severity": "HIGH",
        "category": "Missing Section",
        "requirement_summary": "Overhead guard required for high-lift rider trucks",
        "regulatory_reference": ["OSHA 29 CFR 1910.178(f)(1)"],
        "document_quote": None,
        "regulation_quote": "High Lift Rider trucks shall be fitted with an overhead guard",
        "description": "The procedure does not mention overhead guards.",
        "recommendation": "Add a section on overhead guard requirements."
    },
    {
        "issue_id": "FORK-003",
        "status": "PARTIAL",
        "severity": "MEDIUM",
        "category": "Vague Language",
        "requirement_summary": "Certification must include specific details",
        "regulatory_reference": ["OSHA 29 CFR 1910.178(l)(6)"],
        "document_quote": "operators will receive a certification card",
        "regulation_quote": "certification shall include the name of the operator, the date of the training, the date of the evaluation, and the identity of the person(s) performing the training",
        "description": "Procedure mentions certification but lacks required detail fields.",
        "recommendation": "Specify that certification records must include operator name, training date, evaluation date, and trainer identity."
    },
])

MOCK_LLM_RESPONSE_WITH_FENCE = f"```json\n{MOCK_LLM_RESPONSE}\n```"

MOCK_SUMMARY_RESPONSE = (
    "This compliance audit reviewed the Forklift Safety Procedure against "
    "OSHA 29 CFR 1910.178. The procedure achieved a compliance score of 50%. "
    "One critical gap was identified regarding overhead guards. "
    "Immediate action is recommended."
)


@pytest.fixture
def auditor():
    return ComplianceAuditor(model="gemini-2.0-flash")


@pytest.fixture
def sample_clauses() -> list[dict]:
    return [
        {
            "text": "1910.178(l)(1) The employer shall ensure that each operator has successfully completed training.",
            "metadata": {"source": "OSHA 1910 178", "chunk_index": 0},
        },
        {
            "text": "1910.178(f)(1) High Lift Rider trucks shall be fitted with an overhead guard.",
            "metadata": {"source": "OSHA 1910 178", "chunk_index": 1},
        },
    ]


# ---------------------------------------------------------------------------
# Tests: _parse_findings
# ---------------------------------------------------------------------------

class TestParseFindings:
    def test_parses_valid_json(self, auditor: ComplianceAuditor):
        findings = auditor._parse_findings(MOCK_LLM_RESPONSE)
        assert len(findings) == 3
        assert all(isinstance(f, ComplianceFinding) for f in findings)

    def test_parses_fenced_json(self, auditor: ComplianceAuditor):
        findings = auditor._parse_findings(MOCK_LLM_RESPONSE_WITH_FENCE)
        assert len(findings) == 3

    def test_correct_statuses(self, auditor: ComplianceAuditor):
        findings = auditor._parse_findings(MOCK_LLM_RESPONSE)
        assert findings[0].status == ComplianceStatus.PASS
        assert findings[1].status == ComplianceStatus.FAIL
        assert findings[2].status == ComplianceStatus.PARTIAL

    def test_correct_severities(self, auditor: ComplianceAuditor):
        findings = auditor._parse_findings(MOCK_LLM_RESPONSE)
        assert findings[0].severity == IssueSeverity.INFO
        assert findings[1].severity == IssueSeverity.HIGH
        assert findings[2].severity == IssueSeverity.MEDIUM

    def test_correct_categories(self, auditor: ComplianceAuditor):
        findings = auditor._parse_findings(MOCK_LLM_RESPONSE)
        assert findings[1].category == IssueCategory.MISSING_SECTION
        assert findings[2].category == IssueCategory.VAGUE_LANGUAGE

    def test_handles_string_regulatory_reference(self, auditor: ComplianceAuditor):
        """regulatory_reference might come as a string instead of a list."""
        response = json.dumps([{
            "issue_id": "X-001",
            "status": "PASS",
            "severity": "LOW",
            "category": "Best Practice Gap",
            "requirement_summary": "Test",
            "regulatory_reference": "OSHA 29 CFR 1910.178(l)(4)",
            "document_quote": "quote",
            "regulation_quote": "reg quote",
            "description": "Test",
            "recommendation": "",
        }])
        findings = auditor._parse_findings(response)
        assert isinstance(findings[0].regulatory_reference, list)
        assert len(findings[0].regulatory_reference) == 1

    def test_handles_not_found_with_space(self, auditor: ComplianceAuditor):
        """LLM might return 'NOT FOUND' instead of 'NOT_FOUND'."""
        response = json.dumps([{
            "issue_id": "X-002",
            "status": "NOT FOUND",
            "severity": "MEDIUM",
            "category": "Missing Section",
            "requirement_summary": "Test requirement",
            "regulatory_reference": [],
            "description": "Not addressed",
            "recommendation": "Add it",
        }])
        findings = auditor._parse_findings(response)
        assert findings[0].status == ComplianceStatus.NOT_FOUND

    def test_raises_on_no_json(self, auditor: ComplianceAuditor):
        with pytest.raises(ValueError, match="Could not find JSON"):
            auditor._parse_findings("No JSON here.")

    def test_raises_on_invalid_json(self, auditor: ComplianceAuditor):
        with pytest.raises(ValueError, match="Invalid JSON"):
            auditor._parse_findings("[{invalid json}]")

    def test_skips_malformed_items(self, auditor: ComplianceAuditor):
        response = json.dumps([
            {"issue_id": "GOOD", "status": "PASS",
             "requirement_summary": "OK", "description": "OK"},
            {"status": "INVALID_STATUS"},  # Bad status -> skipped
        ])
        findings = auditor._parse_findings(response)
        # At least the good one should parse; the bad one may be skipped
        assert len(findings) >= 1


# ---------------------------------------------------------------------------
# Tests: audit (with mocked LLM)
# ---------------------------------------------------------------------------

class TestAudit:
    @patch.object(ComplianceAuditor, "_call_llm")
    def test_full_audit(
        self,
        mock_call_llm: MagicMock,
        auditor: ComplianceAuditor,
        sample_clauses: list[dict],
    ):
        # First call: audit findings, second call: summary
        mock_call_llm.side_effect = [
            MOCK_LLM_RESPONSE,
            MOCK_SUMMARY_RESPONSE,
        ]

        report = auditor.audit(
            procedure_title="Test Forklift Procedure",
            procedure_source="test.txt",
            procedure_text="All new forklift operators must complete a training program.",
            applicable_standards=["OSHA 29 CFR 1910.178"],
            regulatory_clauses=sample_clauses,
        )

        assert isinstance(report, ComplianceReport)
        assert len(report.findings) == 3
        assert report.passed == 1
        assert report.failed == 1
        assert report.partial == 1
        assert report.compliance_score > 0
        assert report.summary  # Summary should not be empty

    @patch.object(ComplianceAuditor, "_call_llm")
    def test_audit_with_empty_clauses(
        self, mock_call_llm: MagicMock, auditor: ComplianceAuditor
    ):
        mock_call_llm.side_effect = [
            json.dumps([]),  # No findings
            "No regulatory clauses provided.",  # Summary
        ]

        report = auditor.audit(
            procedure_title="Test",
            procedure_source="test.txt",
            procedure_text="Some procedure text.",
            applicable_standards=[],
            regulatory_clauses=[],
        )
        assert report.total_checks == 0
        assert report.compliance_score == 0.0


# ---------------------------------------------------------------------------
# Tests: summary generation
# ---------------------------------------------------------------------------

class TestSummaryGeneration:
    def test_template_summary_high_score(self):
        report = ComplianceReport(
            report_id="RPT-TEST",
            procedure_title="Test",
            procedure_source="test.txt",
            applicable_standards=["OSHA 29 CFR 1910.178"],
            findings=[
                ComplianceFinding(
                    issue_id="T-001",
                    status=ComplianceStatus.PASS,
                    requirement_summary="Req",
                    description="Desc",
                ),
            ],
        )
        report.compute_score()
        summary = ComplianceAuditor._generate_template_summary(report)
        assert "100.0%" in summary
        assert "largely compliant" in summary

    def test_template_summary_low_score(self):
        report = ComplianceReport(
            report_id="RPT-TEST",
            procedure_title="Test",
            procedure_source="test.txt",
            applicable_standards=["OSHA 29 CFR 1910.178"],
            findings=[
                ComplianceFinding(
                    issue_id="T-001",
                    status=ComplianceStatus.FAIL,
                    severity=IssueSeverity.CRITICAL,
                    requirement_summary="Req",
                    description="Desc",
                ),
            ],
        )
        report.compute_score()
        summary = ComplianceAuditor._generate_template_summary(report)
        assert "0.0%" in summary
        assert "significant compliance gaps" in summary
        assert "CRITICAL" in summary or "immediate attention" in summary

    @patch.object(ComplianceAuditor, "_call_llm")
    def test_llm_summary_fallback_on_error(
        self, mock_call_llm: MagicMock
    ):
        """If LLM summary call fails, should fall back to template."""
        auditor = ComplianceAuditor(model="gemini-2.0-flash")
        report = ComplianceReport(
            report_id="RPT-TEST",
            procedure_title="Test",
            procedure_source="test.txt",
            findings=[],
        )
        report.compute_score()

        # Make the LLM call raise an error
        mock_call_llm.side_effect = Exception("API error")
        summary = auditor._generate_summary(report)
        assert "Compliance Audit Report" in summary  # Template fallback


# ---------------------------------------------------------------------------
# Tests: run_audit convenience function
# ---------------------------------------------------------------------------

class TestRunAudit:
    @patch.object(ComplianceAuditor, "_call_llm")
    def test_run_audit(self, mock_call_llm: MagicMock):
        mock_call_llm.side_effect = [
            MOCK_LLM_RESPONSE,
            MOCK_SUMMARY_RESPONSE,
        ]

        report = run_audit(
            procedure_text="Some procedure text about forklift safety.",
            procedure_title="Test Procedure",
        )
        assert isinstance(report, ComplianceReport)
