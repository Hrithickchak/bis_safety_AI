"""
Tests for the Pydantic data models.

Covers: enum values, model construction, score computation, serialisation.
"""

from __future__ import annotations

import datetime

import pytest

from src.models import (
    ComplianceFinding,
    ComplianceReport,
    ComplianceStatus,
    IssueCategory,
    IssueSeverity,
    ProcedureDocument,
    ProcedureSection,
    RegulatoryClause,
    StandardMapping,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_compliance_status_values(self):
        assert ComplianceStatus.PASS.value == "PASS"
        assert ComplianceStatus.FAIL.value == "FAIL"
        assert ComplianceStatus.PARTIAL.value == "PARTIAL"
        assert ComplianceStatus.NOT_FOUND.value == "NOT_FOUND"

    def test_severity_values(self):
        assert IssueSeverity.CRITICAL.value == "CRITICAL"
        assert IssueSeverity.HIGH.value == "HIGH"
        assert IssueSeverity.INFO.value == "INFO"

    def test_issue_category_values(self):
        assert IssueCategory.MISSING_SECTION.value == "Missing Section"
        assert IssueCategory.VAGUE_LANGUAGE.value == "Vague Language"
        assert IssueCategory.CONTRADICTION.value == "Contradiction"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

class TestRegulatoryClause:
    def test_construction(self):
        clause = RegulatoryClause(
            standard_id="OSHA 1910.178",
            section="(l)(4)",
            title="Operator Evaluation",
            text="An evaluation shall be conducted at least once every three years.",
        )
        assert clause.standard_id == "OSHA 1910.178"
        assert clause.source_url is None

    def test_required_fields(self):
        with pytest.raises(Exception):
            RegulatoryClause()  # type: ignore[call-arg]


class TestStandardMapping:
    def test_construction(self):
        m = StandardMapping(
            topic="Forklift",
            keywords=["forklift", "pit"],
            standards=["OSHA 1910.178"],
        )
        assert len(m.keywords) == 2

    def test_defaults(self):
        m = StandardMapping(
            topic="Test",
            standards=["STD-1"],
        )
        assert m.keywords == []
        assert m.description == ""


class TestProcedureSection:
    def test_construction(self):
        s = ProcedureSection(text="Some content")
        assert s.heading == ""
        assert s.page_number is None


class TestProcedureDocument:
    def test_construction(self):
        doc = ProcedureDocument(
            title="Test Procedure",
            source_path="test.txt",
            raw_text="Full text here.",
        )
        assert doc.detected_topics == []
        assert doc.mapped_standards == []
        assert doc.sections == []


# ---------------------------------------------------------------------------
# ComplianceFinding
# ---------------------------------------------------------------------------

class TestComplianceFinding:
    def test_full_construction(self):
        f = ComplianceFinding(
            issue_id="FORK-001",
            status=ComplianceStatus.FAIL,
            severity=IssueSeverity.HIGH,
            category=IssueCategory.MISSING_SECTION,
            requirement_summary="Pre-shift inspection required",
            regulatory_reference=["OSHA 29 CFR 1910.178(q)(6)"],
            document_quote=None,
            regulation_quote="trucks shall be examined at least daily",
            description="The procedure does not mention daily inspections.",
            recommendation="Add daily inspection requirement.",
        )
        assert f.status == ComplianceStatus.FAIL
        assert len(f.regulatory_reference) == 1

    def test_defaults(self):
        f = ComplianceFinding(
            issue_id="X-001",
            status=ComplianceStatus.PASS,
            requirement_summary="Test",
            description="Test description",
        )
        assert f.severity == IssueSeverity.MEDIUM
        assert f.category == IssueCategory.MISSING_SECTION
        assert f.document_quote is None
        assert f.recommendation == ""


# ---------------------------------------------------------------------------
# ComplianceReport + score computation
# ---------------------------------------------------------------------------

class TestComplianceReport:
    def _make_report(self, statuses: list[ComplianceStatus]) -> ComplianceReport:
        findings = [
            ComplianceFinding(
                issue_id=f"T-{i:03d}",
                status=s,
                requirement_summary=f"Requirement {i}",
                description=f"Description {i}",
            )
            for i, s in enumerate(statuses, 1)
        ]
        report = ComplianceReport(
            report_id="RPT-TEST",
            procedure_title="Test Procedure",
            procedure_source="test.txt",
            findings=findings,
        )
        report.compute_score()
        return report

    def test_all_pass(self):
        report = self._make_report([ComplianceStatus.PASS] * 5)
        assert report.passed == 5
        assert report.failed == 0
        assert report.compliance_score == 100.0

    def test_all_fail(self):
        report = self._make_report([ComplianceStatus.FAIL] * 4)
        assert report.failed == 4
        assert report.compliance_score == 0.0

    def test_mixed_results(self):
        report = self._make_report([
            ComplianceStatus.PASS,
            ComplianceStatus.PASS,
            ComplianceStatus.FAIL,
            ComplianceStatus.PARTIAL,
        ])
        assert report.passed == 2
        assert report.failed == 1
        assert report.partial == 1
        # score = (2 + 0.5*1) / 4 = 2.5/4 = 62.5%
        assert report.compliance_score == 62.5

    def test_empty_findings(self):
        report = self._make_report([])
        assert report.total_checks == 0
        assert report.compliance_score == 0.0

    def test_not_found_counts(self):
        report = self._make_report([
            ComplianceStatus.PASS,
            ComplianceStatus.NOT_FOUND,
            ComplianceStatus.NOT_FOUND,
        ])
        assert report.not_found == 2
        # score = (1 + 0*2) / 3 = 33.3%
        assert report.compliance_score == pytest.approx(33.3, abs=0.1)

    def test_serialisation_roundtrip(self):
        report = self._make_report([ComplianceStatus.PASS, ComplianceStatus.FAIL])
        data = report.model_dump(mode="json")
        assert data["report_id"] == "RPT-TEST"
        assert len(data["findings"]) == 2
        # Reconstruct
        reconstructed = ComplianceReport(**data)
        assert reconstructed.report_id == report.report_id

    def test_generated_at(self):
        report = self._make_report([ComplianceStatus.PASS])
        assert isinstance(report.generated_at, datetime.datetime)
