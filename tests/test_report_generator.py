"""
Tests for the report_generator module.

Covers: JSON report, HTML report, console report, and edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import (
    ComplianceFinding,
    ComplianceReport,
    ComplianceStatus,
    IssueCategory,
    IssueSeverity,
)
from src.report_generator import (
    save_json_report,
    save_html_report,
    print_console_report,
    _build_finding_html,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_report() -> ComplianceReport:
    findings = [
        ComplianceFinding(
            issue_id="FORK-001",
            status=ComplianceStatus.PASS,
            severity=IssueSeverity.INFO,
            category=IssueCategory.BEST_PRACTICE_GAP,
            requirement_summary="Operator training required",
            regulatory_reference=["OSHA 29 CFR 1910.178(l)(1)(ii)"],
            document_quote="All operators must complete training",
            regulation_quote="employer shall ensure training",
            description="Procedure requires training, consistent with OSHA.",
            recommendation="",
        ),
        ComplianceFinding(
            issue_id="FORK-002",
            status=ComplianceStatus.FAIL,
            severity=IssueSeverity.HIGH,
            category=IssueCategory.MISSING_SECTION,
            requirement_summary="Overhead guard required",
            regulatory_reference=["OSHA 29 CFR 1910.178(f)(1)"],
            document_quote=None,
            regulation_quote="trucks shall be fitted with an overhead guard",
            description="Procedure does not mention overhead guards.",
            recommendation="Add overhead guard requirements.",
        ),
        ComplianceFinding(
            issue_id="FORK-003",
            status=ComplianceStatus.PARTIAL,
            severity=IssueSeverity.MEDIUM,
            category=IssueCategory.VAGUE_LANGUAGE,
            requirement_summary="Certification details",
            regulatory_reference=["OSHA 29 CFR 1910.178(l)(6)"],
            document_quote="certification card valid for 3 years",
            regulation_quote=None,
            description="Certification mentioned but missing detail fields.",
            recommendation="Include operator name, dates, trainer identity.",
        ),
        ComplianceFinding(
            issue_id="FORK-004",
            status=ComplianceStatus.NOT_FOUND,
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.MISSING_SECTION,
            requirement_summary="Elevator capacity check",
            regulatory_reference=["OSHA 29 CFR 1910.178(n)(15)"],
            document_quote=None,
            regulation_quote="driver must be informed of elevator capacity",
            description="No mention of elevator use.",
            recommendation="Add elevator capacity verification.",
        ),
    ]
    report = ComplianceReport(
        report_id="RPT-TEST001",
        procedure_title="Forklift Safety Procedure",
        procedure_source="data/procedures/forklift_safety_procedure.txt",
        applicable_standards=[
            "OSHA 29 CFR 1910.178",
            "ANSI/ITSDF B56.1",
            "CSA B335",
        ],
        findings=findings,
        summary="Test executive summary for the forklift procedure.",
    )
    report.compute_score()
    return report


@pytest.fixture
def empty_report() -> ComplianceReport:
    report = ComplianceReport(
        report_id="RPT-EMPTY",
        procedure_title="Empty Procedure",
        procedure_source="empty.txt",
        findings=[],
        summary="No findings.",
    )
    report.compute_score()
    return report


# ---------------------------------------------------------------------------
# Tests: JSON report
# ---------------------------------------------------------------------------

class TestJsonReport:
    def test_creates_file(self, sample_report: ComplianceReport, tmp_path: Path):
        out = tmp_path / "report.json"
        result = save_json_report(sample_report, out)
        assert result == out
        assert out.exists()

    def test_valid_json(self, sample_report: ComplianceReport, tmp_path: Path):
        out = tmp_path / "report.json"
        save_json_report(sample_report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["report_id"] == "RPT-TEST001"
        assert len(data["findings"]) == 4

    def test_creates_parent_dirs(
        self, sample_report: ComplianceReport, tmp_path: Path
    ):
        out = tmp_path / "sub" / "dir" / "report.json"
        result = save_json_report(sample_report, out)
        assert result.exists()

    def test_empty_report(self, empty_report: ComplianceReport, tmp_path: Path):
        out = tmp_path / "empty.json"
        save_json_report(empty_report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["findings"] == []
        assert data["compliance_score"] == 0.0


# ---------------------------------------------------------------------------
# Tests: HTML report
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_creates_file(self, sample_report: ComplianceReport, tmp_path: Path):
        out = tmp_path / "report.html"
        result = save_html_report(sample_report, out)
        assert result == out
        assert out.exists()

    def test_contains_procedure_title(
        self, sample_report: ComplianceReport, tmp_path: Path
    ):
        out = tmp_path / "report.html"
        save_html_report(sample_report, out)
        html = out.read_text(encoding="utf-8")
        assert "Forklift Safety Procedure" in html

    def test_contains_findings(
        self, sample_report: ComplianceReport, tmp_path: Path
    ):
        out = tmp_path / "report.html"
        save_html_report(sample_report, out)
        html = out.read_text(encoding="utf-8")
        assert "FORK-001" in html
        assert "FORK-002" in html
        assert "FORK-003" in html
        assert "FORK-004" in html

    def test_contains_score(
        self, sample_report: ComplianceReport, tmp_path: Path
    ):
        out = tmp_path / "report.html"
        save_html_report(sample_report, out)
        html = out.read_text(encoding="utf-8")
        assert f"{sample_report.compliance_score}%" in html

    def test_status_css_classes(
        self, sample_report: ComplianceReport, tmp_path: Path
    ):
        out = tmp_path / "report.html"
        save_html_report(sample_report, out)
        html = out.read_text(encoding="utf-8")
        assert "status-pass" in html
        assert "status-fail" in html
        assert "status-partial" in html
        assert "status-not-found" in html

    def test_empty_report(self, empty_report: ComplianceReport, tmp_path: Path):
        out = tmp_path / "empty.html"
        save_html_report(empty_report, out)
        html = out.read_text(encoding="utf-8")
        assert "0%" in html


# ---------------------------------------------------------------------------
# Tests: _build_finding_html
# ---------------------------------------------------------------------------

class TestBuildFindingHtml:
    def test_includes_document_quote(self, sample_report: ComplianceReport):
        finding = sample_report.findings[0]
        html = _build_finding_html(finding)
        assert "Procedure states:" in html
        assert finding.document_quote in html

    def test_includes_regulation_quote(self, sample_report: ComplianceReport):
        finding = sample_report.findings[1]
        html = _build_finding_html(finding)
        assert "Regulation states:" in html
        assert finding.regulation_quote in html

    def test_includes_recommendation(self, sample_report: ComplianceReport):
        finding = sample_report.findings[1]
        html = _build_finding_html(finding)
        assert "Recommendation:" in html

    def test_no_quote_when_null(self, sample_report: ComplianceReport):
        finding = sample_report.findings[3]  # NOT_FOUND, no document_quote
        html = _build_finding_html(finding)
        assert "Procedure states:" not in html


# ---------------------------------------------------------------------------
# Tests: console report
# ---------------------------------------------------------------------------

class TestConsoleReport:
    def test_prints_without_error(
        self, sample_report: ComplianceReport, capsys
    ):
        """Just verify it doesn't raise."""
        print_console_report(sample_report)
        # No assertion on output shape — just checking no exceptions

    def test_empty_report_prints(
        self, empty_report: ComplianceReport, capsys
    ):
        print_console_report(empty_report)
