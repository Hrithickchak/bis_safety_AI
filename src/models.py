
from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ComplianceStatus(str, Enum):
    """Result of checking a single regulatory requirement."""
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"


class IssueSeverity(str, Enum):
    """Severity of a compliance finding."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class IssueCategory(str, Enum):
    """Category of a compliance finding."""
    MISSING_SECTION = "Missing Section"
    OUTDATED_PRACTICE = "Outdated Practice"
    INCORRECT_VALUE = "Incorrect Value"
    VAGUE_LANGUAGE = "Vague Language"
    CONTRADICTION = "Contradiction"
    INCOMPLETE_REFERENCE = "Incomplete Reference"
    BEST_PRACTICE_GAP = "Best Practice Gap"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class RegulatoryClause(BaseModel):
    """A single clause or requirement extracted from a regulation."""
    standard_id: str = Field(
        ..., description="Identifier of the standard, e.g. 'OSHA 1910.178'"
    )
    section: str = Field(
        ..., description="Section or paragraph reference, e.g. '(l)(4)'"
    )
    title: str = Field(
        default="", description="Short title of the clause"
    )
    text: str = Field(
        ..., description="Full text of the regulatory clause"
    )
    source_url: Optional[str] = Field(
        default=None, description="URL or file path where the clause was sourced"
    )


class StandardMapping(BaseModel):
    """Maps a safety topic to one or more regulatory standards."""
    topic: str = Field(
        ..., description="Safety topic, e.g. 'Forklift Operation'"
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords that trigger this mapping"
    )
    standards: list[str] = Field(
        ..., description="List of standard IDs, e.g. ['OSHA 1910.178', 'CSA B335']"
    )
    description: str = Field(
        default="", description="Human-readable description of the mapping"
    )


class ProcedureSection(BaseModel):
    """A parsed section of a safety procedure document."""
    heading: str = Field(
        default="", description="Section heading"
    )
    text: str = Field(
        ..., description="Full text of the section"
    )
    page_number: Optional[int] = Field(
        default=None, description="Page number in original document"
    )
    line_range: Optional[str] = Field(
        default=None, description="Line range in the source, e.g. 'L45-L60'"
    )


class ProcedureDocument(BaseModel):
    """A parsed safety procedure document."""
    title: str = Field(..., description="Title of the procedure document")
    source_path: str = Field(..., description="File path or URL of the source")
    raw_text: str = Field(..., description="Full raw text of the document")
    sections: list[ProcedureSection] = Field(
        default_factory=list,
        description="Parsed sections of the document"
    )
    detected_topics: list[str] = Field(
        default_factory=list,
        description="Detected safety topics"
    )
    mapped_standards: list[str] = Field(
        default_factory=list,
        description="Standards identified as applicable"
    )


# ---------------------------------------------------------------------------
# Compliance findings
# ---------------------------------------------------------------------------

class ComplianceFinding(BaseModel):
    """A single compliance finding from the audit."""
    issue_id: str = Field(
        ..., description="Unique identifier for this finding"
    )
    status: ComplianceStatus = Field(
        ..., description="PASS, FAIL, PARTIAL, or NOT_FOUND"
    )
    severity: IssueSeverity = Field(
        default=IssueSeverity.MEDIUM,
        description="Severity level of the finding"
    )
    category: IssueCategory = Field(
        default=IssueCategory.MISSING_SECTION,
        description="Category of the finding"
    )
    requirement_summary: str = Field(
        ..., description="Short summary of the regulatory requirement being checked"
    )
    regulatory_reference: list[str] = Field(
        default_factory=list,
        description="Regulatory references, e.g. ['OSHA 1910.178(l)(4)']"
    )
    document_quote: Optional[str] = Field(
        default=None,
        description="Exact quote from the procedure document relevant to this finding"
    )
    regulation_quote: Optional[str] = Field(
        default=None,
        description="Exact quote from the regulation text"
    )
    description: str = Field(
        ..., description="Detailed description of the compliance issue"
    )
    recommendation: str = Field(
        default="",
        description="Suggested corrective action"
    )


class ComplianceReport(BaseModel):
    """Full structured compliance report."""
    report_id: str = Field(
        ..., description="Unique report identifier"
    )
    generated_at: datetime.datetime = Field(
        default_factory=datetime.datetime.now,
        description="Timestamp when the report was generated"
    )
    procedure_title: str = Field(
        ..., description="Title of the audited procedure document"
    )
    procedure_source: str = Field(
        ..., description="Source path or URL of the procedure"
    )
    applicable_standards: list[str] = Field(
        default_factory=list,
        description="List of standards the procedure was checked against"
    )
    total_checks: int = Field(
        default=0, description="Total number of compliance checks performed"
    )
    passed: int = Field(default=0, description="Number of checks that passed")
    failed: int = Field(default=0, description="Number of checks that failed")
    partial: int = Field(
        default=0, description="Number of checks partially met"
    )
    not_found: int = Field(
        default=0, description="Number of requirements not addressed in the procedure"
    )
    compliance_score: float = Field(
        default=0.0,
        description="Overall compliance score (0-100%)"
    )
    findings: list[ComplianceFinding] = Field(
        default_factory=list,
        description="List of all compliance findings"
    )
    summary: str = Field(
        default="",
        description="Executive summary of the audit results"
    )

    def compute_score(self) -> None:
        """Compute the compliance score from findings."""
        self.total_checks = len(self.findings)
        self.passed = sum(
            1 for f in self.findings if f.status == ComplianceStatus.PASS
        )
        self.failed = sum(
            1 for f in self.findings if f.status == ComplianceStatus.FAIL
        )
        self.partial = sum(
            1 for f in self.findings if f.status == ComplianceStatus.PARTIAL
        )
        self.not_found = sum(
            1 for f in self.findings if f.status == ComplianceStatus.NOT_FOUND
        )
        if self.total_checks > 0:
            # PASS = 1.0, PARTIAL = 0.5, FAIL/NOT_FOUND = 0.0
            score = (self.passed + 0.5 * self.partial) / self.total_checks
            self.compliance_score = round(score * 100, 1)
        else:
            self.compliance_score = 0.0
