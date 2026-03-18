"""
Report Generator – produce structured compliance reports in JSON and HTML.

Takes a ComplianceReport object and renders it into:
- JSON (machine-readable, for dashboards / downstream tools)
- HTML (human-readable, styled report for safety managers)
- Console (rich terminal output for CLI)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import ComplianceReport, ComplianceStatus, IssueSeverity


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def save_json_report(
    report: ComplianceReport,
    output_path: str | Path,
) -> Path:
    """
    Save the compliance report as a JSON file.

    Parameters
    ----------
    report : ComplianceReport
        The compliance report to save.
    output_path : str or Path
        Path for the output JSON file.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = report.model_dump(mode="json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Report – {procedure_title}</title>
    <style>
        :root {{
            --pass: #22c55e;
            --fail: #ef4444;
            --partial: #f59e0b;
            --not-found: #6b7280;
            --critical: #dc2626;
            --high: #ea580c;
            --medium: #d97706;
            --low: #2563eb;
            --info: #6b7280;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background: #f8fafc;
            padding: 2rem;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, #1e3a5f, #2563eb);
            color: white;
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}

        header .meta {{
            opacity: 0.9;
            font-size: 0.95rem;
        }}

        .score-card {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .score-card .card {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .score-card .card .number {{
            font-size: 2rem;
            font-weight: 700;
        }}

        .score-card .card .label {{
            font-size: 0.85rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .score-card .card.score .number {{
            color: {score_color};
        }}

        .score-card .card.pass .number {{ color: var(--pass); }}
        .score-card .card.fail .number {{ color: var(--fail); }}
        .score-card .card.partial .number {{ color: var(--partial); }}
        .score-card .card.notfound .number {{ color: var(--not-found); }}

        .summary-box {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            white-space: pre-line;
        }}

        .summary-box h2 {{
            margin-bottom: 0.75rem;
            color: #1e3a5f;
        }}

        .findings {{
            margin-bottom: 2rem;
        }}

        .findings h2 {{
            margin-bottom: 1rem;
            color: #1e3a5f;
        }}

        .finding {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border-left: 4px solid var(--not-found);
        }}

        .finding.status-pass {{ border-left-color: var(--pass); }}
        .finding.status-fail {{ border-left-color: var(--fail); }}
        .finding.status-partial {{ border-left-color: var(--partial); }}

        .finding-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        .finding-header h3 {{
            font-size: 1.05rem;
        }}

        .badges {{
            display: flex;
            gap: 0.5rem;
        }}

        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}

        .badge-pass {{ background: #dcfce7; color: #166534; }}
        .badge-fail {{ background: #fef2f2; color: #991b1b; }}
        .badge-partial {{ background: #fffbeb; color: #92400e; }}
        .badge-not-found {{ background: #f3f4f6; color: #374151; }}

        .badge-critical {{ background: #fef2f2; color: #991b1b; }}
        .badge-high {{ background: #fff7ed; color: #9a3412; }}
        .badge-medium {{ background: #fffbeb; color: #92400e; }}
        .badge-low {{ background: #eff6ff; color: #1e40af; }}
        .badge-info {{ background: #f3f4f6; color: #374151; }}

        .finding-body p {{
            margin-bottom: 0.5rem;
        }}

        .quote {{
            background: #f1f5f9;
            border-left: 3px solid #94a3b8;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            font-style: italic;
            font-size: 0.93rem;
            color: #475569;
            border-radius: 0 6px 6px 0;
        }}

        .recommendation {{
            background: #eff6ff;
            border-left: 3px solid #3b82f6;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            font-size: 0.93rem;
            border-radius: 0 6px 6px 0;
        }}

        .reg-refs {{
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }}

        footer {{
            text-align: center;
            color: #9ca3af;
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Safety Compliance Audit Report</h1>
        <div class="meta">
            <strong>Procedure:</strong> {procedure_title}<br>
            <strong>Source:</strong> {procedure_source}<br>
            <strong>Standards:</strong> {standards_list}<br>
            <strong>Generated:</strong> {generated_at}<br>
            <strong>Report ID:</strong> {report_id}
        </div>
    </header>

    <div class="score-card">
        <div class="card score">
            <div class="number">{compliance_score}%</div>
            <div class="label">Compliance Score</div>
        </div>
        <div class="card pass">
            <div class="number">{passed}</div>
            <div class="label">Passed</div>
        </div>
        <div class="card fail">
            <div class="number">{failed}</div>
            <div class="label">Failed</div>
        </div>
        <div class="card partial">
            <div class="number">{partial}</div>
            <div class="label">Partial</div>
        </div>
        <div class="card notfound">
            <div class="number">{not_found}</div>
            <div class="label">Not Found</div>
        </div>
    </div>

    <div class="summary-box">
        <h2>Executive Summary</h2>
        <p>{summary}</p>
    </div>

    <div class="findings">
        <h2>Detailed Findings ({total_checks} items)</h2>
        {findings_html}
    </div>

    <footer>
        AI-Driven Compliance Checking System &middot;
        Generated {generated_at}
    </footer>
</div>
</body>
</html>"""


_FINDING_TEMPLATE = """
<div class="finding status-{status_lower}">
    <div class="finding-header">
        <h3>{issue_id}: {requirement_summary}</h3>
        <div class="badges">
            <span class="badge badge-{status_lower}">{status}</span>
            <span class="badge badge-{severity_lower}">{severity}</span>
            <span class="badge">{category}</span>
        </div>
    </div>
    <div class="finding-body">
        <p>{description}</p>
        {doc_quote_html}
        {reg_quote_html}
        {recommendation_html}
        <div class="reg-refs">📖 {reg_refs}</div>
    </div>
</div>
"""


def _status_badge(status: str) -> str:
    return status.lower().replace("_", "-")


def _build_finding_html(finding) -> str:
    """Render a single finding as HTML."""
    doc_quote_html = ""
    if finding.document_quote:
        doc_quote_html = (
            f'<div class="quote"><strong>Procedure states:</strong> '
            f'"{finding.document_quote}"</div>'
        )

    reg_quote_html = ""
    if finding.regulation_quote:
        reg_quote_html = (
            f'<div class="quote"><strong>Regulation states:</strong> '
            f'"{finding.regulation_quote}"</div>'
        )

    recommendation_html = ""
    if finding.recommendation:
        recommendation_html = (
            f'<div class="recommendation"><strong>Recommendation:</strong> '
            f'{finding.recommendation}</div>'
        )

    reg_refs = ", ".join(finding.regulatory_reference) if finding.regulatory_reference else "N/A"

    return _FINDING_TEMPLATE.format(
        status_lower=_status_badge(finding.status.value),
        status=finding.status.value,
        severity_lower=finding.severity.value.lower(),
        severity=finding.severity.value,
        category=finding.category.value,
        issue_id=finding.issue_id,
        requirement_summary=finding.requirement_summary,
        description=finding.description,
        doc_quote_html=doc_quote_html,
        reg_quote_html=reg_quote_html,
        recommendation_html=recommendation_html,
        reg_refs=reg_refs,
    )


def save_html_report(
    report: ComplianceReport,
    output_path: str | Path,
) -> Path:
    """
    Save the compliance report as a styled HTML file.

    Parameters
    ----------
    report : ComplianceReport
        The compliance report to render.
    output_path : str or Path
        Path for the output HTML file.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine score color
    if report.compliance_score >= 80:
        score_color = "#22c55e"
    elif report.compliance_score >= 60:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"

    # Sort findings: FAIL/NOT_FOUND first, then by severity
    severity_order = {
        IssueSeverity.CRITICAL: 0,
        IssueSeverity.HIGH: 1,
        IssueSeverity.MEDIUM: 2,
        IssueSeverity.LOW: 3,
        IssueSeverity.INFO: 4,
    }
    status_order = {
        ComplianceStatus.FAIL: 0,
        ComplianceStatus.NOT_FOUND: 1,
        ComplianceStatus.PARTIAL: 2,
        ComplianceStatus.PASS: 3,
    }
    sorted_findings = sorted(
        report.findings,
        key=lambda f: (status_order.get(f.status, 9), severity_order.get(f.severity, 9)),
    )

    findings_html = "\n".join(_build_finding_html(f) for f in sorted_findings)

    html = _HTML_TEMPLATE.format(
        procedure_title=report.procedure_title,
        procedure_source=report.procedure_source,
        standards_list=", ".join(report.applicable_standards),
        generated_at=report.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        report_id=report.report_id,
        compliance_score=report.compliance_score,
        score_color=score_color,
        passed=report.passed,
        failed=report.failed,
        partial=report.partial,
        not_found=report.not_found,
        total_checks=report.total_checks,
        summary=report.summary,
        findings_html=findings_html,
    )

    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Console (Rich) report
# ---------------------------------------------------------------------------

def print_console_report(report: ComplianceReport) -> None:
    """Print a rich console report using the `rich` library."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
    except ImportError:
        # Fallback to plain print
        _print_plain_report(report)
        return

    console = Console()

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold white]Safety Compliance Audit Report[/bold white]\n"
            f"[dim]{report.procedure_title}[/dim]\n"
            f"[dim]Standards: {', '.join(report.applicable_standards)}[/dim]",
            style="bold blue",
            padding=(1, 2),
        )
    )

    # Score
    score = report.compliance_score
    if score >= 80:
        score_style = "bold green"
    elif score >= 60:
        score_style = "bold yellow"
    else:
        score_style = "bold red"

    console.print(f"\n  Compliance Score: [{score_style}]{score}%[/{score_style}]")
    console.print(
        f"  [green]✓ {report.passed} Pass[/green]  "
        f"[red]✗ {report.failed} Fail[/red]  "
        f"[yellow]◐ {report.partial} Partial[/yellow]  "
        f"[dim]? {report.not_found} Not Found[/dim]"
    )

    # Findings table
    table = Table(
        title="\nDetailed Findings",
        box=box.ROUNDED,
        show_lines=True,
        pad_edge=True,
    )
    table.add_column("ID", style="bold", width=12)
    table.add_column("Status", width=10)
    table.add_column("Severity", width=10)
    table.add_column("Requirement", width=35)
    table.add_column("Description", width=45)

    status_colors = {
        ComplianceStatus.PASS: "green",
        ComplianceStatus.FAIL: "red",
        ComplianceStatus.PARTIAL: "yellow",
        ComplianceStatus.NOT_FOUND: "dim",
    }
    severity_colors = {
        IssueSeverity.CRITICAL: "bold red",
        IssueSeverity.HIGH: "red",
        IssueSeverity.MEDIUM: "yellow",
        IssueSeverity.LOW: "blue",
        IssueSeverity.INFO: "dim",
    }

    for f in report.findings:
        s_color = status_colors.get(f.status, "white")
        sv_color = severity_colors.get(f.severity, "white")
        table.add_row(
            f.issue_id,
            f"[{s_color}]{f.status.value}[/{s_color}]",
            f"[{sv_color}]{f.severity.value}[/{sv_color}]",
            f.requirement_summary[:60],
            f.description[:80] + ("..." if len(f.description) > 80 else ""),
        )

    console.print(table)

    # Summary
    console.print(
        Panel(report.summary, title="Executive Summary", style="blue")
    )
    console.print()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _severity_emoji(severity: IssueSeverity) -> str:
    return {
        IssueSeverity.CRITICAL: "🔴",
        IssueSeverity.HIGH: "🟠",
        IssueSeverity.MEDIUM: "🟡",
        IssueSeverity.LOW: "🔵",
        IssueSeverity.INFO: "⚪",
    }.get(severity, "⚪")


def _status_icon(status: ComplianceStatus) -> str:
    return {
        ComplianceStatus.PASS: "✅",
        ComplianceStatus.FAIL: "❌",
        ComplianceStatus.PARTIAL: "⚠️",
        ComplianceStatus.NOT_FOUND: "❓",
    }.get(status, "❓")


def _score_bar(score: float, width: int = 20) -> str:
    """Render a text-based progress bar: ████████░░░░ 40.7%"""
    filled = round(score / 100 * width)
    empty = width - filled
    return f"{'█' * filled}{'░' * empty} {score}%"


def _score_grade(score: float) -> str:
    if score >= 90:
        return "🟢 Excellent"
    elif score >= 75:
        return "🟡 Good"
    elif score >= 50:
        return "🟠 Needs Improvement"
    else:
        return "🔴 Critical — Immediate Action Required"


def save_markdown_report(
    report: ComplianceReport,
    output_path: str | Path,
) -> Path:
    """
    Save the compliance report as a polished Markdown file.

    Parameters
    ----------
    report : ComplianceReport
        The compliance report to render.
    output_path : str or Path
        Path for the output .md file.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort findings: FAIL/NOT_FOUND first, then by severity
    severity_order = {
        IssueSeverity.CRITICAL: 0,
        IssueSeverity.HIGH: 1,
        IssueSeverity.MEDIUM: 2,
        IssueSeverity.LOW: 3,
        IssueSeverity.INFO: 4,
    }
    status_order = {
        ComplianceStatus.FAIL: 0,
        ComplianceStatus.NOT_FOUND: 1,
        ComplianceStatus.PARTIAL: 2,
        ComplianceStatus.PASS: 3,
    }
    sorted_findings = sorted(
        report.findings,
        key=lambda f: (
            status_order.get(f.status, 9),
            severity_order.get(f.severity, 9),
        ),
    )

    # ── Header ──────────────────────────────────────────
    lines: list[str] = []
    lines.append("# 🛡️ Safety Compliance Audit Report")
    lines.append("")
    lines.append(f"**Procedure:** {report.procedure_title}  ")
    lines.append(f"**Source:** `{report.procedure_source}`  ")
    lines.append(
        f"**Generated:** "
        f"{report.generated_at.strftime('%B %d, %Y at %I:%M %p')}  "
    )
    lines.append(f"**Report ID:** `{report.report_id}`")
    lines.append("")

    # ── Standards Checked ───────────────────────────────
    lines.append("## 📋 Standards Checked")
    lines.append("")
    for std in report.applicable_standards:
        lines.append(f"- {std}")
    lines.append("")

    # ── Scorecard ───────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 📊 Compliance Scorecard")
    lines.append("")
    lines.append(
        f"### Overall Score: {_score_bar(report.compliance_score)}"
    )
    lines.append(f"**Rating:** {_score_grade(report.compliance_score)}")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|------:|")
    lines.append(f"| ✅ Passed | {report.passed} |")
    lines.append(f"| ❌ Failed | {report.failed} |")
    lines.append(f"| ⚠️ Partial | {report.partial} |")
    lines.append(f"| ❓ Not Found | {report.not_found} |")
    lines.append(f"| **Total Checks** | **{report.total_checks}** |")
    lines.append("")

    # ── Executive Summary ───────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 📝 Executive Summary")
    lines.append("")
    lines.append(report.summary.strip())
    lines.append("")

    # ── Findings ────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(f"## 🔍 Detailed Findings ({report.total_checks} items)")
    lines.append("")

    # Summary table
    lines.append(
        "| # | ID | Status | Severity | Requirement |"
    )
    lines.append(
        "|--:|:---|:------:|:--------:|:------------|"
    )
    for i, f in enumerate(sorted_findings, 1):
        status_icon = _status_icon(f.status)
        sev_emoji = _severity_emoji(f.severity)
        req = f.requirement_summary.replace("|", "\\|")
        lines.append(
            f"| {i} | {f.issue_id} | {status_icon} {f.status.value} "
            f"| {sev_emoji} {f.severity.value} | {req} |"
        )
    lines.append("")

    # Detailed findings
    for f in sorted_findings:
        status_icon = _status_icon(f.status)
        sev_emoji = _severity_emoji(f.severity)

        lines.append("---")
        lines.append("")
        lines.append(
            f"### {status_icon} {f.issue_id} — {f.requirement_summary}"
        )
        lines.append("")
        lines.append(
            f"| Field | Value |"
        )
        lines.append("|:------|:------|")
        lines.append(f"| **Status** | {status_icon} {f.status.value} |")
        lines.append(
            f"| **Severity** | {sev_emoji} {f.severity.value} |"
        )
        lines.append(f"| **Category** | {f.category.value} |")
        if f.regulatory_reference:
            refs = ", ".join(f.regulatory_reference)
            lines.append(f"| **Regulatory Reference** | {refs} |")
        lines.append("")

        # Description
        lines.append(f"{f.description}")
        lines.append("")

        # Quotes
        if f.document_quote:
            lines.append(
                f"> **📄 Procedure states:** \"{f.document_quote}\""
            )
            lines.append("")
        if f.regulation_quote:
            lines.append(
                f"> **📖 Regulation states:** \"{f.regulation_quote}\""
            )
            lines.append("")

        # Recommendation
        if f.recommendation:
            lines.append(
                f"💡 **Recommendation:** {f.recommendation}"
            )
            lines.append("")

    # ── Footer ──────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Report generated by AI-Driven Compliance Checking System "
        f"on {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _print_plain_report(report: ComplianceReport) -> None:
    """Fallback plain-text report printer."""
    print("=" * 70)
    print(f"SAFETY COMPLIANCE AUDIT REPORT")
    print(f"Procedure: {report.procedure_title}")
    print(f"Standards: {', '.join(report.applicable_standards)}")
    print(f"Score: {report.compliance_score}%")
    print(f"Pass: {report.passed} | Fail: {report.failed} | "
          f"Partial: {report.partial} | Not Found: {report.not_found}")
    print("=" * 70)

    for f in report.findings:
        print(f"\n[{f.status.value}] {f.issue_id}: {f.requirement_summary}")
        print(f"  Severity: {f.severity.value} | Category: {f.category.value}")
        print(f"  {f.description}")
        if f.document_quote:
            print(f'  Procedure: "{f.document_quote}"')
        if f.regulation_quote:
            print(f'  Regulation: "{f.regulation_quote}"')
        if f.recommendation:
            print(f"  → Recommendation: {f.recommendation}")
        if f.regulatory_reference:
            print(f"  Refs: {', '.join(f.regulatory_reference)}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(report.summary)
    print("=" * 70)
