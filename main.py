#!/usr/bin/env python3
"""
AI-Driven Compliance Checking for Safety Procedures – CLI Entry Point.

Usage:
    python main.py audit <procedure_file> [options]
    python main.py ingest [options]
    python main.py list-standards
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings, REGULATIONS_DIR, REPORTS_DIR

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="bis-safety-ai")
def cli():
    """AI-Driven Compliance Checking for Safety Procedures."""
    pass


# ---------------------------------------------------------------------------
# Ingest command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--regulations-dir", "-r",
    default=str(REGULATIONS_DIR),
    help="Directory containing regulation text files (.txt).",
    type=click.Path(exists=True),
)
@click.option(
    "--force-rebuild", "-f",
    is_flag=True,
    default=False,
    help="Force rebuild the vector store from scratch.",
)
def ingest(regulations_dir: str, force_rebuild: bool):
    """Ingest regulation texts into the vector store for RAG retrieval."""
    from src.regulatory_store import RegulatoryStore

    console.print("\n[bold blue]📚 Ingesting regulatory texts...[/bold blue]\n")

    import logging
    logging.basicConfig(level=logging.INFO)

    store = RegulatoryStore(force_rebuild=force_rebuild)

    reg_dir = Path(regulations_dir)
    supported = {".txt", ".pdf", ".docx", ".html", ".htm"}
    reg_files = sorted(f for f in reg_dir.iterdir() if f.suffix.lower() in supported)

    if not reg_files:
        console.print(f"[yellow]⚠️  No supported files found in {regulations_dir}[/yellow]")
        return

    total_chunks = 0
    for reg_file in reg_files:
        console.print(f"  → Ingesting [cyan]{reg_file.name}[/cyan] (rate-limited batching)...")
        n = store.ingest_file(reg_file)
        console.print(f"  ✓ {reg_file.name}: {n} chunks")
        total_chunks += n

    store.save()
    console.print(
        f"\n[green]✅ Ingested {len(reg_files)} file(s), "
        f"{total_chunks} total chunks into vector store.[/green]\n"
    )


# ---------------------------------------------------------------------------
# Audit command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("procedure_file", type=click.Path(exists=True))
@click.option(
    "--title", "-t",
    default=None,
    help="Override title for the procedure document.",
)
@click.option(
    "--output-json", "-j",
    default=None,
    help="Path to save JSON report.",
    type=click.Path(),
)
@click.option(
    "--output-html", "-h",
    default=None,
    help="Path to save HTML report.",
    type=click.Path(),
)
@click.option(
    "--model", "-m",
    default=None,
    help="Override LLM model (e.g. gemini-2.0-flash, gemini-1.5-pro).",
)
@click.option(
    "--top-k", "-k",
    default=None,
    type=int,
    help="Number of regulatory chunks to retrieve.",
)
@click.option(
    "--regulations-dir", "-r",
    default=str(REGULATIONS_DIR),
    help="Directory containing regulation text files.",
    type=click.Path(exists=True),
)
@click.option(
    "--auto-ingest / --no-auto-ingest",
    default=True,
    help="Auto-ingest regulations if vector store is empty.",
)
def audit(
    procedure_file: str,
    title: str | None,
    output_json: str | None,
    output_html: str | None,
    model: str | None,
    top_k: int | None,
    regulations_dir: str,
    auto_ingest: bool,
):
    """Run a compliance audit on a safety procedure document."""
    from src.document_loader import load_procedure
    from src.standards_mapper import StandardsMapper
    from src.regulatory_store import RegulatoryStore
    from src.compliance_auditor import ComplianceAuditor
    from src.report_generator import (
        save_json_report,
        save_html_report,
        save_markdown_report,
        print_console_report,
    )

    settings = get_settings()

    # Check API key
    if not settings.google_api_key:
        console.print(
            "[bold red]❌ GOOGLE_API_KEY not set![/bold red]\n"
            "Set it in your .env file or environment:\n"
            "  export GOOGLE_API_KEY=your-key-here\n"
        )
        sys.exit(1)

    console.print("\n[bold blue]🔍 AI Safety Compliance Auditor[/bold blue]\n")

    # Step 1: Load the procedure document
    console.print("[bold]Step 1:[/bold] Loading procedure document...")
    procedure = load_procedure(procedure_file, title=title)
    console.print(
        f"  ✓ Loaded: [cyan]{procedure.title}[/cyan] "
        f"({len(procedure.sections)} sections, "
        f"{len(procedure.raw_text):,} chars)"
    )

    # Step 2: Map to applicable standards
    console.print("[bold]Step 2:[/bold] Identifying applicable standards...")
    mapper = StandardsMapper()
    procedure = mapper.map_procedure(procedure)

    if not procedure.mapped_standards:
        console.print(
            "[yellow]  ⚠️  No standards automatically matched. "
            "Will use all available regulation texts.[/yellow]"
        )

    for topic in procedure.detected_topics:
        console.print(f"  ✓ Topic: [cyan]{topic}[/cyan]")
    for std in procedure.mapped_standards:
        console.print(f"  ✓ Standard: [green]{std}[/green]")

    # Step 3: Load/build vector store and retrieve regulatory clauses
    console.print("[bold]Step 3:[/bold] Retrieving regulatory clauses (RAG)...")
    store = RegulatoryStore()

    if not store.load() and auto_ingest:
        console.print("  → Auto-ingesting regulations...")
        n = store.ingest_directory(regulations_dir)
        store.save()
        console.print(f"  ✓ Ingested {n} chunks")

    if store.load() or store.document_count > 0:
        clauses = store.retrieve_for_procedure(
            procedure.raw_text,
            standards=procedure.mapped_standards or None,
            top_k=top_k,
        )
        console.print(f"  ✓ Retrieved {len(clauses)} relevant regulatory clauses")
    else:
        console.print(
            "[yellow]  ⚠️  No regulations in vector store. "
            "Run 'python main.py ingest' first.[/yellow]"
        )
        clauses = []

    # Step 4: Run the compliance audit
    console.print("[bold]Step 4:[/bold] Running LLM compliance audit...")
    console.print(f"  → Using model: [cyan]{model or settings.gemini_model}[/cyan]")

    auditor = ComplianceAuditor(model=model)
    report = auditor.audit(
        procedure_title=procedure.title,
        procedure_source=procedure.source_path,
        procedure_text=procedure.raw_text,
        applicable_standards=procedure.mapped_standards,
        regulatory_clauses=clauses,
    )
    console.print(
        f"  ✓ Audit complete: {len(report.findings)} findings"
    )

    # Step 5: Generate reports
    console.print("[bold]Step 5:[/bold] Generating reports...")

    # Always show console report
    print_console_report(report)

    # Save JSON if requested (or by default)
    if output_json is None:
        # Default output path
        output_json = str(
            REPORTS_DIR
            / f"{Path(procedure_file).stem}_compliance_report.json"
        )
    json_path = save_json_report(report, output_json)
    console.print(f"  ✓ JSON report: [green]{json_path}[/green]")

    # Save HTML if requested (or by default)
    if output_html is None:
        output_html = str(
            REPORTS_DIR
            / f"{Path(procedure_file).stem}_compliance_report.html"
        )
    html_path = save_html_report(report, output_html)
    console.print(f"  ✓ HTML report: [green]{html_path}[/green]")

    md_path = save_markdown_report(
        report,
        str(
            REPORTS_DIR
            / f"{Path(procedure_file).stem}_compliance_report.md"
        ),
    )
    console.print(f"  ✓ Markdown report: [green]{md_path}[/green]")

    console.print("\n[bold green]✅ Compliance audit complete![/bold green]\n")

    # Return non-zero exit code if there are failures
    if report.failed > 0 or report.not_found > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# List standards command
# ---------------------------------------------------------------------------

@cli.command("list-standards")
def list_standards():
    """List all known safety topics and their mapped standards."""
    from src.standards_mapper import StandardsMapper

    mapper = StandardsMapper()

    console.print("\n[bold blue]📋 Known Safety Standards Mappings[/bold blue]\n")

    for mapping in mapper.mappings:
        console.print(f"[bold cyan]{mapping.topic}[/bold cyan]")
        console.print(f"  {mapping.description}")
        console.print(f"  Standards: {', '.join(mapping.standards)}")
        console.print(f"  Keywords: {', '.join(mapping.keywords[:5])}...")
        console.print()


# ---------------------------------------------------------------------------
# Pipeline command (ingest + audit in one step)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("procedure_file", type=click.Path(exists=True))
@click.option("--title", "-t", default=None, help="Override procedure title.")
@click.option("--model", "-m", default=None, help="Override LLM model.")
@click.option(
    "--regulations-dir", "-r",
    default=str(REGULATIONS_DIR),
    help="Directory containing regulation text files.",
    type=click.Path(exists=True),
)
def pipeline(
    procedure_file: str,
    title: str | None,
    model: str | None,
    regulations_dir: str,
):
    """Run the full pipeline: ingest regulations → audit procedure → report."""
    from src.document_loader import load_procedure
    from src.standards_mapper import StandardsMapper
    from src.regulatory_store import RegulatoryStore
    from src.compliance_auditor import ComplianceAuditor
    from src.report_generator import (
        save_json_report,
        save_html_report,
        save_markdown_report,
        print_console_report,
    )

    settings = get_settings()

    if not settings.google_api_key:
        console.print("[bold red]❌ GOOGLE_API_KEY not set![/bold red]")
        sys.exit(1)

    import logging
    logging.basicConfig(level=logging.INFO)

    console.print("\n[bold blue]🔄 Running Full Compliance Pipeline[/bold blue]\n")

    # 1. Ingest
    console.print("[bold]Phase 1: Ingesting regulations (with rate-limit batching)...[/bold]")
    store = RegulatoryStore(force_rebuild=True)
    n = store.ingest_directory(regulations_dir)
    store.save()
    console.print(f"  ✓ {n} chunks ingested\n")

    # 2. Load procedure
    console.print("[bold]Phase 2: Loading procedure...[/bold]")
    procedure = load_procedure(procedure_file, title=title)
    console.print(f"  ✓ {procedure.title}\n")

    # 3. Map standards
    console.print("[bold]Phase 3: Mapping standards...[/bold]")
    mapper = StandardsMapper()
    procedure = mapper.map_procedure(procedure)
    for std in procedure.mapped_standards:
        console.print(f"  ✓ {std}")
    console.print()

    # 4. Retrieve clauses
    console.print("[bold]Phase 4: Retrieving regulatory clauses...[/bold]")
    clauses = store.retrieve_for_procedure(
        procedure.raw_text,
        standards=procedure.mapped_standards or None,
    )
    console.print(f"  ✓ {len(clauses)} clauses retrieved\n")

    # 5. Audit (multi-pass: one pass per standard group)
    auditor = ComplianceAuditor(model=model)
    groups = auditor._group_standards(procedure.mapped_standards)
    console.print(
        f"[bold]Phase 5: Running compliance audit "
        f"({len(groups)} pass{'es' if len(groups) > 1 else ''})...[/bold]"
    )
    for i, g in enumerate(groups, 1):
        console.print(f"  Pass {i}: {', '.join(g)}")

    report = auditor.audit(
        procedure_title=procedure.title,
        procedure_source=procedure.source_path,
        procedure_text=procedure.raw_text,
        applicable_standards=procedure.mapped_standards,
        regulatory_clauses=clauses,
    )
    console.print(f"  ✓ {len(report.findings)} findings\n")

    # 6. Reports
    console.print("[bold]Phase 6: Generating reports...[/bold]")
    print_console_report(report)

    stem = Path(procedure_file).stem
    json_path = save_json_report(
        report, REPORTS_DIR / f"{stem}_compliance_report.json"
    )
    html_path = save_html_report(
        report, REPORTS_DIR / f"{stem}_compliance_report.html"
    )
    md_path = save_markdown_report(
        report, REPORTS_DIR / f"{stem}_compliance_report.md"
    )
    console.print(f"  ✓ JSON: {json_path}")
    console.print(f"  ✓ HTML: {html_path}")
    console.print(f"  ✓ Markdown: {md_path}")
    console.print("\n[bold green]✅ Pipeline complete![/bold green]\n")


# ---------------------------------------------------------------------------
# Fetch regulations from eCFR
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("standard", type=str)
@click.option(
    "--output-dir", "-o",
    default=str(REGULATIONS_DIR),
    help="Directory to save the fetched regulation text.",
    type=click.Path(),
)
@click.option(
    "--date", "-d",
    default="current",
    help="eCFR date version ('current' or 'YYYY-MM-DD').",
)
def fetch(standard: str, output_dir: str, date: str):
    """
    Fetch a regulation from the eCFR API and save it locally.

    STANDARD should be in the form 'PART.SECTION', e.g. '1910.178' or '1910.147'.

    \b
    Examples:
        python main.py fetch 1910.178
        python main.py fetch 1910.147 --output-dir data/regulations
        python main.py fetch 1926.501 --date 2024-01-01
    """
    from src.ecfr_client import ECFRClient, OSHA_TITLE

    # Parse standard into part and section
    parts = standard.split(".")
    if len(parts) != 2:
        console.print(
            "[red]❌ STANDARD must be in PART.SECTION format "
            "(e.g. '1910.178')[/red]"
        )
        sys.exit(1)

    part, section = parts

    console.print(
        f"\n[bold blue]📥 Fetching OSHA 29 CFR {standard} from eCFR...[/bold blue]\n"
    )

    try:
        client = ECFRClient()
        output_path = str(
            Path(output_dir) / f"osha_{part}_{section}.txt"
        )
        client.fetch_and_save(
            title=OSHA_TITLE,
            part=part,
            section=section,
            output_path=output_path,
            date=date,
        )
        console.print(f"  ✓ Saved to: [green]{output_path}[/green]")
    except ConnectionError as e:
        console.print(f"[yellow]⚠️  {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)

    console.print("\n[bold green]✅ Fetch complete![/bold green]\n")


if __name__ == "__main__":
    cli()
