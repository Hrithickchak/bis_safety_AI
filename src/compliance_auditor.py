"""
Compliance Auditor – LLM-based agent that compares safety procedures
against regulatory requirements and produces structured findings.

Uses Google Gemini as the LLM backbone with a multi-step approach:
1. Receive procedure text + relevant regulatory clauses from RAG
2. Systematically check each regulatory requirement against the procedure
3. Output structured ComplianceFinding objects with citations
4. Generate an LLM-powered executive summary with prioritised action items
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

from config import get_settings
from src.models import (
    ComplianceFinding,
    ComplianceReport,
    ComplianceStatus,
    IssueCategory,
    IssueSeverity,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a strict Safety Compliance Auditor AI. Your job is to compare a \
workplace safety procedure document against the applicable OSHA, ANSI, and \
CSA regulatory requirements.

## Your Approach
1. Carefully read the PROCEDURE TEXT provided.
2. Carefully read each REGULATORY REQUIREMENT provided.
3. For EACH regulatory requirement, determine whether the procedure \
adequately addresses it.
4. Cite exact quotes from both the procedure and the regulation to support \
your assessment.

## Rules
- You MUST cite exact text from the procedure document when claiming \
compliance or non-compliance.
- You MUST reference the specific regulatory clause \
(e.g., "OSHA 1910.178(l)(4)").
- You MUST NOT hallucinate or invent requirements that are not in the \
provided regulatory text.
- You MUST NOT assume the procedure covers a topic unless the text \
explicitly addresses it.
- Be thorough: check for missing sections, outdated practices, incorrect \
values, and vague language.
- Consider both the letter and spirit of the regulations.

## Output Format
Return a JSON array of findings. Each finding must be a JSON object with \
these fields:
{
  "issue_id": "string - short identifier like 'FORK-001'",
  "status": "PASS | FAIL | PARTIAL | NOT_FOUND",
  "severity": "CRITICAL | HIGH | MEDIUM | LOW | INFO",
  "category": "Missing Section | Outdated Practice | Incorrect Value | \
Vague Language | Contradiction | Incomplete Reference | Best Practice Gap",
  "requirement_summary": "string - one-line summary of what the regulation \
requires",
  "regulatory_reference": ["list of regulation references, e.g. \
'OSHA 29 CFR 1910.178(l)(4)'"],
  "document_quote": "string or null - exact quote from the procedure \
document",
  "regulation_quote": "string or null - exact quote from the regulation \
text",
  "description": "string - detailed explanation of the compliance status",
  "recommendation": "string - specific corrective action if needed"
}

Return ONLY the JSON array, no additional text before or after.
"""


ANALYSIS_PROMPT_TEMPLATE = """\
## PROCEDURE DOCUMENT
Title: {procedure_title}
Source: {procedure_source}

### Full Procedure Text:
{procedure_text}

---

## APPLICABLE REGULATORY STANDARDS
The following standards have been identified as applicable:
{standards_list}

---

## RETRIEVED REGULATORY CLAUSES
The following regulatory text excerpts are the most relevant to this \
procedure:

{regulatory_clauses}

---

## INSTRUCTIONS
Analyze the procedure document above against ALL the regulatory clauses \
provided. Be exhaustive.

For EVERY distinct regulatory requirement in the clauses above, create \
a finding entry. Do not skip any requirement — even if it seems minor.

Check for:
1. **Missing requirements**: Things the regulation requires but the \
procedure doesn't mention at all.
2. **Incorrect values**: Numbers, frequencies, or thresholds that don't \
match the regulation.
3. **Vague language**: Areas where the procedure is too vague to satisfy \
the specific regulation wording.
4. **Outdated practices**: Anything that conflicts with current regulatory \
requirements.
5. **Compliant items**: Also note items where the procedure IS compliant \
(status: PASS) — this is important for a balanced report.

You MUST produce at least 10 findings. Cover every major section of the \
regulatory clauses provided. A thorough audit typically yields 15-30 \
findings per standard.

Return your findings as a JSON array.
"""


SUMMARY_PROMPT_TEMPLATE = """\
You are an expert safety compliance consultant. Based on the structured \
audit findings below, write a concise **executive summary** (3-6 \
paragraphs) suitable for a safety manager.

Include:
1. An overview sentence stating the procedure name, standards checked, \
and overall compliance score.
2. The most critical gaps (FAIL / NOT_FOUND items with HIGH or CRITICAL \
severity) and why they matter.
3. Areas where the procedure is compliant (PASS items).
4. A prioritised list of recommended next steps.
5. A closing statement on the overall risk posture.

## Audit Data
Procedure: {procedure_title}
Standards: {standards_list}
Compliance Score: {compliance_score}%
Pass: {passed} | Fail: {failed} | Partial: {partial} | Not Found: {not_found}

## Findings (JSON)
{findings_json}

Write the summary in clear, professional prose. Do NOT use JSON. \
Do NOT repeat every finding—focus on the most impactful items.
"""


# ---------------------------------------------------------------------------
# Auditor class
# ---------------------------------------------------------------------------

class ComplianceAuditor:
    """
    LLM-powered compliance auditor that checks procedures against regulations.

    Uses Google Gemini via the ``google-genai`` SDK.
    """

    def __init__(self, model: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.gemini_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.api_key = settings.google_api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the Google GenAI client."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        thinking_level: str | None = None,
    ) -> str:
        """Make a call to Gemini and return the response text.

        Includes retry logic with exponential backoff for rate-limit errors.
        If json_mode is True, instructs Gemini to return valid JSON.
        thinking_level controls Gemini 3's reasoning depth:
            "minimal", "low", "medium", "high" (default for Flash).
            Use "low" for structured output to avoid thinking tokens
            consuming the output budget.
        """
        from google.genai import types

        client = self._get_client()
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens

        config_kwargs: dict = {
            "system_instruction": system_prompt,
            "temperature": temp,
            "max_output_tokens": tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if thinking_level:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    # Parse suggested wait time from error, default to backoff
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
                    # Try to extract the retry delay from the error message
                    import re as _re
                    delay_match = _re.search(r"retry in (\d+(?:\.\d+)?)s", err_str)
                    if delay_match:
                        wait = max(float(delay_match.group(1)) + 2, wait)
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %.0fs before retry...",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise  # Non-rate-limit error, re-raise immediately

        raise RuntimeError(
            f"Gemini API rate limit exceeded after {max_retries} retries. "
            "Please wait a few minutes and try again."
        )

    @staticmethod
    def _repair_json(raw: str) -> str:
        """Attempt to fix common JSON issues from LLM output."""
        # Remove trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", raw)
        # Add missing commas between } {  (i.e. between consecutive objects)
        repaired = re.sub(r"}\s*\n\s*{", "},\n{", repaired)
        return repaired

    @staticmethod
    def _extract_objects_individually(raw: str) -> list[dict]:
        """Last-resort parser: extract individual JSON objects from broken array."""
        objects: list[dict] = []
        # Find each top-level { ... } block
        depth = 0
        start = None
        for i, ch in enumerate(raw):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    fragment = raw[start : i + 1]
                    try:
                        obj = json.loads(fragment)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass  # skip unparseable fragment
                    start = None
        return objects

    def _parse_findings(self, llm_response: str) -> list[ComplianceFinding]:
        """Parse the LLM response JSON into ComplianceFinding objects."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", llm_response)
        cleaned = re.sub(r"```\s*$", "", cleaned)

        # Extract JSON array from the response
        json_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not json_match:
            raise ValueError(
                f"Could not find JSON array in LLM response:\n"
                f"{llm_response[:500]}"
            )

        raw_json = json_match.group()
        try:
            items = json.loads(raw_json)
        except json.JSONDecodeError:
            # Attempt to repair common LLM JSON mistakes
            logger.warning("Initial JSON parse failed, attempting repair...")
            repaired = self._repair_json(raw_json)
            try:
                items = json.loads(repaired)
                logger.info("JSON repair succeeded.")
            except json.JSONDecodeError as e2:
                logger.error("JSON repair also failed: %s", e2)
                # Last resort: try parsing individual objects
                items = self._extract_objects_individually(raw_json)
                if not items:
                    raise ValueError(
                        f"Invalid JSON in LLM response (repair failed): {e2}"
                    )

        findings: list[ComplianceFinding] = []
        for item in items:
            try:
                # Normalize status
                status_str = item.get("status", "NOT_FOUND").upper().strip()
                # Handle edge cases like "NOT FOUND" (no underscore)
                status_str = status_str.replace(" ", "_")
                status = ComplianceStatus(status_str)

                # Normalize severity
                severity_str = item.get("severity", "MEDIUM").upper().strip()
                try:
                    severity = IssueSeverity(severity_str)
                except ValueError:
                    severity = IssueSeverity.MEDIUM

                # Normalize category
                category_str = item.get("category", "Missing Section")
                try:
                    category = IssueCategory(category_str)
                except ValueError:
                    category = IssueCategory.MISSING_SECTION

                # Ensure regulatory_reference is always a list
                reg_ref = item.get("regulatory_reference", [])
                if isinstance(reg_ref, str):
                    reg_ref = [reg_ref]

                finding = ComplianceFinding(
                    issue_id=item.get(
                        "issue_id",
                        f"ISSUE-{uuid.uuid4().hex[:6].upper()}",
                    ),
                    status=status,
                    severity=severity,
                    category=category,
                    requirement_summary=item.get("requirement_summary", ""),
                    regulatory_reference=reg_ref,
                    document_quote=item.get("document_quote"),
                    regulation_quote=item.get("regulation_quote"),
                    description=item.get("description", ""),
                    recommendation=item.get("recommendation", ""),
                )
                findings.append(finding)
            except Exception as exc:
                logger.warning("Skipping malformed finding: %s – %s", item, exc)
                continue

        return findings

    # ---- standard grouping for multi-pass audit ----

    @staticmethod
    def _group_standards(standards: list[str]) -> list[list[str]]:
        """
        Group related standards so each LLM call focuses on a manageable
        set of regulations.  Standards containing the same base identifier
        (e.g. all "1910.178" variants) are kept together.
        """
        groups: dict[str, list[str]] = {}
        for std in standards:
            # Extract a short key — e.g. "1910.178" from
            # "OSHA 29 CFR 1910.178" or "ANSI/ITSDF B56.1"
            key = std.strip()
            # Try to grab the numeric portion for OSHA-style refs
            import re as _re
            m = _re.search(r"(\d{4}\.\d+)", std)
            if m:
                key = m.group(1)
            else:
                # For ANSI/CSA/NFPA — use the alphanumeric code
                m2 = _re.search(r"([A-Z]+[\s/-]?[A-Z]*\d+)", std)
                if m2:
                    key = m2.group(1)
            groups.setdefault(key, []).append(std)

        # Merge tiny groups (< 2 standards) into a single "other" batch
        result: list[list[str]] = []
        small_batch: list[str] = []
        for key, stds in groups.items():
            if len(stds) >= 2:
                result.append(stds)
            else:
                small_batch.extend(stds)
            # Keep batches manageable
            if len(small_batch) >= 5:
                result.append(small_batch)
                small_batch = []
        if small_batch:
            result.append(small_batch)

        return result if result else [standards]

    # ---- single-pass audit (internal) ----

    def _audit_single_pass(
        self,
        procedure_title: str,
        procedure_source: str,
        procedure_text: str,
        standards_subset: list[str],
        regulatory_clauses: list[dict],
        pass_label: str = "",
    ) -> list[ComplianceFinding]:
        """Run one audit pass for a subset of standards."""
        # Format regulatory clauses for the prompt
        clauses_text = ""
        for i, clause in enumerate(regulatory_clauses, 1):
            source = clause.get("metadata", {}).get("source", "Unknown")
            text = clause.get("text", "")
            clauses_text += f"\n### Clause {i} (Source: {source})\n{text}\n"

        standards_text = "\n".join(f"- {s}" for s in standards_subset)

        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            procedure_title=procedure_title,
            procedure_source=procedure_source,
            procedure_text=procedure_text[:30000],
            standards_list=standards_text,
            regulatory_clauses=clauses_text[:60000],
        )

        # Retry up to 3 times if JSON parsing fails
        findings = None
        for parse_attempt in range(3):
            logger.info(
                "Audit pass %s — calling Gemini (%s), attempt %d",
                pass_label, self.model, parse_attempt + 1,
            )
            llm_response = self._call_llm(
                SYSTEM_PROMPT, user_prompt,
                json_mode=True, thinking_level="low",
            )
            logger.info(
                "Gemini response received (%d chars)", len(llm_response)
            )

            try:
                findings = self._parse_findings(llm_response)
                break
            except ValueError as parse_err:
                logger.warning(
                    "JSON parse attempt %d failed: %s\nRaw response:\n%s",
                    parse_attempt + 1,
                    parse_err,
                    llm_response[:2000],
                )
                if parse_attempt < 2:
                    logger.info("Retrying LLM call...")
                    time.sleep(5)
                else:
                    raise

        assert findings is not None
        return findings

    # ---- main audit entry point ----

    def audit(
        self,
        procedure_title: str,
        procedure_source: str,
        procedure_text: str,
        applicable_standards: list[str],
        regulatory_clauses: list[dict],
    ) -> ComplianceReport:
        """
        Run a full compliance audit.

        When multiple standard groups are identified the audit is split
        into separate LLM passes — one per group — so each set of
        regulations gets focused attention.  Findings are merged and
        deduplicated before scoring.

        Parameters
        ----------
        procedure_title : str
            Title of the procedure document.
        procedure_source : str
            File path or URL of the source.
        procedure_text : str
            Full text of the safety procedure.
        applicable_standards : list[str]
            List of applicable standard IDs.
        regulatory_clauses : list[dict]
            Retrieved regulatory clause chunks (from RegulatoryStore.search).
            Each dict has 'text' and 'metadata' keys.

        Returns
        -------
        ComplianceReport
            Structured compliance report with findings and score.
        """
        groups = self._group_standards(applicable_standards)
        logger.info(
            "Standards split into %d audit pass(es): %s",
            len(groups),
            [", ".join(g) for g in groups],
        )

        all_findings: list[ComplianceFinding] = []
        seen_ids: set[str] = set()

        for idx, std_group in enumerate(groups, 1):
            pass_label = f"{idx}/{len(groups)}"
            logger.info(
                "Starting audit pass %s for: %s", pass_label,
                ", ".join(std_group),
            )

            # Filter clauses to those relevant to this standard group
            group_clauses = self._filter_clauses_for_standards(
                regulatory_clauses, std_group
            )
            # Fall back to all clauses if filter is too aggressive
            if len(group_clauses) < 3:
                group_clauses = regulatory_clauses

            findings = self._audit_single_pass(
                procedure_title=procedure_title,
                procedure_source=procedure_source,
                procedure_text=procedure_text,
                standards_subset=std_group,
                regulatory_clauses=group_clauses,
                pass_label=pass_label,
            )

            # Deduplicate across passes by issue_id
            for f in findings:
                if f.issue_id not in seen_ids:
                    seen_ids.add(f.issue_id)
                    all_findings.append(f)

            # Pause between passes to respect rate limits
            if idx < len(groups):
                logger.info("Pausing 5s between audit passes...")
                time.sleep(5)

        # Build report
        report = ComplianceReport(
            report_id=f"RPT-{uuid.uuid4().hex[:8].upper()}",
            procedure_title=procedure_title,
            procedure_source=procedure_source,
            applicable_standards=applicable_standards,
            findings=all_findings,
        )
        report.compute_score()

        # Generate executive summary (LLM-powered with fallback)
        report.summary = self._generate_summary(report)

        return report

    @staticmethod
    def _filter_clauses_for_standards(
        clauses: list[dict], standards: list[str]
    ) -> list[dict]:
        """Return clauses whose source mentions any of the given standards."""
        keywords: list[str] = []
        for std in standards:
            # Extract searchable fragments
            # e.g. "OSHA 29 CFR 1910.178" → ["1910.178", "osha"]
            import re as _re
            numbers = _re.findall(r"\d{2,}\.\d+", std)
            keywords.extend(n.lower() for n in numbers)
            codes = _re.findall(r"[A-Z]{2,}\d+", std)
            keywords.extend(c.lower() for c in codes)
            # Also add simple words like "b56", "b335"
            codes2 = _re.findall(r"[A-Za-z]+\d+", std)
            keywords.extend(c.lower() for c in codes2)

        if not keywords:
            return clauses

        matched = []
        for c in clauses:
            source = c.get("metadata", {}).get("source", "").lower()
            text_lower = c.get("text", "").lower()[:200]
            if any(kw in source or kw in text_lower for kw in keywords):
                matched.append(c)
        return matched

    def _generate_summary(self, report: ComplianceReport) -> str:
        """
        Generate an executive summary.

        Attempts an LLM-powered summary for richer insights; falls back
        to a deterministic template if the LLM call fails.
        """
        try:
            return self._generate_llm_summary(report)
        except Exception as exc:
            logger.warning(
                "LLM summary generation failed (%s); using template fallback.",
                exc,
            )
            return self._generate_template_summary(report)

    def _generate_llm_summary(self, report: ComplianceReport) -> str:
        """Call Gemini to produce a rich executive summary."""
        # Build a compact JSON representation of findings for the prompt
        compact_findings = []
        for f in report.findings:
            compact_findings.append({
                "id": f.issue_id,
                "status": f.status.value,
                "severity": f.severity.value,
                "category": f.category.value,
                "requirement": f.requirement_summary,
                "description": f.description[:200],
                "recommendation": f.recommendation[:200],
            })

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            procedure_title=report.procedure_title,
            standards_list=", ".join(report.applicable_standards),
            compliance_score=report.compliance_score,
            passed=report.passed,
            failed=report.failed,
            partial=report.partial,
            not_found=report.not_found,
            findings_json=json.dumps(compact_findings, indent=2)[:6000],
        )

        summary = self._call_llm(
            system_prompt=(
                "You are a professional safety compliance report writer. "
                "Write clear, actionable executive summaries."
            ),
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=1500,
        )

        return summary.strip()

    @staticmethod
    def _generate_template_summary(report: ComplianceReport) -> str:
        """Deterministic fallback summary built from report data."""
        critical_count = sum(
            1 for f in report.findings
            if f.severity in (IssueSeverity.CRITICAL, IssueSeverity.HIGH)
            and f.status in (ComplianceStatus.FAIL, ComplianceStatus.NOT_FOUND)
        )

        summary_parts = [
            f"Compliance Audit Report for: {report.procedure_title}",
            f"Standards checked: {', '.join(report.applicable_standards)}",
            f"Total checks: {report.total_checks}",
            (
                f"Results: {report.passed} PASS, {report.failed} FAIL, "
                f"{report.partial} PARTIAL, {report.not_found} NOT FOUND"
            ),
            f"Overall compliance score: {report.compliance_score}%",
        ]

        if critical_count > 0:
            summary_parts.append(
                f"\n⚠️  {critical_count} CRITICAL/HIGH severity issue(s) "
                f"require immediate attention."
            )

        if report.compliance_score >= 90:
            summary_parts.append(
                "\n✅ The procedure is largely compliant with applicable "
                "standards."
            )
        elif report.compliance_score >= 70:
            summary_parts.append(
                "\n⚠️  The procedure has moderate compliance gaps that "
                "should be addressed."
            )
        else:
            summary_parts.append(
                "\n❌ The procedure has significant compliance gaps that "
                "require urgent revision."
            )

        return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Convenience function for single-shot audits
# ---------------------------------------------------------------------------

def run_audit(
    procedure_text: str,
    procedure_title: str = "Untitled Procedure",
    procedure_source: str = "unknown",
    applicable_standards: list[str] | None = None,
    regulatory_clauses: list[dict] | None = None,
    model: str | None = None,
) -> ComplianceReport:
    """
    Convenience function to run a single compliance audit.

    If regulatory_clauses are not provided, returns an empty report.
    """
    auditor = ComplianceAuditor(model=model)
    return auditor.audit(
        procedure_title=procedure_title,
        procedure_source=procedure_source,
        procedure_text=procedure_text,
        applicable_standards=applicable_standards or [],
        regulatory_clauses=regulatory_clauses or [],
    )
