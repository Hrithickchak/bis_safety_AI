"""

See: https://www.ecfr.gov/developers/documentation/api/v1
"""

from __future__ import annotations

import re
import textwrap
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import get_settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# eCFR full-text endpoint serves XML/HTML for a specific section
_ECFR_FULL_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-{title}.xml"
_ECFR_STRUCTURE_URL = "https://www.ecfr.gov/api/versioner/v1/structure/{date}/title-{title}.json"
_ECFR_SEARCH_URL = "https://www.ecfr.gov/api/search/v1/results"

# OSHA regulations live in Title 29
OSHA_TITLE = 29

# Common OSHA parts we care about
OSHA_PARTS = {
    "1910": "Occupational Safety and Health Standards (General Industry)",
    "1926": "Safety and Health Regulations for Construction",
    "1915": "Occupational Safety and Health Standards for Shipyard Employment",
}


# ---------------------------------------------------------------------------
# Helper: extract plain text from eCFR XML/HTML
# ---------------------------------------------------------------------------

def _html_to_text(html: str) -> str:
    """Convert eCFR XML/HTML content to clean plain text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove <HEAD> / <AUTH> / <SOURCE> meta blocks
    for tag in soup.find_all(["auth", "source", "head"]):
        if tag.name in ("auth", "source"):
            tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_section_from_xml(xml_text: str, section_number: str) -> str:
    """
    Extract a specific section (e.g. '1910.178') from a full-title XML
    document.  Returns the plain-text content of that section.
    """
    soup = BeautifulSoup(xml_text, "html.parser")

    # eCFR marks sections with <DIV8 N="§ 1910.178" ...> or similar
    # Try several selector strategies
    target = None

    # Strategy 1: look for a DIV with N attribute containing the section number
    for div in soup.find_all(["div5", "div6", "div7", "div8", "div9", "section"]):
        n_attr = div.get("n", "") or div.get("N", "")
        if section_number in n_attr:
            target = div
            break

    # Strategy 2: look for text containing "§ <section>"
    if target is None:
        section_marker = f"§ {section_number}"
        section_marker_alt = f"§{section_number}"
        for el in soup.find_all(string=re.compile(re.escape(section_number))):
            parent = el.find_parent(
                ["div5", "div6", "div7", "div8", "div9", "section", "div"]
            )
            if parent:
                target = parent
                break

    if target:
        return _html_to_text(str(target))

    # Fallback: return everything (may be large)
    return _html_to_text(xml_text)


# ---------------------------------------------------------------------------
# eCFR Client class
# ---------------------------------------------------------------------------

class ECFRClient:
    """
    Client for the eCFR API.

    Fetches regulatory text from the Electronic Code of Federal Regulations.

    Usage
    -----
    >>> client = ECFRClient()
    >>> text = client.get_section(title=29, part="1910", section="178")
    >>> print(text[:200])
    """

    def __init__(self, base_url: str | None = None, timeout: int = 30):
        settings = get_settings()
        self.base_url = (base_url or settings.ecfr_base_url).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/xml, application/json, text/html",
            "User-Agent": "bis-safety-ai/0.1.0",
        })

    # ---- public API ----

    def get_section(
        self,
        title: int = OSHA_TITLE,
        part: str = "1910",
        section: str = "178",
        date: str = "current",
    ) -> str:
        """
        Fetch the full text of a specific CFR section.

        Parameters
        ----------
        title : int
            CFR title number (default 29 for OSHA).
        part : str
            Part number within the title (e.g. '1910').
        section : str
            Section number (e.g. '178' for 1910.178).
        date : str
            Date for the version ('current' or 'YYYY-MM-DD').

        Returns
        -------
        str
            Plain-text content of the regulation section.
        """
        section_number = f"{part}.{section}"
        url = (
            f"{self.base_url}/full/{date}/title-{title}.xml"
            f"?part={part}&section={section_number}"
        )

        try:
            resp = self._session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            # Fall back to the structure + content endpoint
            return self._get_section_fallback(
                title, part, section, date, exc
            )

        return _extract_section_from_xml(resp.text, section_number)

    def get_part(
        self,
        title: int = OSHA_TITLE,
        part: str = "1910",
        subpart: str | None = None,
        date: str = "current",
    ) -> str:
        """
        Fetch the full text of an entire CFR part (or a specific subpart).

        Parameters
        ----------
        title : int
            CFR title number.
        part : str
            Part number (e.g. '1910').
        subpart : str, optional
            Subpart letter (e.g. 'N' for Materials Handling).
        date : str
            Version date ('current' or 'YYYY-MM-DD').

        Returns
        -------
        str
            Plain text of the regulation part/subpart.
        """
        url = f"{self.base_url}/full/{date}/title-{title}.xml?part={part}"
        if subpart:
            url += f"&subpart={subpart}"

        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        return _html_to_text(resp.text)

    def get_structure(
        self,
        title: int = OSHA_TITLE,
        date: str = "current",
    ) -> dict:
        """
        Fetch the hierarchical structure (table of contents) of a CFR title.

        Returns the JSON structure from the eCFR API.
        """
        url = f"{self.base_url}/structure/{date}/title-{title}.json"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        query: str,
        title: int = OSHA_TITLE,
        part: str | None = None,
        per_page: int = 10,
    ) -> list[dict]:
        """
        Search the eCFR for sections matching a query string.

        Parameters
        ----------
        query : str
            Search terms (e.g. 'lockout tagout hazardous energy').
        title : int
            Limit to a specific CFR title.
        part : str, optional
            Limit to a specific part.
        per_page : int
            Number of results to return.

        Returns
        -------
        list[dict]
            List of search result dicts with 'title', 'section',
            'snippet', and 'url' keys.
        """
        params = {
            "query": query,
            "per_page": per_page,
            "page": 1,
            "order": "relevance",
        }
        if title:
            params["title"] = title
        if part:
            params["part"] = part

        url = _ECFR_SEARCH_URL
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()

        data = resp.json()
        results = []
        for hit in data.get("results", []):
            results.append({
                "title": hit.get("hierarchy_title", ""),
                "section": hit.get("full_text_excerpt_url", ""),
                "snippet": hit.get("snippet", ""),
                "headings": hit.get("headings", {}),
                "starts_on": hit.get("starts_on", ""),
            })

        return results

    def fetch_and_save(
        self,
        title: int,
        part: str,
        section: str,
        output_path: str,
        date: str = "current",
    ) -> str:
        """
        Fetch a regulation section and save it to a text file.

        Parameters
        ----------
        title, part, section, date : see get_section()
        output_path : str
            File path to save the text output.

        Returns
        -------
        str
            Path to the saved file.
        """
        from pathlib import Path

        text = self.get_section(title=title, part=part, section=section, date=date)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Add a header
        header = textwrap.dedent(f"""\
            OSHA 29 CFR {part}.{section}
            Title {title} – Labor, Part {part}
            Source: Electronic Code of Federal Regulations (eCFR)
            URL: https://www.ecfr.gov/current/title-{title}/part-{part}/section-{part}.{section}
            Fetched: {date}

            {"=" * 80}
        """)

        out.write_text(header + text, encoding="utf-8")
        return str(out)

    # ---- private helpers ----

    def _get_section_fallback(
        self,
        title: int,
        part: str,
        section: str,
        date: str,
        original_error: Exception,
    ) -> str:
        """
        Fallback approach: try the content endpoint with a different URL
        pattern or return an informative error message.
        """
        section_number = f"{part}.{section}"

        # Try alternative endpoint: direct section URL
        alt_url = (
            f"https://www.ecfr.gov/current/title-{title}"
            f"/part-{part}/section-{section_number}"
        )

        try:
            resp = self._session.get(alt_url, timeout=self.timeout)
            resp.raise_for_status()
            return _html_to_text(resp.text)
        except requests.RequestException:
            pass

        raise ConnectionError(
            f"Could not fetch CFR section {section_number} from eCFR. "
            f"Original error: {original_error}. "
            f"Please download the regulation text manually from:\n"
            f"  https://www.ecfr.gov/current/title-{title}/part-{part}"
            f"/section-{section_number}\n"
            f"and save it to data/regulations/"
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def fetch_osha_standard(
    part: str = "1910",
    section: str = "178",
    output_dir: str | None = None,
) -> str:
    """
    Convenience function to fetch an OSHA standard and optionally save it.

    Parameters
    ----------
    part : str
        OSHA part (e.g. '1910').
    section : str
        Section number (e.g. '178').
    output_dir : str, optional
        If provided, saves the text to this directory.

    Returns
    -------
    str
        The regulation text.
    """
    client = ECFRClient()

    if output_dir:
        from pathlib import Path
        output_path = str(
            Path(output_dir) / f"osha_{part}_{section}.txt"
        )
        client.fetch_and_save(
            title=OSHA_TITLE,
            part=part,
            section=section,
            output_path=output_path,
        )
        return Path(output_path).read_text(encoding="utf-8")

    return client.get_section(title=OSHA_TITLE, part=part, section=section)
