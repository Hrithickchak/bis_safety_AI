"""
Tests for the ecfr_client module.

Covers: HTML-to-text conversion, section extraction, and client methods
(with mocked HTTP calls).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ecfr_client import (
    ECFRClient,
    _html_to_text,
    _extract_section_from_xml,
    fetch_osha_standard,
    OSHA_TITLE,
)


# ---------------------------------------------------------------------------
# Tests: _html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_basic_conversion(self):
        html = "<p>Hello <b>world</b></p>"
        text = _html_to_text(html)
        assert "Hello" in text
        assert "world" in text
        assert "<p>" not in text

    def test_removes_tags(self):
        html = "<div><auth>AUTH BLOCK</auth><p>Content here.</p></div>"
        text = _html_to_text(html)
        assert "Content here" in text
        assert "AUTH BLOCK" not in text

    def test_empty_html(self):
        assert _html_to_text("") == ""

    def test_collapses_blank_lines(self):
        html = "<p>Line 1</p>\n\n\n\n\n<p>Line 2</p>"
        text = _html_to_text(html)
        assert "\n\n\n" not in text


# ---------------------------------------------------------------------------
# Tests: _extract_section_from_xml
# ---------------------------------------------------------------------------

class TestExtractSectionFromXml:
    def test_extracts_by_n_attribute(self):
        xml = """
        <root>
            <div8 N="§ 1910.178">
                <p>Section content about forklifts.</p>
            </div8>
            <div8 N="§ 1910.179">
                <p>Other section.</p>
            </div8>
        </root>
        """
        text = _extract_section_from_xml(xml, "1910.178")
        assert "forklifts" in text
        assert "Other section" not in text

    def test_fallback_to_full_text(self):
        xml = "<root><p>Only content, no section markers.</p></root>"
        text = _extract_section_from_xml(xml, "1910.999")
        assert "Only content" in text


# ---------------------------------------------------------------------------
# Tests: ECFRClient (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestECFRClient:
    @pytest.fixture
    def client(self):
        return ECFRClient()

    @patch("src.ecfr_client.requests.Session.get")
    def test_get_section(self, mock_get: MagicMock, client: ECFRClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <root>
            <div8 N="§ 1910.178">
                <p>Forklift regulation text.</p>
            </div8>
        </root>
        """
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        text = client.get_section(title=29, part="1910", section="178")
        assert "Forklift regulation text" in text

    @patch("src.ecfr_client.requests.Session.get")
    def test_search(self, mock_get: MagicMock, client: ECFRClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {
                    "hierarchy_title": "Test Result",
                    "snippet": "lockout tagout snippet",
                    "headings": {},
                    "starts_on": "2024-01-01",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        results = client.search("lockout tagout", title=29)
        assert len(results) == 1
        assert "lockout" in results[0]["snippet"]

    @patch("src.ecfr_client.requests.Session.get")
    def test_get_structure(self, mock_get: MagicMock, client: ECFRClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"title": 29, "parts": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        structure = client.get_structure(title=29)
        assert structure["title"] == 29

    @patch("src.ecfr_client.requests.Session.get")
    def test_fetch_and_save(
        self, mock_get: MagicMock, client: ECFRClient, tmp_path
    ):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<root><p>Regulation text.</p></root>"
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        output_path = str(tmp_path / "osha_1910_178.txt")
        result = client.fetch_and_save(
            title=29, part="1910", section="178",
            output_path=output_path,
        )
        assert result == output_path
        content = (tmp_path / "osha_1910_178.txt").read_text()
        assert "OSHA 29 CFR 1910.178" in content
        assert "Regulation text" in content

    @patch("src.ecfr_client.requests.Session.get")
    def test_connection_error_with_fallback(
        self, mock_get: MagicMock, client: ECFRClient
    ):
        """Both primary and fallback fail → raises ConnectionError."""
        import requests as req
        mock_get.side_effect = req.RequestException("Connection refused")

        with pytest.raises(ConnectionError, match="Could not fetch"):
            client.get_section(title=29, part="1910", section="999")


# ---------------------------------------------------------------------------
# Tests: fetch_osha_standard convenience function
# ---------------------------------------------------------------------------

class TestFetchOshaStandard:
    @patch("src.ecfr_client.ECFRClient.get_section")
    def test_returns_text(self, mock_get_section: MagicMock):
        mock_get_section.return_value = "Fetched regulation text."
        text = fetch_osha_standard(part="1910", section="178")
        assert text == "Fetched regulation text."

    @patch("src.ecfr_client.ECFRClient.fetch_and_save")
    def test_saves_to_directory(
        self, mock_fetch_save: MagicMock, tmp_path
    ):
        mock_fetch_save.return_value = str(tmp_path / "osha_1910_178.txt")
        # Create the file so the read works
        out = tmp_path / "osha_1910_178.txt"
        out.write_text("Saved regulation text.")

        text = fetch_osha_standard(
            part="1910", section="178", output_dir=str(tmp_path)
        )
        assert "Saved regulation text" in text
