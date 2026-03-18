"""
AI-Driven Compliance Checking for Safety Procedures.

This package provides an automated compliance auditing pipeline that:
1. Parses safety procedure documents (PDF, DOCX, TXT, HTML)
2. Maps procedures to relevant OSHA/ANSI/CSA standards
3. Uses RAG to retrieve authoritative regulatory clauses
4. Employs an LLM-based auditor to compare procedures against regulations
5. Generates structured, actionable compliance reports
6. Fetches live regulation texts from the eCFR API
"""

__version__ = "0.1.0"
