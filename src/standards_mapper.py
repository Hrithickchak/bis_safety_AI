"""
Standards Mapper – map safety procedure topics to regulatory standards.

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from src.models import ProcedureDocument, StandardMapping


# ---------------------------------------------------------------------------
# Default mapping data (fallback if JSON not found)
# ---------------------------------------------------------------------------

DEFAULT_MAPPINGS: list[dict] = [
    {
        "topic": "Forklift / Powered Industrial Truck Operation",
        "keywords": [
            "forklift", "fork lift", "powered industrial truck",
            "pit", "pallet jack", "order picker", "reach truck",
            "industrial truck", "material handling vehicle",
        ],
        "standards": [
            "OSHA 29 CFR 1910.178",
            "ANSI/ITSDF B56.1",
            "CSA B335",
        ],
        "description": (
            "Requirements for the design, maintenance, and use of "
            "powered industrial trucks including operator training."
        ),
    },
    {
        "topic": "Lockout / Tagout (Control of Hazardous Energy)",
        "keywords": [
            "lockout", "tagout", "loto", "lock out", "tag out",
            "hazardous energy", "energy isolation", "energy control",
        ],
        "standards": [
            "OSHA 29 CFR 1910.147",
            "ANSI/ASSE Z244.1",
            "CSA Z460",
        ],
        "description": (
            "Procedures for controlling hazardous energy during "
            "servicing and maintenance of machines and equipment."
        ),
    },
    {
        "topic": "Confined Space Entry",
        "keywords": [
            "confined space", "permit-required confined space",
            "permit required", "prcs", "entry permit",
            "atmospheric testing", "confined area",
        ],
        "standards": [
            "OSHA 29 CFR 1910.146",
            "ANSI/ASSE Z117.1",
            "CSA Z1006",
        ],
        "description": (
            "Requirements for practices and procedures to protect "
            "employees from the hazards of entry into confined spaces."
        ),
    },
    {
        "topic": "Fall Protection",
        "keywords": [
            "fall protection", "fall arrest", "guardrail",
            "safety net", "personal fall", "harness",
            "lanyard", "anchorage", "leading edge",
            "roof work", "elevated work",
        ],
        "standards": [
            "OSHA 29 CFR 1910.28",
            "OSHA 29 CFR 1926.501",
            "ANSI/ASSP Z359.1",
            "CSA Z259",
        ],
        "description": (
            "Requirements for fall protection systems and practices "
            "for workers at elevation."
        ),
    },
    {
        "topic": "Fire Safety / Emergency Evacuation",
        "keywords": [
            "fire safety", "fire drill", "fire extinguisher",
            "evacuation", "emergency action plan", "eap",
            "fire alarm", "fire prevention", "fire protection",
            "sprinkler", "fire exit",
        ],
        "standards": [
            "OSHA 29 CFR 1910.38",
            "OSHA 29 CFR 1910.39",
            "OSHA 29 CFR 1910.157",
            "NFPA 10",
            "NFPA 101",
        ],
        "description": (
            "Emergency action plans, fire prevention plans, "
            "and portable fire extinguisher requirements."
        ),
    },
    {
        "topic": "Hazard Communication (HazCom)",
        "keywords": [
            "hazard communication", "hazcom", "ghs",
            "safety data sheet", "sds", "msds",
            "chemical labeling", "chemical hazard",
            "globally harmonized",
        ],
        "standards": [
            "OSHA 29 CFR 1910.1200",
            "ANSI/ISEA Z400.1",
            "GHS Rev 10",
        ],
        "description": (
            "Requirements for classifying chemical hazards and "
            "communicating information via labels and SDSs."
        ),
    },
    {
        "topic": "Personal Protective Equipment (PPE)",
        "keywords": [
            "ppe", "personal protective equipment",
            "safety glasses", "hard hat", "gloves",
            "hearing protection", "respirator",
            "protective clothing", "eye protection",
            "face shield",
        ],
        "standards": [
            "OSHA 29 CFR 1910.132",
            "OSHA 29 CFR 1910.133",
            "OSHA 29 CFR 1910.134",
            "OSHA 29 CFR 1910.135",
            "OSHA 29 CFR 1910.136",
            "OSHA 29 CFR 1910.138",
            "ANSI/ISEA Z87.1",
            "ANSI/ISEA Z89.1",
            "CSA Z94.1",
            "CSA Z94.3",
        ],
        "description": (
            "Requirements for assessing hazards, selecting, and "
            "using personal protective equipment."
        ),
    },
    {
        "topic": "Aerial Lift / Elevated Work Platform",
        "keywords": [
            "aerial lift", "scissor lift", "boom lift",
            "cherry picker", "elevated work platform",
            "ewp", "mewp", "man lift",
        ],
        "standards": [
            "OSHA 29 CFR 1910.67",
            "OSHA 29 CFR 1926.453",
            "ANSI/SAIA A92",
            "CSA B354",
        ],
        "description": (
            "Requirements for vehicle-mounted aerial platforms "
            "and elevating work platforms."
        ),
    },
    {
        "topic": "Electrical Safety",
        "keywords": [
            "electrical safety", "arc flash",
            "lockout tagout electrical", "nfpa 70e",
            "electrical hazard", "energized work",
            "qualified electrical worker",
        ],
        "standards": [
            "OSHA 29 CFR 1910.301-399 (Subpart S)",
            "NFPA 70E",
            "CSA Z462",
        ],
        "description": (
            "Electrical safety-related work practices, "
            "maintenance, and safety requirements."
        ),
    },
    {
        "topic": "Respiratory Protection",
        "keywords": [
            "respiratory protection", "respirator",
            "scba", "air purifying", "fit test",
            "respiratory program", "breathing apparatus",
        ],
        "standards": [
            "OSHA 29 CFR 1910.134",
            "ANSI/AIHA Z88.2",
            "CSA Z94.4",
        ],
        "description": (
            "Requirements for establishing and maintaining a "
            "respiratory protection program."
        ),
    },
]


# ---------------------------------------------------------------------------
# Mapping engine
# ---------------------------------------------------------------------------

class StandardsMapper:
    """Maps safety procedure documents to applicable regulatory standards."""

    def __init__(self, mapping_file: Optional[str | Path] = None):
        """
        Parameters
        ----------
        mapping_file : path, optional
            Path to a JSON file with custom mappings.  Falls back to
            built-in DEFAULT_MAPPINGS.
        """
        self.mappings: list[StandardMapping] = []
        raw: list[dict] = []

        if mapping_file and Path(mapping_file).exists():
            with open(mapping_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = DEFAULT_MAPPINGS

        for item in raw:
            self.mappings.append(StandardMapping(**item))

    # ---- keyword-based matching ----

    def identify_topics(self, text: str) -> list[StandardMapping]:
        """
        Scan text for known keywords and return matching StandardMappings.

        Parameters
        ----------
        text : str
            Full text of the safety procedure document.

        Returns
        -------
        list[StandardMapping]
            All mappings whose keywords appear in the text.
        """
        text_lower = text.lower()
        matched: list[StandardMapping] = []
        for mapping in self.mappings:
            for kw in mapping.keywords:
                # Use word-boundary matching for short keywords
                pattern = r"\b" + re.escape(kw.lower()) + r"\b"
                if re.search(pattern, text_lower):
                    matched.append(mapping)
                    break  # one keyword is enough to match this topic
        return matched

    def get_applicable_standards(self, text: str) -> list[str]:
        """Return a flat list of standard IDs applicable to the given text."""
        topics = self.identify_topics(text)
        standards: list[str] = []
        for t in topics:
            for s in t.standards:
                if s not in standards:
                    standards.append(s)
        return standards

    def map_procedure(self, procedure: ProcedureDocument) -> ProcedureDocument:
        """
        Enrich a ProcedureDocument with detected topics and mapped standards.

        Mutates the document in place and also returns it.
        """
        topics = self.identify_topics(procedure.raw_text)
        procedure.detected_topics = [t.topic for t in topics]
        procedure.mapped_standards = self.get_applicable_standards(
            procedure.raw_text
        )
        return procedure

    # ---- LLM-assisted mapping (optional) ----

    def identify_topics_with_llm(
        self,
        text: str,
        llm_callable=None,
    ) -> list[str]:
        """
        Use an LLM to identify applicable standards when keyword matching
        is insufficient.

        Parameters
        ----------
        text : str
            Procedure text (or a summary/excerpt).
        llm_callable : callable, optional
            A function that takes a prompt string and returns a string response.
            If None, falls back to keyword matching only.

        Returns
        -------
        list[str]
            List of standard IDs identified by the LLM.
        """
        if llm_callable is None:
            return self.get_applicable_standards(text)

        standards_list = "\n".join(
            f"- {m.topic}: {', '.join(m.standards)}"
            for m in self.mappings
        )

        prompt = (
            "You are a safety-regulation expert. Given the following excerpt "
            "from a workplace safety procedure, identify which OSHA/ANSI/CSA "
            "standards apply.\n\n"
            "Known standards:\n"
            f"{standards_list}\n\n"
            "Procedure excerpt (first 3000 chars):\n"
            f"{text[:3000]}\n\n"
            "Return ONLY a JSON list of standard IDs that apply, e.g.:\n"
            '[\"OSHA 29 CFR 1910.178\", \"CSA B335\"]\n'
        )

        response = llm_callable(prompt)
        # Parse the JSON list from the response
        try:
            import json as _json
            # Find JSON array in the response
            match = re.search(r"\[.*?\]", response, re.DOTALL)
            if match:
                return _json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        # Fallback
        return self.get_applicable_standards(text)
