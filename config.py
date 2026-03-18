"""
Configuration for the AI-Driven Compliance Checking system.

Loads settings from environment variables and/or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
PROCEDURES_DIR = DATA_DIR / "procedures"
REGULATIONS_DIR = DATA_DIR / "regulations"
REPORTS_DIR = PROJECT_ROOT / "reports"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
VECTORSTORE_DIR = PROJECT_ROOT / ".vectorstore"

# Ensure directories exist
for _dir in [PROCEDURES_DIR, REGULATIONS_DIR, REPORTS_DIR, TEMPLATES_DIR, VECTORSTORE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # --- Google Gemini ---
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini models and embeddings",
    )
    gemini_model: str = Field(
        default="gemini-3-flash-preview",
        description="Gemini model to use for the compliance auditor",
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Gemini embedding model for vector store",
    )

    # --- LLM parameters ---
    llm_temperature: float = Field(
        default=1.0,
        description="Temperature for the auditor LLM (low = factual)",
    )
    llm_max_tokens: int = Field(
        default=16384,
        description="Max tokens for LLM responses",
    )

    # --- Vector store ---
    vector_store_type: str = Field(
        default="faiss",
        description="Vector store backend: 'faiss' or 'chroma'",
    )
    chunk_size: int = Field(
        default=800,
        description="Text chunk size for splitting regulatory documents",
    )
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between text chunks",
    )
    retrieval_top_k: int = Field(
        default=25,
        description="Number of regulatory chunks to retrieve per query",
    )

    # --- Paths ---
    data_dir: str = Field(default=str(DATA_DIR))
    procedures_dir: str = Field(default=str(PROCEDURES_DIR))
    regulations_dir: str = Field(default=str(REGULATIONS_DIR))
    reports_dir: str = Field(default=str(REPORTS_DIR))
    vectorstore_dir: str = Field(default=str(VECTORSTORE_DIR))

    # --- eCFR API ---
    ecfr_base_url: str = Field(
        default="https://www.ecfr.gov/api/versioner/v1",
        description="Base URL for the eCFR API",
    )

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
