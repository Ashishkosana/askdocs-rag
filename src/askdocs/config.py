"""Runtime configuration, sourced from environment variables (prefix ``ASKDOCS_``)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for tunable knobs.

    Every field can be overridden by an environment variable, e.g.
    ``ASKDOCS_TOP_K=6`` or a line in a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="ASKDOCS_", env_file=".env", extra="ignore"
    )

    # --- Claude (generation + judge) ---
    answer_model: str = "claude-opus-4-8"
    judge_model: str = "claude-opus-4-8"
    effort: str = "medium"  # low | medium | high | max
    max_tokens: int = 1024

    # --- Embeddings (local, no API key required) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Chunking (character-based) ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- Retrieval ---
    top_k: int = 4

    # --- Vector store ---
    persist_dir: Path = Path(".chroma")
    collection: str = "askdocs"


settings = Settings()
