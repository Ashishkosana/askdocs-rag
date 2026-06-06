"""Load source documents (.txt, .md, .pdf) and turn them into chunks."""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_text

SUPPORTED = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> str:
    """Read a single file to plain text. PDFs are extracted page by page."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # local import keeps the dep optional at import time

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def discover(corpus: Path) -> list[Path]:
    """Find every supported document under ``corpus`` (recursively)."""
    if corpus.is_file():
        return [corpus] if corpus.suffix.lower() in SUPPORTED else []
    return sorted(p for p in corpus.rglob("*") if p.suffix.lower() in SUPPORTED)


def ingest(corpus: Path, *, size: int, overlap: int) -> list[Chunk]:
    """Discover, read, and chunk every document under ``corpus``."""
    chunks: list[Chunk] = []
    for path in discover(corpus):
        text = load_file(path)
        if text.strip():
            chunks.extend(chunk_text(text, source=path.name, size=size, overlap=overlap))
    return chunks
