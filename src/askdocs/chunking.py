"""Split documents into overlapping, paragraph-aware chunks.

Chunking is the single biggest lever on retrieval quality: too large and a chunk
dilutes the relevant sentence; too small and it loses the context needed to answer.
We pack whole paragraphs up to ``size`` characters and carry ``overlap`` characters
of tail into the next chunk so a fact split across a boundary stays retrievable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text plus enough metadata to cite it."""

    text: str
    source: str
    index: int  # position of this chunk within its source document


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]


def chunk_text(text: str, source: str, *, size: int, overlap: int) -> list[Chunk]:
    """Pack paragraphs into ~``size``-char chunks with ``overlap``-char carryover."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    chunks: list[Chunk] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        body = buf.strip()
        if body:
            chunks.append(Chunk(text=body, source=source, index=len(chunks)))
        # carry the tail so context spanning a boundary survives
        buf = body[-overlap:] if overlap and len(body) > overlap else ""

    for para in _split_paragraphs(text):
        # a single oversized paragraph is hard-split on word boundaries
        if len(para) > size:
            if buf:
                flush()
            for piece in _hard_split(para, size, overlap):
                chunks.append(Chunk(text=piece, source=source, index=len(chunks)))
            continue

        if buf and len(buf) + 2 + len(para) > size:
            flush()
        buf = f"{buf}\n\n{para}" if buf else para

    if buf.strip():
        chunks.append(Chunk(text=buf.strip(), source=source, index=len(chunks)))
    return chunks


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Fallback for a paragraph longer than ``size`` — split on whitespace."""
    words = text.split()
    pieces: list[str] = []
    cur: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > size and cur:
            pieces.append(" ".join(cur))
            # rebuild overlap tail
            tail, tail_len = [], 0
            for w in reversed(cur):
                if tail_len + len(w) + 1 > overlap:
                    break
                tail.insert(0, w)
                tail_len += len(w) + 1
            cur, length = tail, tail_len
        cur.append(word)
        length += len(word) + 1
    if cur:
        pieces.append(" ".join(cur))
    return pieces
