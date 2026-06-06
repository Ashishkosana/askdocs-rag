import pytest

from askdocs.chunking import chunk_text


def test_packs_small_paragraphs_into_one_chunk():
    text = "Para one has words.\n\nPara two has words."
    chunks = chunk_text(text, "doc.md", size=1000, overlap=0)
    assert len(chunks) == 1
    assert "Para one" in chunks[0].text and "Para two" in chunks[0].text
    assert chunks[0].source == "doc.md"
    assert chunks[0].index == 0


def test_splits_when_total_exceeds_size():
    text = "\n\n".join(f"Paragraph {i} with several words here." for i in range(20))
    chunks = chunk_text(text, "d.md", size=80, overlap=10)
    assert len(chunks) > 1
    # indexes are contiguous
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_hard_splits_single_oversized_paragraph():
    long = ("word " * 200).strip()
    chunks = chunk_text(long, "d.md", size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 120 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("x", "d.md", size=10, overlap=10)


def test_blank_input_yields_no_chunks():
    assert chunk_text("   \n\n  ", "d.md", size=100, overlap=0) == []
