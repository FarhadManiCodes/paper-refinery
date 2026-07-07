"""Tests for the section-aware, soft-overlap chunker (deterministic, no network)."""

from paper_refinery.chunker import Chunk, _overlap_before, chunk_markdown
from paper_refinery.config import ChunkConfig


def _prose(word: str, n: int) -> str:
    return " ".join(f"{word} sentence number {i}." for i in range(n))


def test_chunks_are_produced_and_indexed():
    md = "# Title\n\n## A\n\nalpha beta gamma.\n\n## B\n\ndelta epsilon."
    chunks = chunk_markdown(md, ChunkConfig(min_chars=1))
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_small_sections_merge_into_neighbour():
    cfg = ChunkConfig(max_chars=100_000, min_chars=1000)
    md = f"# T\n\ntiny\n\n## Big\n\n{_prose('alpha', 200)}"
    chunks = chunk_markdown(md, cfg)
    # the tiny title fragment must not survive as its own chunk
    assert not any(c.text.strip() == "tiny" for c in chunks)
    assert all(len(c.text) >= cfg.min_chars for c in chunks)


def test_big_sections_are_subsplit():
    cfg = ChunkConfig(max_chars=2000, min_chars=1, overlap_lo=1, overlap_hi=10, overlap_ideal=5)
    big = _prose("alpha", 600)  # one section, well over max_chars
    chunks = chunk_markdown(f"## Big\n\n{big}", cfg)
    assert len(chunks) > 1  # the over-long section was split
    assert max(len(c.text) for c in chunks) < len(big)  # no chunk holds the whole thing


def test_overlap_helper_window_and_exact_substring():
    cfg = ChunkConfig(overlap_lo=300, overlap_hi=700, overlap_ideal=500)
    prev = _prose("alpha", 200)  # long prose, many sentence boundaries
    ov, mode = _overlap_before(prev, cfg)
    assert cfg.overlap_lo <= len(ov) <= cfg.overlap_hi
    assert ov in prev  # exact substring — no whitespace normalization
    assert mode in {"PARA", "SENT", "NL", "WORD", "HARD"}


def test_overlap_short_prev_returns_all():
    ov, mode = _overlap_before("short", ChunkConfig(overlap_lo=300))
    assert ov == "short" and mode == "ALL"


def test_overlap_present_on_every_non_first_chunk():
    cfg = ChunkConfig(
        max_chars=100_000, min_chars=300, overlap_lo=200, overlap_hi=400, overlap_ideal=300
    )
    md = (
        f"## A\n\n{_prose('alpha', 60)}\n\n"
        f"## B\n\n{_prose('beta', 60)}\n\n"
        f"## C\n\n{_prose('c', 60)}"
    )
    chunks = chunk_markdown(md, cfg)
    assert chunks[0].overlap_chars == 0
    assert all(c.overlap_chars >= cfg.overlap_lo for c in chunks[1:])


def test_pages_resolved_and_markers_stripped():
    cfg = ChunkConfig(max_chars=100_000, min_chars=1)
    md = (
        f"## A\n\n<page_number>3</page_number>\n{_prose('alpha', 5)}"
        f"\n\n## B\n\n<page_number>4</page_number>\n{_prose('beta', 5)}"
    )
    chunks = chunk_markdown(md, cfg)
    assert all("<page_number>" not in c.text for c in chunks)
    assert chunks[0].page_start == 3 and chunks[0].page_end == 3


def test_page_inherited_when_no_marker():
    cfg = ChunkConfig(max_chars=100_000, min_chars=1)
    md = (
        f"## A\n\n<page_number>5</page_number>\n{_prose('alpha', 5)}\n\n## B\n\n{_prose('beta', 5)}"
    )
    chunks = chunk_markdown(md, cfg)
    assert chunks[-1].page_start == 5  # B has no marker -> inherits page 5


def test_page_range_spans_multiple_markers():
    cfg = ChunkConfig(max_chars=100_000, min_chars=100_000)  # force one chunk
    md = "## A\n\n<page_number>1</page_number>\nx\n\n<page_number>2</page_number>\ny"
    chunks = chunk_markdown(md, cfg)
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2


def test_name_for_formats_page_range():
    assert Chunk("t", 0, 1, 2).name_for("doc") == "doc pages 1-2"
    assert Chunk("t", 0, 4, 4).name_for("doc") == "doc pages 4"
    assert Chunk("t", 3, None, None).name_for("doc") == "doc chunk 3"
