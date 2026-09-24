"""Tests for the section-aware, soft-overlap chunker (deterministic, no network)."""

from paper_refinery.chunker import (
    Chunk,
    _drop_back_matter,
    _drop_unheaded_runs,
    _overlap_before,
    chunk_markdown,
)
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


# ---------------------------------------------------------------------------
# back matter (reference lists, back-of-book indexes)
# ---------------------------------------------------------------------------


def test_headed_reference_list_is_dropped_but_its_page_markers_stay():
    md = (
        "## 5. Conclusion\n\nWe conclude.\n\n## References\n\n[1] A. Smith. Paper. 2020.\n\n"
        "<page_number>9</page_number>\n\n[2] B. Jones. Other. 2021.\n\n## Appendix A\n\nProof."
    )
    out = _drop_back_matter(md)
    assert "Smith" not in out and "Jones" not in out
    assert "<page_number>9</page_number>" in out and "## Appendix A" in out and "Proof." in out


def test_book_index_with_letter_headings_is_dropped_up_to_the_next_section():
    md = (
        "## Index\n\nA\n\nalias templates, 63\n\n## B\n\nbraced init, 52\n\n## Z\n\nzero, 58\n\n"
        "## About the Author\n\nScott Meyers."
    )
    out = _drop_back_matter(md)
    assert "alias templates" not in out and "zero, 58" not in out and "## B" not in out
    assert "## About the Author" in out and "Scott Meyers." in out


def test_a_figure_on_a_reference_page_survives():
    md = (
        "## REFERENCES\n\n[1] A. Smith. Paper. 2020.\n\n![FIGURE 11](figures/fig_11.png)\n\n"
        "> **Figure description (auto, block diagram):** Two pipelines."
    )
    out = _drop_back_matter(md)
    assert "![FIGURE 11]" in out and "Two pipelines." in out and "Smith" not in out


def test_index_terms_heading_is_body_text():
    md = "## Index Terms\n\nsteering, control"
    assert _drop_back_matter(md) == md


def test_unheaded_index_run_is_dropped_short_runs_are_kept():
    entries = "\n\n".join(f"Term{i}, {i + 10}" for i in range(25))
    md = f"Body paragraph about lasso.\n\n{entries}\n\nPRIM, see Patient rule\n\nLast, 99"
    out = _drop_unheaded_runs(md)
    assert out.startswith("Body paragraph") and "Term3, 13" not in out and "Last, 99" not in out
    short = "\n\n".join(f"Term{i}, {i}" for i in range(5))
    assert _drop_unheaded_runs(short) == short


def test_back_matter_dropping_can_be_turned_off():
    md = "## Intro\n\n" + "Body. " * 300 + "\n\n## References\n\n[1] A. Smith. Paper. 2020."
    kept = " ".join(c.text for c in chunk_markdown(md, ChunkConfig(drop_back_matter=False)))
    dropped = " ".join(c.text for c in chunk_markdown(md))
    assert "Smith" in kept and "Smith" not in dropped


def test_numbered_subsection_titled_references_is_body_text():
    # live: gottschling-2021's "1.8.4 References" is about C++ references
    md = (
        "## 1.8.4 References\n\nThe following code introduces a reference.\n\n"
        "```cpp\nint& r = i;\n```\n\n## 1.8.5 Comparison"
    )
    assert _drop_back_matter(md) == md


def test_section_titled_index_but_holding_prose_is_kept():
    md = "## Index\n\nA B-tree index speeds up lookups by key.\n\nIt costs extra writes."
    assert _drop_back_matter(md) == md


def test_dropped_index_hands_its_last_page_to_the_next_section_only():
    index = "\n\n".join(f"<page_number>{p}</page_number>\n\nTerm{p}, {p}" for p in range(300, 305))
    md = f"Body on page 299.\n\n## Index\n\n{index}\n\n## About the Author\n\nBio."
    out = _drop_back_matter(md)
    before, after = out.split("## About the Author")
    assert "<page_number>" not in before  # the text before the index is not re-dated
    assert after.strip().startswith("<page_number>304</page_number>") and "Bio." in after
    assert "<page_number>" not in _drop_back_matter(f"Body.\n\n## Index\n\n{index}")


def test_numeric_headings_continue_an_index():
    md = "## Index\n\n## 1\n\n1-D arrays, 12\n\n## A\n\nalias, 63\n\n## About the Author\n\nBio."
    out = _drop_back_matter(md)
    assert "arrays" not in out and "alias" not in out and "Bio." in out


def test_prose_paragraphs_with_years_are_not_a_reference_run():
    prose = "\n\n".join(
        f"In {1990 + i}, Researchers extended the method to setting {i}, as discussed in the "
        "previous section, which motivates the approach taken throughout this chapter."
        for i in range(12)
    )
    assert _drop_unheaded_runs(prose) == prose


def test_a_heading_ends_an_index_run():
    entries = "\n\n".join(f"Term{i}, {i + 10}" for i in range(25))
    md = f"{entries}\n\n## Papers and reviews\n\nReal prose follows here."
    out = _drop_unheaded_runs(md)
    assert "Term3, 13" not in out and "## Papers and reviews" in out and "Real prose" in out
