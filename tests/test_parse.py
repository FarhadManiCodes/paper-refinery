"""Tests for the LlamaParse wrapper.

The pure helpers (markdown assembly, page-render mapping) are tested directly; the
network-bound ``parse_pdf`` is exercised separately with a recorded fixture (skipped).
"""

from types import SimpleNamespace

import pytest

from paper_refinery.parse import ParseResult, _build_markdown, _page_renders


def test_parse_result_defaults_to_no_renders():
    assert ParseResult(markdown="x").page_renders == {}


def test_build_markdown_inserts_one_marker_per_page():
    pages = [SimpleNamespace(page=1, md="alpha"), SimpleNamespace(page=2, md="beta")]
    md = _build_markdown(pages)
    assert md.count("<page_number>") == 2
    assert "<page_number>1</page_number>" in md and "<page_number>2</page_number>" in md
    assert "alpha" in md and "beta" in md


def test_build_markdown_replaces_stale_inline_tags():
    pages = [SimpleNamespace(page=1, md="x <page_number>99</page_number> y")]
    md = _build_markdown(pages)
    assert "99" not in md  # stale inline tag stripped
    assert md.count("<page_number>") == 1
    assert "<page_number>1</page_number>" in md


def test_page_renders_keeps_renders_and_ignores_figure_crops():
    paths = ["/o/page_1.jpg", "/o/page_10.png", "/o/chart_p6_0.png", "/o/img_p8_1.png"]
    renders = _page_renders(paths)
    assert set(renders) == {1, 10}  # only page_N renders, crops dropped
    assert renders[1].name == "page_1.jpg"
    assert renders[10].name == "page_10.png"


@pytest.mark.skip(reason="parse_pdf needs LlamaParse (network); add with a recorded fixture")
def test_parse_pdf_builds_markers_and_page_renders():
    """When implemented: assert one <page_number> per page and page_renders covers
    every page."""
