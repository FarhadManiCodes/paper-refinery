"""Tests for caption-anchored figure-description splicing (pure logic, injected describer)."""

from pathlib import Path

from paper_refinery.config import FigureConfig
from paper_refinery.enrich import (
    _assign_caption,
    _build_context,
    _find_captions,
    _find_mentions,
    _nearest_caption,
    enrich_markdown,
)
from paper_refinery.parse import Figure, ParseResult

MD = (
    "<page_number>1</page_number>\n\n## Intro\n\nSome text mentioning Figure 1 here.\n\n"
    "<page_number>2</page_number>\n\nMore. As shown in Figure 1, the trend is up.\n\n"
    "FIGURE 1. A comparison of methods A and B.\n\n## Next\n\nUnrelated paragraph."
)


def test_find_captions_picks_uppercase_with_page():
    caps = _find_captions(MD)
    assert len(caps) == 1
    assert caps[0].number == "1"
    assert "comparison of methods" in caps[0].text
    assert caps[0].page == 2  # caption sits after the page-2 marker


def test_find_mentions_excludes_the_caption():
    mentions = _find_mentions(MD, "1")
    assert any("Some text mentioning Figure 1" in m for m in mentions)
    assert any("As shown in Figure 1" in m for m in mentions)
    assert not any(m.startswith("FIGURE 1.") for m in mentions)


def test_build_context_has_caption_then_mentions():
    caps = _find_captions(MD)
    ctx = _build_context(caps[0], _find_mentions(MD, "1"))
    assert "FIGURE 1. A comparison of methods A and B." in ctx
    assert "trend is up" in ctx


def test_nearest_caption_within_one_page():
    caps = _find_captions(MD)
    assert _nearest_caption(2, caps).number == "1"
    assert _nearest_caption(3, caps).number == "1"
    assert _nearest_caption(10, caps) is None


def test_assign_caption_skips_used():
    caps = _find_captions(MD)
    assert _assign_caption(2, caps, set()).number == "1"
    assert _assign_caption(2, caps, {id(caps[0])}) is None  # the only caption is taken


def test_enrich_caption_only_context_by_default():
    captured = {}

    def fake_describe(path, context, cfg):
        captured["context"] = context
        return "A real figure description."

    parsed = ParseResult(markdown=MD, figures=[Figure(Path("f1.png"), page=2)])
    out = enrich_markdown(parsed, describe=fake_describe)
    ctx = captured["context"]
    assert "comparison of methods A and B" in ctx  # the caption (title)
    assert "trend is up" not in ctx  # in-text mentions excluded by default
    assert out.index("FIGURE 1.") < out.index("A real figure description.")


def test_enrich_includes_references_when_enabled():
    captured = {}

    def fake_describe(path, context, cfg):
        captured["context"] = context
        return "desc"

    parsed = ParseResult(markdown=MD, figures=[Figure(Path("f1.png"), page=2)])
    enrich_markdown(parsed, cfg=FigureConfig(include_references=True), describe=fake_describe)
    assert "comparison of methods A and B" in captured["context"]  # caption
    assert "trend is up" in captured["context"]  # mentions now included


def test_enrich_resolves_tie_by_greedy_page_order():
    # p8 is equidistant from captions on pages 7 and 9; greedy order must give
    # 6 -> FIGURE 1 and 8 -> FIGURE 2 (not both -> FIGURE 1).
    md = (
        "<page_number>6</page_number>\n\nx\n\n<page_number>7</page_number>\n\nFIGURE 1. A.\n\n"
        "<page_number>8</page_number>\n\ny\n\n<page_number>9</page_number>\n\nFIGURE 2. B.\n\n"
    )
    figs = [Figure(Path("p6.png"), page=6), Figure(Path("p8.png"), page=8)]
    descs = {Path("p6.png"): "D6", Path("p8.png"): "D8"}
    out = enrich_markdown(ParseResult(md, figs), describe=lambda p, c, cfg: descs[p])
    i1, i2 = out.index("FIGURE 1. A."), out.index("FIGURE 2. B.")
    assert i1 < out.index("D6") < i2  # D6 under FIGURE 1
    assert i2 < out.index("D8")  # D8 under FIGURE 2


def test_enrich_drops_non_figures():
    md = "<page_number>1</page_number>\n\nFIGURE 1. A.\n\n"
    parsed = ParseResult(markdown=md, figures=[Figure(Path("table.png"), page=1)])
    out = enrich_markdown(parsed, describe=lambda p, c, cfg: "")  # Gemini -> non-figure
    assert "Figure description (auto)" not in out
    assert out == md  # unchanged
