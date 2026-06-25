"""Tests for placeholder-anchored figure-description splicing (pure logic, injected describer)."""

from pathlib import Path

from paper_refinery.config import FigureConfig
from paper_refinery.enrich import (
    _caption_after,
    _find_mentions,
    _find_placeholders,
    _pick_image,
    enrich_markdown,
)
from paper_refinery.parse import Figure, ParseResult

MD = (
    "<page_number>1</page_number>\n\n## Intro\n\nWe reference Figure 1 here.\n\n"
    "<page_number>2</page_number>\n\n"
    "![Alt describing fig one: methods A and B.](image_url_placeholder)\n\n"
    "FIGURE 1. A comparison of methods A and B.\n\n"
    "<page_number>3</page_number>\n\n"
    "![Alt for fig two.](image_url_placeholder)\n\n"
    "FIGURE 2. Second figure caption.\n\n"
)


def test_find_placeholders_with_pages():
    phs = _find_placeholders(MD)
    assert len(phs) == 2
    assert phs[0].alt.startswith("Alt describing fig one")
    assert phs[0].page == 2 and phs[1].page == 3


def test_caption_after_placeholder():
    phs = _find_placeholders(MD)
    cap = _caption_after(MD, phs[0].end)
    assert cap.number == "1"
    assert "comparison of methods A and B" in cap.text


def test_pick_image_prefers_img_over_chart_and_by_page():
    figs = [
        Figure(Path("chart_p2_0.png"), 2),
        Figure(Path("img_p2_1.png"), 2),
        Figure(Path("img_p3_1.png"), 3),
    ]
    assert _pick_image(2, figs).image_path.name == "img_p2_1.png"  # img beats chart
    assert _pick_image(3, figs).image_path.name == "img_p3_1.png"
    assert _pick_image(10, figs) is None


def test_pick_image_prefers_img_one_page_away_over_chart_on_page():
    # figures float above their captions: caption on page 7, real image on page 6,
    # junk chart crop on page 7 -> must pick the img on page 6, not the chart on page 7
    figs = [Figure(Path("img_p6_1.png"), 6), Figure(Path("chart_p7_0.png"), 7)]
    assert _pick_image(7, figs).image_path.name == "img_p6_1.png"


def test_find_mentions_excludes_caption():
    mentions = _find_mentions(MD, "1")
    assert any("We reference Figure 1" in m for m in mentions)
    assert not any(m.startswith("FIGURE 1.") for m in mentions)


def test_enrich_splices_after_each_caption():
    figs = [Figure(Path("img_p2_1.png"), 2), Figure(Path("img_p3_1.png"), 3)]
    descs = {Path("img_p2_1.png"): "DESC-ONE", Path("img_p3_1.png"): "DESC-TWO"}
    out = enrich_markdown(ParseResult(MD, figs), describe=lambda p, c, cfg: descs[p])
    i1, i2 = out.index("FIGURE 1."), out.index("FIGURE 2.")
    assert i1 < out.index("DESC-ONE") < i2  # DESC-ONE under FIGURE 1
    assert i2 < out.index("DESC-TWO")  # DESC-TWO under FIGURE 2


def test_enrich_context_is_caption_only_by_default():
    captured = {}

    def fake(p, c, cfg):
        captured["c"] = c
        return "D"

    enrich_markdown(ParseResult(MD, [Figure(Path("img_p2_1.png"), 2)]), describe=fake)
    assert "comparison of methods A and B" in captured["c"]  # the caption
    assert "We reference Figure 1" not in captured["c"]  # mentions excluded by default


def test_enrich_includes_references_when_enabled():
    captured = {}

    def fake(p, c, cfg):
        captured["c"] = c
        return "D"

    enrich_markdown(
        ParseResult(MD, [Figure(Path("img_p2_1.png"), 2)]),
        cfg=FigureConfig(include_references=True),
        describe=fake,
    )
    assert "comparison of methods A and B" in captured["c"]  # caption
    assert "We reference Figure 1" in captured["c"]  # mentions now included


def test_enrich_falls_back_to_alt_text_when_no_image_file():
    # no extracted image -> describe is never reached -> use the placeholder's alt-text
    out = enrich_markdown(ParseResult(MD, figures=[]), describe=lambda p, c, cfg: "X")
    assert "Figure description (auto):** Alt describing fig one" in out
    assert "Figure description (auto):** X" not in out


def test_enrich_falls_back_when_gemini_drops_nonfigure():
    figs = [Figure(Path("img_p2_1.png"), 2), Figure(Path("img_p3_1.png"), 3)]
    out = enrich_markdown(ParseResult(MD, figs), describe=lambda p, c, cfg: "")  # all dropped
    assert "Figure description (auto):** Alt describing fig one" in out
    assert "Figure description (auto):** Alt for fig two" in out
