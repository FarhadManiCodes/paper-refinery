"""Tests for caption-anchored, page-batched figure enrichment (injected describer)."""

from pathlib import Path

from paper_refinery.config import FigureConfig
from paper_refinery.enrich import _find_captions, _find_mentions, enrich_markdown
from paper_refinery.parse import ParseResult

MD = (
    "<page_number>1</page_number>\n\n## Intro\n\nWe reference Figure 1 here.\n\n"
    "<page_number>2</page_number>\n\nFIGURE 1. A comparison of methods A and B.\n\n"
    "<page_number>3</page_number>\n\nFIGURE 2. Second figure caption.\n\n"
)
CROPS = {2: [Path("page_2_fig_0.png")], 3: [Path("page_3_fig_0.png")]}


def test_find_captions_with_pages_and_numbers():
    caps = {c.number: c for c in _find_captions(MD)}
    assert set(caps) == {"1", "2"}
    assert caps["1"].page == 2 and "comparison of methods A and B" in caps["1"].text
    assert caps["2"].page == 3


def test_find_captions_prefers_uppercase_over_a_mention_at_line_start():
    md = "Figure 1 shows stuff here in a sentence.\n\nFIGURE 1. The real caption.\n\n"
    caps = {c.number: c for c in _find_captions(md)}
    assert caps["1"].text == "The real caption." and caps["1"].upper


def test_find_mentions_excludes_caption():
    mentions = _find_mentions(MD, "1")
    assert any("We reference Figure 1" in m for m in mentions)
    assert not any(m.startswith("FIGURE 1.") for m in mentions)


def test_enrich_one_call_per_page_and_splices_after_caption():
    calls = []

    def fake(crops, requests, cfg):
        calls.append((crops[0].name, [n for n, _ in requests]))
        return {num: f"DESC-{num}" for num, _ in requests}

    out = enrich_markdown(ParseResult(MD, figure_crops=CROPS), describe=fake)
    assert ("page_2_fig_0.png", ["1"]) in calls and ("page_3_fig_0.png", ["2"]) in calls
    i1, i2 = out.index("FIGURE 1."), out.index("FIGURE 2.")
    assert i1 < out.index("DESC-1") < i2
    assert i2 < out.index("DESC-2")


def test_enrich_batches_multiple_figures_on_one_page():
    md = "<page_number>5</page_number>\n\nFIGURE 3. First.\n\nFIGURE 4. Second.\n\n"
    calls = []

    def fake(crops, requests, cfg):
        calls.append(sorted(n for n, _ in requests))
        return {num: f"D{num}" for num, _ in requests}

    out = enrich_markdown(
        ParseResult(md, figure_crops={5: [Path("page_5_fig_0.png")]}), describe=fake
    )
    assert len(calls) == 1 and calls[0] == ["3", "4"]  # one call, both figures
    assert "FIGURE 3. First.\n\n> **Figure description (auto):** D3" in out
    assert "FIGURE 4. Second.\n\n> **Figure description (auto):** D4" in out


def test_enrich_sends_all_crops_for_a_multi_figure_page():
    crops = [Path("page_2_fig_0.png"), Path("page_2_fig_1.png")]
    captured = {}

    def fake(crops_arg, requests, cfg):
        captured["crops"] = crops_arg
        return {n: "D" for n, _ in requests}

    enrich_markdown(ParseResult(MD, figure_crops={2: crops}), describe=fake)
    assert captured["crops"] == crops


def test_enrich_context_is_caption_only_by_default():
    captured = {}

    def fake(crops, requests, cfg):
        captured["reqs"] = dict(requests)
        return {n: "D" for n, _ in requests}

    enrich_markdown(ParseResult(MD, figure_crops={2: [Path("page_2_fig_0.png")]}), describe=fake)
    assert "comparison of methods A and B" in captured["reqs"]["1"]
    assert "We reference Figure 1" not in captured["reqs"]["1"]


def test_enrich_includes_references_when_enabled():
    captured = {}

    def fake(crops, requests, cfg):
        captured["reqs"] = dict(requests)
        return {n: "D" for n, _ in requests}

    enrich_markdown(
        ParseResult(MD, figure_crops={2: [Path("page_2_fig_0.png")]}),
        cfg=FigureConfig(include_references=True),
        describe=fake,
    )
    assert "We reference Figure 1" in captured["reqs"]["1"]


def test_enrich_skips_figure_with_no_description():
    out = enrich_markdown(ParseResult(MD, figure_crops=CROPS), describe=lambda c, q, cfg: {})
    assert "Figure description (auto)" not in out


def test_enrich_skips_when_no_crops_for_the_page():
    out = enrich_markdown(ParseResult(MD, figure_crops={}), describe=lambda c, q, cfg: {"1": "X"})
    assert "Figure description (auto)" not in out
