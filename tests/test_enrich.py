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


def test_find_captions_matches_fig_abbreviation():
    # real GLM-OCR output from a journal using "FIG." rather than "FIGURE" -- both with
    # and without a space before the number
    md = (
        "FIG.1. Dispersion of passive scalar in parallel flow.\n\n"
        "FIG. 3. Convergence of the analytical model family.\n\n"
    )
    caps = {c.number: c for c in _find_captions(md)}
    assert set(caps) == {"1", "3"}
    assert caps["1"].text == "Dispersion of passive scalar in parallel flow."
    assert caps["3"].text == "Convergence of the analytical model family."
    assert caps["1"].upper and caps["3"].upper  # "FIG" is ALL-CAPS -> canonical form


def test_find_captions_lowercase_fig_is_not_canonical():
    md = "Fig.1 shows something in a sentence.\n\nFIG.1. The real caption.\n\n"
    caps = {c.number: c for c in _find_captions(md)}
    assert caps["1"].text == "The real caption." and caps["1"].upper


def test_find_captions_with_known_set_rejects_non_matching_lines():
    md = "Figure 1 shows stuff in prose.\n\nFIGURE 1. Real caption.\n\n"
    caps = _find_captions(md, known={"FIGURE 1. Real caption."})
    assert len(caps) == 1
    assert caps[0].text == "Real caption."


def test_find_captions_known_prefix_matches_multiline_caption_first_line():
    # a multi-line caption region lands in markdown with only its first line matching
    # the regex; the line is a prefix of the known caption text
    md = "FIGURE 2. First line of caption\n\n"
    caps = _find_captions(md, known={"FIGURE 2. First line of caption continued below"})
    assert len(caps) == 1 and caps[0].number == "2"


def test_enrich_known_captions_stop_mention_stealing_the_anchor():
    # both lines are the same case, so without known captions the *mention* would win
    # (first match). With layout-model captions provided, it can't.
    md = (
        "<page_number>2</page_number>\n\n"
        "Figure 7 shows the general trend of the comparison.\n\n"
        "Figure 7. The real caption text.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={2: [Path("page_2_fig_0.png")]},
        figure_captions={2: ["Figure 7. The real caption text."]},
    )
    out = enrich_markdown(parsed, describe=lambda c, q, cfg: {n: "DESC" for n, _ in q})
    assert "The real caption text.\n\n> **Figure description (auto):** DESC" in out
    assert "general trend of the comparison.\n\n> **Figure" not in out


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


def test_enrich_renames_crop_to_its_figure_number_before_describing(tmp_path):
    crop = tmp_path / "page_2_fig_0.png"
    crop.write_bytes(b"fake png bytes")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({crop})\n\n"
        "FIGURE 4.1. A comparison.\n\n"
    )
    captured = {}

    def fake(crops, requests, cfg):
        captured["crops"] = crops
        return {n: "D" for n, _ in requests}

    out = enrich_markdown(ParseResult(md, figure_crops={2: [crop]}), describe=fake)

    renamed = tmp_path / "fig_4.1.png"
    assert renamed.exists() and not crop.exists()
    assert captured["crops"] == [renamed]
    assert f"![FIGURE 4.1]({renamed})" in out
    assert f"![FIGURE_CROP 2:0]({crop})" not in out


def test_enrich_leaves_unmatched_crop_name_unchanged(tmp_path):
    # more crops than captions on a page: the extra crop has nothing to pair with
    crop0 = tmp_path / "page_2_fig_0.png"
    crop1 = tmp_path / "page_2_fig_1.png"
    crop0.write_bytes(b"a")
    crop1.write_bytes(b"b")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({crop0})\n\n![FIGURE_CROP 2:1]({crop1})\n\n"
        "FIGURE 5. Only one caption.\n\n"
    )
    enrich_markdown(ParseResult(md, figure_crops={2: [crop0, crop1]}), describe=lambda c, q, cfg: {})

    assert (tmp_path / "fig_5.png").exists()
    assert crop1.exists()  # unmatched; left as-is
