"""Tests for caption-anchored, per-figure enrichment (injected describer).

The describer contract: fake(crops, number, caption, context, cfg) -> dict | None,
mirroring figures.describe_figure. Geometric pairing fixtures use (x1, y1, x2, y2)
bboxes with y growing downward, captions sitting below their figures.
"""

import logging
from pathlib import Path

from paper_refinery.config import FigureConfig
from paper_refinery.enrich import _find_captions, enrich_markdown
from paper_refinery.parse import CaptionRegion, CropRegion, ParseResult

MD = (
    "<page_number>1</page_number>\n\n## Intro\n\nWe reference Figure 1 here.\n\n"
    "<page_number>2</page_number>\n\nFIGURE 1. A comparison of methods A and B.\n\n"
    "<page_number>3</page_number>\n\nFIGURE 2. Second figure caption.\n\n"
)
CROPS = {2: [CropRegion(Path("page_2_fig_0.png"))], 3: [CropRegion(Path("page_3_fig_0.png"))]}


def _desc(description="DESC", figure_type="line_plot"):
    return {"figure_type": figure_type, "description": description}


# ---------------------------------------------------------------------------
# _find_captions
# ---------------------------------------------------------------------------


def test_find_captions_with_pages_and_numbers():
    caps = {c.number: c for c in _find_captions(MD)}
    assert set(caps) == {"1", "2"}
    assert caps["1"].page == 2 and "comparison of methods A and B" in caps["1"].text
    assert caps["2"].page == 3


def test_find_captions_prefers_uppercase_over_a_mention_at_line_start():
    md = "Figure 1 shows stuff here in a sentence.\n\nFIGURE 1. The real caption.\n\n"
    caps = {c.number: c for c in _find_captions(md)}
    assert caps["1"].text == "The real caption." and caps["1"].upper


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


# ---------------------------------------------------------------------------
# enrich_markdown: anchoring + splicing (positional-fallback fixtures, no bboxes)
# ---------------------------------------------------------------------------


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
        figure_crops={2: [CropRegion(Path("page_2_fig_0.png"))]},
        figure_captions={2: [CaptionRegion("Figure 7. The real caption text.")]},
    )
    out = enrich_markdown(parsed, describe=lambda c, n, cap, ctx, cfg: _desc())
    assert "The real caption text.\n\n> **Figure description (auto, line plot):** DESC" in out
    assert "general trend of the comparison.\n\n> **Figure" not in out


def test_enrich_one_call_per_figure_and_splices_after_caption():
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((crops[0].name, number))
        return _desc(f"DESC-{number}")

    out = enrich_markdown(ParseResult(MD, figure_crops=CROPS), describe=fake)
    assert ("page_2_fig_0.png", "1") in calls and ("page_3_fig_0.png", "2") in calls
    i1, i2 = out.index("FIGURE 1."), out.index("FIGURE 2.")
    assert i1 < out.index("DESC-1") < i2
    assert i2 < out.index("DESC-2")


def test_enrich_description_label_carries_the_figure_type():
    out = enrich_markdown(
        ParseResult(MD, figure_crops=CROPS),
        describe=lambda c, n, cap, ctx, cfg: _desc("X", "convergence_plot"),
    )
    assert "> **Figure description (auto, convergence plot):** X" in out


def test_enrich_two_captions_one_page_positional_fallback():
    md = "<page_number>5</page_number>\n\nFIGURE 3. First.\n\nFIGURE 4. Second.\n\n"
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((crops[0].name, number))
        return _desc(f"D{number}")

    crops = {5: [CropRegion(Path("page_5_fig_0.png")), CropRegion(Path("page_5_fig_1.png"))]}
    out = enrich_markdown(ParseResult(md, figure_crops=crops), describe=fake)
    # no bboxes anywhere -> crop i pairs with caption i
    assert sorted(calls) == [("page_5_fig_0.png", "3"), ("page_5_fig_1.png", "4")]
    assert "FIGURE 3. First.\n\n> **Figure description (auto, line plot):** D3" in out
    assert "FIGURE 4. Second.\n\n> **Figure description (auto, line plot):** D4" in out


def test_enrich_skips_figure_when_describer_returns_none():
    out = enrich_markdown(ParseResult(MD, figure_crops=CROPS), describe=lambda *a: None)
    assert "Figure description" not in out


def test_enrich_skips_when_no_crops_for_the_page():
    def boom(*a):
        raise AssertionError("no crops -> no describe call")

    out = enrich_markdown(ParseResult(MD, figure_crops={}), describe=boom)
    assert "Figure description" not in out


def test_enrich_failed_describer_warns_and_leaves_figure_undescribed(caplog):
    def fake(crops, number, caption, context, cfg):
        if number == "1":
            raise RuntimeError("gemini down")
        return _desc("D2")

    with caplog.at_level(logging.WARNING):
        out = enrich_markdown(ParseResult(MD, figure_crops=CROPS), describe=fake)
    assert "D2" in out and "DESC-1" not in out
    assert "FIGURE 1" in caplog.text


# ---------------------------------------------------------------------------
# context assembly
# ---------------------------------------------------------------------------

CONTEXT_MD = (
    "<page_number>1</page_number>\n\n"
    "# Churning Losses in Spiral Bevel Gears\n\n"
    "Author One and Author Two\n\n"
    + "This paper investigates churning power losses in gearboxes. "
    * 8  # long => abstract
    + "\n\n<page_number>2</page_number>\n\n"
    "## Results\n\n"
    "The paragraph right before the figure.\n\n"
    "$$ E = mc^2 $$\n\n"
    "FIGURE 4. Power loss versus rotational speed.\n\n"
    "![FIGURE_CROP 2:0](page_2_fig_0.png)\n\n"
    "The paragraph right after the figure.\n\n"
    "A second trailing paragraph.\n\n"
)


def test_enrich_builds_title_abstract_and_neighbor_context():
    captured = {}

    def fake(crops, number, caption, context, cfg):
        captured.update(context, caption=caption)
        return _desc()

    enrich_markdown(
        ParseResult(CONTEXT_MD, figure_crops={2: [CropRegion(Path("page_2_fig_0.png"))]}),
        describe=fake,
    )
    assert captured["title"] == "Churning Losses in Spiral Bevel Gears"
    assert captured["abstract"].startswith("This paper investigates churning power losses")
    assert captured["caption"] == "Power loss versus rotational speed."
    # nearest PROSE neighbors: markers, headings, and math blocks are skipped
    assert "right before the figure" in captured["before"]
    assert "E = mc^2" not in captured["before"]
    assert "right after the figure" in captured["after"]
    assert "second trailing paragraph" in captured["after"]  # context_paragraphs = 2


def test_enrich_context_paragraphs_zero_sends_no_neighbors():
    captured = {}

    def fake(crops, number, caption, context, cfg):
        captured.update(context)
        return _desc()

    enrich_markdown(
        ParseResult(CONTEXT_MD, figure_crops={2: [CropRegion(Path("page_2_fig_0.png"))]}),
        cfg=FigureConfig(context_paragraphs=0),
        describe=fake,
    )
    assert captured["before"] == "" and captured["after"] == ""


# ---------------------------------------------------------------------------
# geometric pairing + renaming (bbox fixtures)
# ---------------------------------------------------------------------------


def _geo_md(tmp_path, crops):
    links = "\n\n".join(f"![FIGURE_CROP 2:{i}]({c})" for i, c in enumerate(crops))
    return (
        "<page_number>2</page_number>\n\n"
        f"{links}\n\n"
        "FIGURE 1. First caption.\n\nFIGURE 2. Second caption.\n\n"
    )


def test_enrich_geometric_pairing_beats_list_order(tmp_path):
    # crops listed in the REVERSE of their vertical order: positional pairing would
    # cross-label them; geometry (each caption sits right under its figure) cannot
    lower = tmp_path / "page_2_fig_0.png"  # y 200-295, belongs to FIGURE 2 (cap at 300)
    upper = tmp_path / "page_2_fig_1.png"  # y 0-95, belongs to FIGURE 1 (cap at 100)
    lower.write_bytes(b"lower")
    upper.write_bytes(b"upper")
    parsed = ParseResult(
        _geo_md(tmp_path, [lower, upper]),
        figure_crops={
            2: [CropRegion(lower, (0, 200, 100, 295)), CropRegion(upper, (0, 0, 100, 95))]
        },
        figure_captions={
            2: [
                CaptionRegion("FIGURE 1. First caption.", (0, 100, 100, 110)),
                CaptionRegion("FIGURE 2. Second caption.", (0, 300, 100, 310)),
            ]
        },
    )
    calls = {}

    def fake(crops, number, caption, context, cfg):
        calls[number] = [c.read_bytes() for c in crops]
        return

    enrich_markdown(parsed, describe=fake)
    assert calls["1"] == [b"upper"] and calls["2"] == [b"lower"]
    assert (tmp_path / "fig_1.png").read_bytes() == b"upper"
    assert (tmp_path / "fig_2.png").read_bytes() == b"lower"


def test_enrich_multi_panel_crops_share_one_caption_and_call(tmp_path):
    panel_a = tmp_path / "page_2_fig_0.png"
    panel_b = tmp_path / "page_2_fig_1.png"
    panel_a.write_bytes(b"a")
    panel_b.write_bytes(b"b")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({panel_a})\n\n![FIGURE_CROP 2:1]({panel_b})\n\n"
        "FIGURE 7. Two panels, one figure.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={
            2: [CropRegion(panel_a, (0, 0, 45, 90)), CropRegion(panel_b, (55, 0, 100, 90))]
        },
        figure_captions={
            2: [CaptionRegion("FIGURE 7. Two panels, one figure.", (0, 100, 100, 110))]
        },
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((number, [c.name for c in crops]))
        return _desc()

    out = enrich_markdown(parsed, describe=fake)
    assert calls == [("7", ["fig_7_1.png", "fig_7_2.png"])]  # ONE call, both panels
    assert f"![FIGURE 7]({tmp_path / 'fig_7_1.png'})" in out
    assert f"![FIGURE 7]({tmp_path / 'fig_7_2.png'})" in out


def test_enrich_crop_with_no_x_overlap_is_left_alone(tmp_path):
    # a decorative banner in the margin: overlaps no caption horizontally -> no
    # rename, no describe call, placeholder untouched
    figure = tmp_path / "page_2_fig_0.png"
    banner = tmp_path / "page_2_fig_1.png"
    figure.write_bytes(b"f")
    banner.write_bytes(b"b")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({figure})\n\n![FIGURE_CROP 2:1]({banner})\n\n"
        "FIGURE 3. The real figure.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={
            2: [CropRegion(figure, (0, 0, 100, 90)), CropRegion(banner, (400, 0, 500, 20))]
        },
        figure_captions={2: [CaptionRegion("FIGURE 3. The real figure.", (0, 100, 100, 110))]},
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append([c.name for c in crops])
        return _desc()

    out = enrich_markdown(parsed, describe=fake)
    assert calls == [["fig_3.png"]]
    assert banner.exists()  # never renamed
    assert f"![FIGURE_CROP 2:1]({banner})" in out  # placeholder untouched


def test_enrich_renames_crop_and_rewrites_link_positional(tmp_path):
    crop = tmp_path / "page_2_fig_0.png"
    crop.write_bytes(b"fake png bytes")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({crop})\n\n"
        "FIGURE 4.1. A comparison.\n\n"
    )
    captured = {}

    def fake(crops, number, caption, context, cfg):
        captured["crops"] = crops
        return _desc()

    out = enrich_markdown(ParseResult(md, figure_crops={2: [CropRegion(crop)]}), describe=fake)

    renamed = tmp_path / "fig_4.1.png"
    assert renamed.exists() and not crop.exists()
    assert captured["crops"] == [renamed]
    assert f"![FIGURE 4.1]({renamed})" in out
    assert f"![FIGURE_CROP 2:0]({crop})" not in out


def test_enrich_single_caption_page_without_geometry_takes_all_crops(tmp_path):
    # the brunton Fig-3 case: one caption, several crops, no usable bboxes -- the
    # multi-panel assumption wins: every crop on the page belongs to that caption
    crop0 = tmp_path / "page_2_fig_0.png"
    crop1 = tmp_path / "page_2_fig_1.png"
    crop0.write_bytes(b"a")
    crop1.write_bytes(b"b")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({crop0})\n\n![FIGURE_CROP 2:1]({crop1})\n\n"
        "FIGURE 5. Only one caption.\n\n"
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((number, [c.name for c in crops]))
        return

    enrich_markdown(
        ParseResult(md, figure_crops={2: [CropRegion(crop0), CropRegion(crop1)]}),
        describe=fake,
    )
    assert calls == [("5", ["fig_5_1.png", "fig_5_2.png"])]  # one call, both crops


def test_enrich_regex_caption_survives_known_filter_when_number_uncovered(tmp_path):
    # the layout model produced a caption region for FIGURE 1 but MISSED Fig. 3
    # entirely (live: brunton page 5): the strict known filter must not orphan 3
    crop1 = tmp_path / "page_2_fig_0.png"
    crop3 = tmp_path / "page_5_fig_0.png"
    crop1.write_bytes(b"one")
    crop3.write_bytes(b"three")
    md = (
        "<page_number>2</page_number>\n\n"
        f"![FIGURE_CROP 2:0]({crop1})\n\n"
        "FIGURE 1. Known caption.\n\n"
        "<page_number>5</page_number>\n\n"
        f"![FIGURE_CROP 5:0]({crop3})\n\n"
        "Fig. 3. Regex-only caption the layout model missed.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={2: [CropRegion(crop1)], 5: [CropRegion(crop3)]},
        figure_captions={2: [CaptionRegion("FIGURE 1. Known caption.")]},
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append(number)
        return

    enrich_markdown(parsed, describe=fake)
    assert sorted(calls) == ["1", "3"]
    assert (tmp_path / "fig_3.png").exists()


def test_enrich_known_filter_still_strict_for_covered_numbers():
    # FIGURE 7 HAS a caption region: a non-matching line for 7 stays rejected
    md = "<page_number>2</page_number>\n\nFigure 7 shows the general trend in prose form.\n\n"
    parsed = ParseResult(
        md,
        figure_crops={2: [CropRegion(Path("page_2_fig_0.png"))]},
        figure_captions={2: [CaptionRegion("Figure 7. The real caption text.")]},
    )

    def boom(*a):
        raise AssertionError("the mention must not anchor a describe call")

    out = enrich_markdown(parsed, describe=boom)
    assert "Figure description" not in out


def test_enrich_side_caption_pairs_by_y_overlap(tmp_path):
    # the brunton Fig-3 case: caption in the right column BESIDE its panels -- no
    # x-overlap with any crop, but full y-overlap
    panel = tmp_path / "page_5_fig_0.png"
    panel.write_bytes(b"p")
    md = (
        "<page_number>5</page_number>\n\n"
        f"![FIGURE_CROP 5:0]({panel})\n\n"
        "Fig. 3. Side caption text.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={5: [CropRegion(panel, (60, 87, 700, 300))]},
        figure_captions={5: [CaptionRegion("Fig. 3. Side caption text.", (731, 87, 937, 308))]},
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((number, [c.name for c in crops]))
        return

    enrich_markdown(parsed, describe=fake)
    assert calls == [("3", ["fig_3.png"])]


def test_enrich_diagonal_panel_joins_cluster_on_single_caption_page(tmp_path):
    # side caption top-right; panels fill the page; the bottom-LEFT panel is diagonal
    # to the caption but adjacent to its sibling panels -> cluster growth pulls it in
    top = tmp_path / "page_5_fig_0.png"
    bottom_left = tmp_path / "page_5_fig_1.png"
    top.write_bytes(b"t")
    bottom_left.write_bytes(b"bl")
    md = (
        "<page_number>5</page_number>\n\n"
        f"![FIGURE_CROP 5:0]({top})\n\n![FIGURE_CROP 5:1]({bottom_left})\n\n"
        "Fig. 3. Side caption.\n\n"
    )
    parsed = ParseResult(
        md,
        figure_crops={
            5: [
                CropRegion(top, (60, 87, 700, 300)),  # y-overlaps the caption
                CropRegion(bottom_left, (60, 320, 400, 600)),  # diagonal to it
            ]
        },
        figure_captions={5: [CaptionRegion("Fig. 3. Side caption.", (731, 87, 937, 308))]},
    )
    calls = []

    def fake(crops, number, caption, context, cfg):
        calls.append((number, [c.name for c in crops]))
        return

    enrich_markdown(parsed, describe=fake)
    assert calls == [("3", ["fig_3_1.png", "fig_3_2.png"])]
