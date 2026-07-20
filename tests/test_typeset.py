"""typeset.py: markdown repairs pandoc/xelatex need to survive OCR'd math and headings,
plus one real end-to-end compile (skipped when pandoc/xelatex aren't on PATH)."""

import shutil
from pathlib import Path

import pytest

from paper_refinery.config import CitationConfig, TypesetConfig
from paper_refinery.typeset import (
    _apply_heading_levels,
    _convert_footnotes_to_pandoc_notes,
    _find_headings,
    _fix_double_scripts,
    _font_available,
    _llm_heading_levels,
    _normalize_math_dollars,
    _prepare_pandoc_input,
    _promote_orphaned_chapter_subsections,
    classify_chapter_level,
    classify_real_headings,
    heading_candidates,
    render_pdf,
)

_HAS_PANDOC_TOOLCHAIN = shutil.which("pandoc") is not None and shutil.which("xelatex") is not None
_HAS_FC_MATCH = shutil.which("fc-match") is not None


@pytest.mark.skipif(not _HAS_FC_MATCH, reason="needs fc-match on PATH")
def test_font_available_rejects_a_fake_font_name():
    assert _font_available("Definitely Not A Real Font XYZ123") is False


def test_font_available_fails_open_when_fc_match_is_missing(monkeypatch):
    import subprocess

    def _raise(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _font_available("anything") is True


def test_normalize_math_dollars_strips_inner_padding():
    # pandoc's dollar-math parser only fires when no whitespace touches $/$$; refinery's
    # OCR convention always pads with a space
    assert _normalize_math_dollars("the value $ x = 1 $ here") == "the value $x = 1$ here"
    assert _normalize_math_dollars("$$\nx = 1\n$$") == "$$x = 1$$"


def test_normalize_math_dollars_leaves_unpadded_math_alone():
    assert _normalize_math_dollars("$x=1$") == "$x=1$"


def test_fix_double_scripts_merges_stacked_superscript():
    # "Double superscript" is a hard TeX error even in nonstopmode, unlike most others
    assert _fix_double_scripts(r"y ^ {\prime} ^ {2}") == r"y ^{\prime 2}"


def test_fix_double_scripts_merges_stacked_subscript():
    assert _fix_double_scripts(r"x _ {i} _ {0}") == r"x _{i 0}"


def test_fix_double_scripts_leaves_single_script_alone():
    assert _fix_double_scripts(r"y^{\prime}") == r"y^{\prime}"


def test_apply_heading_levels_demotes_immediate_repeat():
    md = "## PREFACE\n\nbody text\n\n## PREFACE\n\nmore text\n"
    out = _apply_heading_levels(md)
    assert out.count("## PREFACE") == 1
    assert "PREFACE\n\nmore text" in out  # demoted heading keeps its text


def test_apply_heading_levels_keeps_distant_same_titled_sections():
    md = "## EXERCISES\n\na\n\n## CHAPTER 4\n\nb\n\n## EXERCISES\n\nc\n"
    out = _apply_heading_levels(md)
    assert out.count("## EXERCISES") == 2  # a different heading intervened


def test_find_headings_returns_line_index_and_text_in_order():
    md = "intro\n\n## First\n\nbody\n\n## Second\n\nmore\n"
    assert _find_headings(md) == [(2, "First"), (6, "Second")]


def test_apply_heading_levels_not_heading_is_positional_not_textual():
    # a heading like "EXERCISES" can legitimately repeat many times as real section
    # starts -- demoting by matched text would wrongly collapse every occurrence, so the
    # levels dict must target one specific occurrence (by position), not the text
    md = "## EXERCISES\n\na\n\n## CHAPTER 4\n\nb\n\n## EXERCISES\n\nc\n"
    out = _apply_heading_levels(md, levels={0: "not_heading"})
    assert out.count("## EXERCISES") == 1
    assert "## CHAPTER 4" in out
    assert "EXERCISES\n\na" in out  # position 0 demoted, kept as plain text
    assert "## EXERCISES\n\nc" in out  # position 2 (the later, distinct occurrence) kept


def test_apply_heading_levels_promotes_chapter_to_h1():
    md = "## CHAPTER 5\n\nsomething\n"
    out = _apply_heading_levels(md, levels={0: "chapter"})
    assert out.startswith("# CHAPTER 5")


def test_apply_heading_levels_merges_adjacent_chapter_headings():
    md = "## CHAPTER 5\n\n## GEOMETRICAL OPTICS\n\nbody\n"
    out = _apply_heading_levels(md, levels={0: "chapter", 1: "chapter"})
    assert out.count("#") == 1  # only one heading line left (merged)
    assert "# CHAPTER 5: GEOMETRICAL OPTICS" in out
    assert "## GEOMETRICAL OPTICS" not in out  # the second line was merged away, not kept


def test_apply_heading_levels_unclassified_position_defaults_to_section():
    md = "## Heading\n\nbody\n"
    out = _apply_heading_levels(md, levels={})
    assert "## Heading" in out


def test_apply_heading_levels_unrecognized_level_value_defaults_to_section():
    md = "## Heading\n\nbody\n"
    out = _apply_heading_levels(md, levels={0: "bogus_level"})
    assert "## Heading" in out


# ---------------------------------------------------------------------------
# _promote_orphaned_chapter_subsections
# ---------------------------------------------------------------------------


def test_promote_orphaned_chapter_subsections_promotes_missing_marker_chapter():
    # confirmed live (Weinstock "Calculus of Variations"): a chapter with no "CHAPTER N"
    # marker and no preceding "EXERCISES" -- classification alone left its title at
    # "section"; subsection numbering resetting to "9-1." is the deterministic tell
    texts = ["SEVERAL INDEPENDENT VARIABLES", "9-1. Extremization of a Multiple Integral"]
    resolved = ["section", "section"]
    out = _promote_orphaned_chapter_subsections(texts, resolved)
    assert out == ["chapter", "section"]


def test_promote_orphaned_chapter_subsections_noop_when_already_chapter_level():
    # normal case: a properly marked/merged chapter heading already precedes "N-1." --
    # nothing to promote
    texts = ["CHAPTER 9: SEVERAL INDEPENDENT VARIABLES", "9-1. Extremization"]
    resolved = ["chapter", "section"]
    out = _promote_orphaned_chapter_subsections(texts, resolved)
    assert out == ["chapter", "section"]


def test_promote_orphaned_chapter_subsections_never_promotes_exercises():
    # EXERCISES always precedes the next chapter's start (marked or not) -- promoting it
    # would misfire on every ordinary chapter transition, marked or not
    texts = ["EXERCISES", "9-1. Extremization"]
    resolved = ["section", "section"]
    out = _promote_orphaned_chapter_subsections(texts, resolved)
    assert out == ["section", "section"]


def test_promote_orphaned_chapter_subsections_skips_demoted_headings():
    # a not_heading entry between two real headings must not become the promotion target
    texts = ["Real Title", "TOC Pollution Line", "9-1. Extremization"]
    resolved = ["section", "not_heading", "section"]
    out = _promote_orphaned_chapter_subsections(texts, resolved)
    assert out == ["chapter", "not_heading", "section"]


def test_promote_orphaned_chapter_subsections_handles_first_position():
    # "N-1." as the very first heading in the document has no preceding heading to
    # promote -- must not crash
    texts = ["9-1. Extremization"]
    resolved = ["section"]
    out = _promote_orphaned_chapter_subsections(texts, resolved)
    assert out == ["section"]


def test_heading_candidates_pairs_preceding_text_and_following_context():
    md = "## First\n\nsome prose here\n\n## Second\n\nmore prose\n"
    candidates = heading_candidates(md)
    assert candidates == [
        ("", "First", "some prose here"),
        ("First", "Second", "more prose"),
    ]


def test_heading_candidates_truncates_long_context():
    md = "## Heading\n\n" + ("word " * 200) + "\n"
    (_preceding, _text, context) = heading_candidates(md, max_context_chars=50)[0]
    assert len(context) == 50


def test_heading_candidates_empty_for_no_headings():
    assert heading_candidates("just some text\n") == []


def test_prepare_pandoc_input_strips_page_markers():
    md = "<page_number>1</page_number>\n\nbody\n"
    assert "page_number" not in _prepare_pandoc_input(md)


def test_prepare_pandoc_input_blanks_figure_crop_alt_text():
    md = "![FIGURE_CROP 3:0](figures/page_3_fig_0.png)"
    out = _prepare_pandoc_input(md)
    assert out == "![](figures/page_3_fig_0.png)"


def test_prepare_pandoc_input_blanks_enrich_renamed_figure_alt_text():
    # enrich.py renames "FIGURE_CROP page:idx" to "FIGURE number" once crops are paired to
    # captions -- both are refinery's own internal placeholder, never a real caption
    md = "![FIGURE 3.1](figures/fig_3.1_1.png)"
    out = _prepare_pandoc_input(md)
    assert out == "![](figures/fig_3.1_1.png)"


def test_prepare_pandoc_input_leaves_real_alt_text_alone():
    md = "![a real caption](figures/x.png)"
    assert _prepare_pandoc_input(md) == md


def test_prepare_pandoc_input_applies_heading_levels():
    md = "## First\n\na\n\n## Second\n\nb\n"
    out = _prepare_pandoc_input(md, heading_levels={1: "not_heading"})
    assert "## First" in out
    assert "## Second" not in out
    assert "Second\n\nb" in out


# ---------------------------------------------------------------------------
# _convert_footnotes_to_pandoc_notes
# ---------------------------------------------------------------------------


def test_convert_footnotes_attaches_reference_to_preceding_paragraph():
    md = "Some body text.\n\n> **Footnote:** See Bliss (1) for details.\n\nMore text."
    out = _convert_footnotes_to_pandoc_notes(md)
    assert out == "Some body text.[^1]\n\n[^1]: See Bliss (1) for details.\n\nMore text."


def test_convert_footnotes_numbers_sequentially():
    md = (
        "First paragraph.\n\n"
        "> **Footnote:** first note.\n\n"
        "Second paragraph.\n\n"
        "> **Footnote:** second note."
    )
    out = _convert_footnotes_to_pandoc_notes(md)
    assert "First paragraph.[^1]" in out
    assert "[^1]: first note." in out
    assert "Second paragraph.[^2]" in out
    assert "[^2]: second note." in out


def test_convert_footnotes_leaves_orphaned_first_block_unreferenced():
    # a footnote as the very first block has no preceding paragraph to anchor on --
    # pandoc drops an unreferenced footnote *definition*, so leave it as a plain quoted
    # paragraph instead (still visible, just not converted) rather than silently losing it
    md = "> **Footnote:** an orphaned note.\n\nBody text follows."
    out = _convert_footnotes_to_pandoc_notes(md)
    assert out == md


def test_convert_footnotes_leaves_non_footnote_text_alone():
    md = "Just ordinary text.\n\n> A real blockquote, not a footnote."
    assert _convert_footnotes_to_pandoc_notes(md) == md


def test_prepare_pandoc_input_converts_footnotes():
    md = "Body text.\n\n> **Footnote:** a note."
    out = _prepare_pandoc_input(md)
    assert "[^1]" in out
    assert "> **Footnote:**" not in out


# ---------------------------------------------------------------------------
# classify_real_headings / classify_chapter_level (mocked; the real Gemini call is
# live-only). Deliberately separate calls, not one combined pass -- see
# heading_candidates's docstring: giving "preceded by" context during the real/not-real
# judgment confirmed live to trick the model into treating a whole polluted run of
# table-of-contents lines as real chapters, since each is "preceded by" something that
# itself looks like a chapter title.
# ---------------------------------------------------------------------------


def _fake_client(parsed):
    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return type("R", (), {"parsed": parsed})()

    return FakeClient()


def test_classify_real_headings_returns_empty_for_no_candidates():
    assert classify_real_headings([], CitationConfig()) == []


def test_classify_real_headings_matches_by_declared_index():
    from paper_refinery.typeset import _RealHeadingJudgment

    parsed = [
        _RealHeadingJudgment(index=1, is_real_heading=True),
        _RealHeadingJudgment(index=2, is_real_heading=False),
    ]
    result = classify_real_headings(
        [("Real Section", "lots of prose"), ("Contents Line", "another heading")],
        CitationConfig(),
        client=_fake_client(parsed),
    )
    assert result == [True, False]


def test_classify_real_headings_fails_open_on_unusable_response():
    result = classify_real_headings(
        [("A", "ctx a"), ("B", "ctx b")], CitationConfig(), client=_fake_client([None])
    )
    assert result == [True, True]  # never silently demotes on a bad response


def test_classify_real_headings_survives_extra_or_out_of_range_indices():
    # confirmed live (Weinstock "Calculus of Variations"): the model returned 150
    # judgments for 147 candidates -- a few stray/duplicate/out-of-range indices must
    # only cost those specific candidates their classification, not the whole batch
    from paper_refinery.typeset import _RealHeadingJudgment

    parsed = [
        _RealHeadingJudgment(index=1, is_real_heading=False),
        _RealHeadingJudgment(index=1, is_real_heading=True),  # duplicate index 1, last wins
        _RealHeadingJudgment(index=2, is_real_heading=False),  # clean, unambiguous match
        _RealHeadingJudgment(index=99, is_real_heading=False),  # out of range, ignored
        # index 3 never shows up at all -> defaults to "keep"
    ]
    result = classify_real_headings(
        [("A", "ctx a"), ("B", "ctx b"), ("C", "ctx c")],
        CitationConfig(),
        client=_fake_client(parsed),
    )
    assert result == [True, False, True]


def test_classify_chapter_level_returns_empty_for_no_candidates():
    assert classify_chapter_level([], CitationConfig()) == []


def test_classify_chapter_level_matches_by_declared_index():
    from paper_refinery.typeset import _ChapterLevelJudgment

    parsed = [
        _ChapterLevelJudgment(index=1, level="chapter"),
        _ChapterLevelJudgment(index=2, level="section"),
    ]
    result = classify_chapter_level(
        [("", "CHAPTER 5", "prose"), ("CHAPTER 5", "5-1. A Section", "more prose")],
        CitationConfig(),
        client=_fake_client(parsed),
    )
    assert result == ["chapter", "section"]


def test_classify_chapter_level_fails_open_on_unusable_response():
    result = classify_chapter_level(
        [("", "A", "ctx a")], CitationConfig(), client=_fake_client([None])
    )
    assert result == ["section"]


def test_classify_chapter_level_rejects_unrecognized_value():
    from paper_refinery.typeset import _ChapterLevelJudgment

    parsed = [_ChapterLevelJudgment(index=1, level="bogus_value")]
    result = classify_chapter_level(
        [("", "A", "ctx a")], CitationConfig(), client=_fake_client(parsed)
    )
    assert result == ["section"]  # unrecognized value treated as unmatched, not trusted


# ---------------------------------------------------------------------------
# _llm_heading_levels
# ---------------------------------------------------------------------------


def test_llm_heading_levels_returns_empty_when_disabled():
    md = "## First\n\na\n\n## Second\n\nb\n"
    assert _llm_heading_levels(md, TypesetConfig(clean_toc_with_llm=False), None) == {}


def test_llm_heading_levels_degrades_gracefully_on_failure(monkeypatch):
    # e.g. missing GOOGLE_API_KEY -- must never abort the whole typeset job
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    md = "## First\n\na\n\n## Second\n\nb\n"
    result = _llm_heading_levels(md, TypesetConfig(clean_toc_with_llm=True), None)
    assert result == {}


def test_llm_heading_levels_only_runs_chapter_pass_on_confirmed_real_headings(monkeypatch):
    # the chapter-level pass must never see a heading the real-heading pass rejected --
    # that's the whole point of splitting them into two sequential calls
    import paper_refinery.typeset as typeset_mod

    md = "## Real One\n\nprose\n\n## Contents Line\n\nmore prose\n"

    def fake_classify_real_headings(candidates, cfg, client=None):
        return [True, False]

    seen_chapter_pass_candidates = []

    def fake_classify_chapter_level(candidates, cfg, client=None):
        seen_chapter_pass_candidates.extend(candidates)
        return ["chapter"] * len(candidates)

    monkeypatch.setattr(typeset_mod, "classify_real_headings", fake_classify_real_headings)
    monkeypatch.setattr(typeset_mod, "classify_chapter_level", fake_classify_chapter_level)
    monkeypatch.setattr(typeset_mod, "make_client", lambda cfg: object())

    result = _llm_heading_levels(md, TypesetConfig(clean_toc_with_llm=True), None)
    assert result == {0: "chapter", 1: "not_heading"}
    assert len(seen_chapter_pass_candidates) == 1
    assert seen_chapter_pass_candidates[0][1] == "Real One"


def test_llm_heading_levels_applies_orphaned_chapter_promotion(monkeypatch):
    # end-to-end: even when both classification passes leave a missing-marker chapter at
    # "section", the deterministic promotion pass should still recover it
    import paper_refinery.typeset as typeset_mod

    md = "## SEVERAL INDEPENDENT VARIABLES\n\nprose\n\n## 9-1. Extremization\n\nmore\n"

    monkeypatch.setattr(
        typeset_mod, "classify_real_headings", lambda candidates, cfg, client=None: [True, True]
    )
    monkeypatch.setattr(
        typeset_mod,
        "classify_chapter_level",
        lambda candidates, cfg, client=None: ["section", "section"],
    )
    monkeypatch.setattr(typeset_mod, "make_client", lambda cfg: object())

    result = _llm_heading_levels(md, TypesetConfig(clean_toc_with_llm=True), None)
    assert result == {0: "chapter", 1: "section"}


_DOC_MD = (
    "# A Title\n\n"
    "## A Section\n\n"
    "Some text with math $ x^{2} + 1 = 0 $ in it.\n\n"
    "$$\nI = \\int_ {0} ^ {1} f(x)\\,dx\n$$\n"
)


@pytest.mark.skipif(not _HAS_PANDOC_TOOLCHAIN, reason="needs pandoc + xelatex on PATH")
def test_render_pdf_end_to_end(tmp_path):
    md_path = tmp_path / "doc.md"
    md_path.write_text(_DOC_MD)
    out_path = tmp_path / "doc.pdf"
    errors = render_pdf(md_path, out_path, TypesetConfig(), title="A Title")
    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert errors == []
    # no stray intermediate files left behind in the work dir
    leftovers = [p.name for p in tmp_path.iterdir() if p not in (md_path, out_path)]
    assert leftovers == []


@pytest.mark.skipif(not _HAS_PANDOC_TOOLCHAIN, reason="needs pandoc + xelatex on PATH")
def test_render_pdf_survives_a_cross_filesystem_move(tmp_path, monkeypatch):
    # reproduces a real bug: work_dir and --out are frequently on different filesystems
    # (e.g. work_dir under the project tree, --out in /tmp), and os.rename raises EXDEV
    # across a filesystem boundary. Force that path deterministically regardless of the
    # test machine's actual mount layout; monkeypatch restores os.rename automatically.
    import os

    def _rename_raises_exdev(src, dst):
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(os, "rename", _rename_raises_exdev)
    md_path = tmp_path / "doc.md"
    md_path.write_text(_DOC_MD)
    out_path = tmp_path / "doc.pdf"
    render_pdf(md_path, out_path, TypesetConfig(), title="A Title")
    assert out_path.exists()


@pytest.mark.skipif(not _HAS_PANDOC_TOOLCHAIN, reason="needs pandoc + xelatex on PATH")
def test_render_pdf_with_relative_paths(tmp_path, monkeypatch):
    # md_path/out_path built from a *relative* input reproduces a real bug: combining
    # cwd=work_dir with argv paths built from the original (pre-chdir) relative path
    # doubled work_dir into the effective lookup path pandoc used, so it couldn't find its
    # own input file.
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "doc.md").write_text(_DOC_MD)
    monkeypatch.chdir(tmp_path)
    errors = render_pdf(Path("sub/doc.md"), Path("sub/doc.pdf"), TypesetConfig(), title="A Title")
    assert (tmp_path / "sub" / "doc.pdf").exists()
    assert errors == []
