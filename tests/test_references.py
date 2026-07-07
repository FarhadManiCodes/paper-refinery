"""Tests for bibliography repair (references.py). Fully offline -- fixtures mirror
the live-confirmed hazard cases: brunton's mislabeled/skipped entries, fmech's
page-break split and copyright tail, kalman's out-of-order columns."""

from __future__ import annotations

import logging

from paper_refinery.references import (
    _drop_trailing_boilerplate,
    _merge_split_references,
    _missing_reference_numbers,
    _normalize_layer_text,
    _recover_missing_references,
    _sort_references_by_number,
    _splice_missing_from_layer,
    reclaim_mislabeled_references,
    render_references_markdown,
    repair_references,
)

# ---------------------------------------------------------------------------
# reclaim_mislabeled_references
# ---------------------------------------------------------------------------


def test_reclaim_body_entry_adjacent_to_references():
    # the real brunton-2016 bug: "1. Jordan MI, ..." labeled as plain text right
    # before the detected reference run
    triples = [
        ("body", "ACKNOWLEDGMENTS. We are grateful...", {}),
        ("body", "1. Jordan MI, Mitchell TM (2015) Machine learning. Science.", {}),
        ("reference", "3. Bongard J, Lipson H (2007) Automated reverse engineering.", {}),
    ]
    out = reclaim_mislabeled_references(triples)
    assert out[0][0] == "body"  # ACK untouched
    assert out[1][0] == "reference"
    assert out[2][0] == "reference"


def test_reclaim_chains_through_a_run_of_mislabeled_entries():
    triples = [
        ("body", "1. First entry.", {}),
        ("body", "2. Second entry.", {}),
        ("reference", "3. Third entry.", {}),
    ]
    out = reclaim_mislabeled_references(triples)
    assert [k for k, _, _ in out] == ["reference", "reference", "reference"]


def test_reclaim_requires_adjacency():
    # a numbered body region separated from the reference run stays body text
    triples = [
        ("body", "1. A numbered list item in the body.", {}),
        ("body", "Some interleaving paragraph.", {}),
        ("reference", "2. Real reference.", {}),
    ]
    out = reclaim_mislabeled_references(triples)
    assert out[0][0] == "body"


def test_reclaim_requires_entry_marker():
    # adjacency alone isn't enough -- text without a "[N] "/"N. " start stays body
    triples = [
        ("body", "Concluding remarks about the method.", {}),
        ("reference", "1. Real reference.", {}),
    ]
    out = reclaim_mislabeled_references(triples)
    assert out[0][0] == "body"


def test_reclaim_noop_without_any_reference_region():
    triples = [("body", "1. Numbered list item.", {}), ("body", "2. Another.", {})]
    assert reclaim_mislabeled_references(triples) == triples


# ---------------------------------------------------------------------------
# render_references_markdown
# ---------------------------------------------------------------------------


def testrender_references_markdown_groups_by_page_with_number_prefix():
    refs = [
        {"page": 1, "number": "1", "text": "Smith, J. (2020)."},
        {"page": 1, "number": "2", "text": "Jones, A. (2019)."},
        {"page": 2, "number": None, "text": "Lee, K. (2018)."},
    ]
    md = render_references_markdown(refs)
    assert "<page_number>1</page_number>" in md
    assert "<page_number>2</page_number>" in md
    assert "[1] Smith, J. (2020)." in md
    assert "[2] Jones, A. (2019)." in md
    assert "Lee, K. (2018)." in md and "[None]" not in md
    assert md.index("<page_number>1</page_number>") < md.index("[1]")
    assert md.index("<page_number>2</page_number>") < md.index("Lee, K.")


def testrender_references_markdown_empty_list():
    assert render_references_markdown([]) == ""


# ---------------------------------------------------------------------------
# _drop_trailing_boilerplate
# ---------------------------------------------------------------------------


def _ref(text: str, page: int = 1, number: str | None = None) -> dict:
    return {"page": page, "number": number, "text": text}


def test_drop_trailing_boilerplate_removes_single_trailing_entry():
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref(
            "Conflict of Interest: The authors declare that the research was conducted "
            "in the absence of any commercial or financial relationships."
        ),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == [refs[0]]


def test_drop_trailing_boilerplate_removes_multiple_trailing_entries():
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref("Conflict of Interest: none declared."),
        _ref("Copyright © 2021 Author. This is an open-access article distributed..."),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == [refs[0]]


def test_drop_trailing_boilerplate_stops_at_first_non_matching_entry():
    # a real reference sits between two boilerplate-like entries -- only the
    # *trailing* run should be dropped, not a match buried earlier in the list
    refs = [
        _ref("Smith, J. (2020). Real paper."),
        _ref("Jones, A. On Copyright Law and Academic Publishing. (2019)."),
        _ref("Conflict of Interest: none declared."),
    ]
    cleaned = _drop_trailing_boilerplate(refs)
    assert cleaned == refs[:2]


def test_drop_trailing_boilerplate_caps_how_many_it_checks():
    # more than _MAX_TRAILING_BOILERPLATE_CHECK consecutive "boilerplate-looking"
    # entries -- only the last few are dropped, not the whole list
    refs = [_ref(f"Copyright © 2021. Entry {i}.") for i in range(5)]
    cleaned = _drop_trailing_boilerplate(refs)
    assert len(cleaned) == 2


def test_drop_trailing_boilerplate_leaves_clean_list_untouched():
    refs = [_ref("Smith, J. (2020)."), _ref("Jones, A. (2019).")]
    assert _drop_trailing_boilerplate(refs) == refs


def test_drop_trailing_boilerplate_handles_empty_list():
    assert _drop_trailing_boilerplate([]) == []


# ---------------------------------------------------------------------------
# _merge_split_references
# ---------------------------------------------------------------------------


def test_merge_split_references_glues_page_break_url_split():
    # the fmech case: ref 28 ends page 9 mid-URL; page 10 opens with its tail
    refs = [
        _ref("Quiban, R. (2019). Churning losses. https://journals.sagepub.", page=9),
        _ref("com/home/pij Proc. Inst. Mech. Eng. doi: 10.1177/1350650119858236", page=10),
    ]
    merged = _merge_split_references(refs)
    assert len(merged) == 1
    assert "https://journals.sagepub.com/home/pij" in merged[0]["text"]  # no space in URL
    assert merged[0]["page"] == 9


def test_merge_split_references_joins_word_split_with_space():
    refs = [
        _ref("Quiban, R. (2019). Churning losses of spiral bevel", page=9),
        _ref("gears at high rotational speed. J. Tribol.", page=10),
    ]
    merged = _merge_split_references(refs)
    assert len(merged) == 1
    assert "spiral bevel gears at high" in merged[0]["text"]


def test_merge_split_references_keeps_surname_particle_entry():
    # "van Wijk" legitimately starts an entry -- lowercase alone must not merge it
    refs = [
        _ref("Turner, A. (2013). Two phase CFD modelling.", page=9),
        _ref("van Wijk, J. (2010). Parametric modelling.", page=10),
    ]
    assert len(_merge_split_references(refs)) == 2


def test_merge_split_references_requires_page_change():
    # a lowercase fragment mid-page is NOT the page-break split this rule targets
    refs = [
        _ref("Turner, A. (2013). Two phase CFD modelling.", page=9),
        _ref("com/home/pij continuation-looking text", page=9),
    ]
    assert len(_merge_split_references(refs)) == 2


def test_merge_split_references_keeps_numbered_entries():
    refs = [
        _ref("[28] Quiban, R. Churning losses.", page=9),
        _ref("[29] Saurer, J. Instationaren.", page=10),
        _ref("Webb, T. (2010).", page=10, number="30"),
    ]
    assert len(_merge_split_references(refs)) == 3


def test_merge_split_references_chains_fragments():
    # two continuation regions on the later page fold into the same entry in order
    refs = [
        _ref("Quiban, R. (2019). Churning losses of spiral", page=9),
        _ref("bevel gears at high", page=10),
        _ref("rotational speed. J. Tribol.", page=10),
        _ref("Saurer, J. (2000). Instationaren.", page=10),
    ]
    merged = _merge_split_references(refs)
    assert len(merged) == 2
    assert merged[0]["text"] == (
        "Quiban, R. (2019). Churning losses of spiral bevel gears at high "
        "rotational speed. J. Tribol."
    )


# ---------------------------------------------------------------------------
# _sort_references_by_number
# ---------------------------------------------------------------------------


def test_sort_references_by_number_fixes_scrambled_order():
    # e.g. kalman-1960.pdf's two-column bibliography: region index order doesn't
    # match reading order, but every entry has a clean numeric marker
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number="1"), _ref("Bode", number="3")]
    sorted_refs = _sort_references_by_number(refs)
    assert [r["number"] for r in sorted_refs] == ["1", "2", "3"]


def test_sort_references_by_number_leaves_unnumbered_style_untouched():
    # author-year style (e.g. fmech-07-655266.pdf): no numbers at all
    refs = [_ref("Bianchini"), _ref("Boness"), _ref("Burberi")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_partial_numbering_untouched():
    # one entry missing its number (e.g. an unpaired reference_content) -- don't
    # guess at a partial sort, leave detected order as-is
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number=None), _ref("Bode", number="3")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_non_numeric_marker_untouched():
    # a garbled/non-integer marker on at least one entry -- bail out entirely
    refs = [_ref("Wiener", number="2"), _ref("Zadeh", number="1a")]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_duplicate_keys_untouched():
    # two entries garbled to the same number -> keys can't be trusted at all
    refs = [
        {"number": "2", "text": "b"},
        {"number": "1", "text": "a"},
        {"number": "2", "text": "c"},
    ]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_leaves_outlier_key_untouched():
    # an unnumbered entry whose text starts with a year would sort to the end --
    # the gap it leaves in the contiguous run is the tell
    refs = [
        {"number": "1", "text": "a"},
        {"number": None, "text": "2019 IEEE Conference on Things. Proceedings."},
        {"number": "2", "text": "b"},
    ]
    assert _sort_references_by_number(refs) == refs


def test_sort_references_by_number_handles_empty_list():
    assert _sort_references_by_number([]) == []


def test_sort_references_by_number_falls_back_to_leading_number_in_text():
    # the real kalman-1960.pdf bug: PP-DocLayout-V3 never produces a separate
    # reference_number region for this paper at all -- the marker is just the leading
    # digits of the OCR'd text blob itself, so `number` is None on every entry
    refs = [
        _ref("9 A. B. Lees, Interpolation and Extrapolation..."),
        _ref("8 G. Franklin, The Optimum Synthesis..."),
        _ref("11 M. Shinbrot, Optimization of Time-Varying..."),
        _ref("10 R. C. Davis, On the Theory of Prediction..."),
    ]
    sorted_refs = _sort_references_by_number(refs)
    # numeric, not lexicographic: "10"/"11" sort after "8"/"9"
    assert [r["text"][:2].strip() for r in sorted_refs] == ["8", "9", "10", "11"]


def test_sort_references_by_number_falls_back_to_bracketed_number_in_text():
    refs = [_ref("[2] Second entry"), _ref("[1] First entry")]
    sorted_refs = _sort_references_by_number(refs)
    assert [r["text"] for r in sorted_refs] == ["[1] First entry", "[2] Second entry"]


def test_sort_references_by_number_mixed_region_and_text_number_sources():
    # one entry has a real paired `number`, another only has it embedded in the text --
    # both should still contribute a usable sort key
    refs = [_ref("Second entry", number="2"), _ref("1 First entry")]
    sorted_refs = _sort_references_by_number(refs)
    assert sorted_refs[0]["text"] == "1 First entry"


# ---------------------------------------------------------------------------
# text-layer recovery of skipped bibliography entries
# ---------------------------------------------------------------------------


def test_splice_missing_recovers_entry_from_text_layer(caplog):
    # the brunton case: entry 2 printed in the PDF but no layout region for it
    refs = [
        _ref("1. Jordan MI, Mitchell TM (2015) Machine learning. Science 349:255-260.", page=6),
        _ref("3. Bongard J, Lipson H (2007) Automated reverse engineering. PNAS.", page=6),
    ]
    layer = (
        "body text before the bibliography\n"
        "1. Jordan MI, Mitchell TM (2015) Machine learning. Science\n349:255-260.\n"
        "2. Marx V (2013) Biology: The big challenges of big data. Nature 498:255-260.\n"
        "3. Bongard J, Lipson H (2007) Automated reverse engineering. PNAS.\n"
    )
    with caplog.at_level(logging.WARNING):
        out = _splice_missing_from_layer(refs, layer, [2])
    assert "recovered missing reference 2" in caplog.text
    assert [r["text"][:10] for r in out] == ["1. Jordan ", "2. Marx V ", "3. Bongard"]
    assert out[1]["number"] is None  # marker lives in the text, like its OCR'd siblings
    assert "big data" in out[1]["text"]
    assert out[1]["page"] == 6  # carried from its predecessor


def test_splice_missing_skips_ambiguous_marker():
    # a "Vol. 2." lookalike inside the neighbor makes the marker non-unique -> no guess
    refs = [
        _ref("1. First entry about Vol. 2. things in detail.", page=1),
        _ref("3. Third entry text here.", page=1),
    ]
    layer = (
        "1. First entry about Vol. 2. things in detail. "
        "2. Second entry that must not be guessed at. "
        "3. Third entry text here."
    )
    assert len(_splice_missing_from_layer(refs, layer, [2])) == 2


def test_splice_missing_skips_when_neighbor_prefix_not_in_layer():
    # heavily garbled OCR text can't anchor into the layer -> leave the gap alone
    refs = [_ref("1. T0tally g@rbled 0CR text.", page=1), _ref("3. Third.", page=1)]
    layer = "1. First. 2. Second. 3. Third."
    assert len(_splice_missing_from_layer(refs, layer, [2])) == 2


def test_recover_missing_references_guards(tmp_path):
    ay = [_ref("Smith, J. (2020). A title.")]
    assert _recover_missing_references(ay, tmp_path / "x.pdf") == ay  # unnumbered style
    numbered = [_ref("1. A. Author, paper one."), _ref("2. B. Author, paper two.")]
    assert _recover_missing_references(numbered, tmp_path / "x.pdf") == numbered  # no gap
    gapped = [_ref("1. A. Author, paper one."), _ref("3. C. Author, paper three.")]
    # gap present but the PDF is unreadable -> unchanged, gap warning handles it
    assert _recover_missing_references(gapped, tmp_path / "missing.pdf") == gapped
    assert _recover_missing_references(gapped, None) == gapped


def test_normalize_layer_text_rejoins_hyphenated_linebreaks():
    assert _normalize_layer_text("dynam-\nical\nsystems") == "dynamical systems"


def test_missing_reference_numbers():
    assert _missing_reference_numbers([_ref("1. A."), _ref("3. C.")]) == [2]
    assert _missing_reference_numbers([_ref("1. A."), _ref("2. B.")]) == []
    assert _missing_reference_numbers([_ref("Smith, J. (2020). A title.")]) == []  # unclean key
    assert _missing_reference_numbers([]) == []
    # duplicated keys make the expected-set math meaningless -> report nothing
    assert _missing_reference_numbers([_ref("1. A."), _ref("1. B."), _ref("3. C.")]) == []


# ---------------------------------------------------------------------------
# repair_references (the one fixed sequence parse.py calls)
# ---------------------------------------------------------------------------


def test_repair_references_runs_merge_then_sort():
    # a page-break fragment is rejoined BEFORE the number sort, so the rejoined
    # bibliography still counts as a clean contiguous run and gets ordered
    refs = [
        _ref("2. Second entry text.", page=1),
        _ref("1. First entry ends https://journals.example.", page=1),
        _ref("org/x doi: 10.1/abc", page=2),
    ]
    out = repair_references(refs)
    assert [r["text"][:2] for r in out] == ["1.", "2."]
    assert "https://journals.example.org/x doi: 10.1/abc" in out[0]["text"]
