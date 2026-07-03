"""Tests for deterministic in-text citation linking. Fully offline -- fixtures mirror
the live-confirmed hazard cases (hyco's [0,1] interval, brunton's paren/equation
collision, fmech's author-year style)."""

from __future__ import annotations

from paper_refinery.citation_linking import (
    Marker,
    infer_marker_style,
    link_citations,
    make_citekey,
    rewrite_markers,
)


def _numbered(n: int, style: str = "[{}]") -> list[dict]:
    return [{"citation_key": style.format(i + 1), "title": f"T{i + 1}"} for i in range(n)]


def _author_year(entries: list[tuple[str, int]]) -> list[dict]:
    return [
        {"citation_key": None, "authors": [{"family": fam}], "year": yr, "title": fam}
        for fam, yr in entries
    ]


# ---------------------------------------------------------------------------
# infer_marker_style
# ---------------------------------------------------------------------------


def test_style_numbered_regardless_of_marker_form():
    # bibliography marker form does NOT predict the body's (kalman: bare-numbered
    # bibliography, bracket-citing body) -- all numeric forms collapse to "numbered"
    assert infer_marker_style(_numbered(5)) == "numbered"
    assert infer_marker_style(_numbered(5, style="{}.")) == "numbered"
    assert infer_marker_style(_numbered(5, style="({})")) == "numbered"


def test_style_author_year_when_keys_absent():
    assert infer_marker_style(_author_year([("Smith", 2020)])) == "author-year"


def test_style_majority_wins_over_stray_key():
    entries = _numbered(4) + [{"citation_key": None, "title": "T5"}]
    assert infer_marker_style(entries) == "numbered"


def test_style_empty_list():
    assert infer_marker_style([]) == "author-year"


# ---------------------------------------------------------------------------
# bracket style
# ---------------------------------------------------------------------------


def test_bracket_single_and_list_and_range():
    md = "As shown [1], later [7,8] and finally [2-4] end."
    res = link_citations(md, _numbered(8))
    got = [(m.text, m.ref_indices) for m in res.markers]
    assert got == [("[1]", [0]), ("[7,8]", [6, 7]), ("[2-4]", [1, 2, 3])]


def test_bracket_rejects_out_of_range_math_interval():
    # the real hyco trap: a [0,1] unit square next to genuine numbered citations
    md = "the unit square [0,1] is used, following [3]."
    res = link_citations(md, _numbered(5))
    assert [m.text for m in res.markers] == ["[3]"]
    assert "[0,1]" in res.ambiguous


def test_bracket_skips_markers_inside_math():
    md = "matrix $A[1]$ here, but citation [1] there. Also $$ x[2] $$ display."
    res = link_citations(md, _numbered(2))
    assert len(res.markers) == 1
    assert md[res.markers[0].start - 9 : res.markers[0].end] == "citation [1]"


def test_bracket_rejects_implausible_range():
    md = "pages [100-400] are irrelevant; [1] is real."
    res = link_citations(md, _numbered(3))
    assert [m.text for m in res.markers] == ["[1]"]


def test_bracket_uncited_lists_never_linked_refs():
    res = link_citations("only [1] and [3].", _numbered(4))
    assert res.uncited == [1, 3]  # refs 2 and 4 never cited


def test_bracket_number_map_respects_gaps():
    # bibliography missing entry 2 (a real brunton-style gap): keys are [1],[3],
    # so position no longer equals number
    extracted = [{"citation_key": "[1]"}, {"citation_key": "[3]"}]
    res = link_citations("see [3] and [1]; also [2] which is missing.", extracted)
    assert [(m.text, m.ref_indices) for m in res.markers] == [("[3]", [1]), ("[1]", [0])]
    assert "[2]" in res.ambiguous


def test_numbered_body_form_decided_by_scan_not_bibliography():
    # kalman's case: bare-numbered bibliography keys, bracket-citing body
    extracted = [{"citation_key": f"{i + 1}"} for i in range(5)]
    res = link_citations("as shown [2] and later [5].", extracted)
    assert res.style == "numbered-bracket"
    assert [m.text for m in res.markers] == ["[2]", "[5]"]


def test_numbered_tie_goes_to_bracket():
    extracted = _numbered(5)
    res = link_citations("no markers at all here.", extracted)
    assert res.style == "numbered-bracket" and res.markers == []


def test_bracket_evidence_decisive_over_higher_paren_count():
    # the real kalman failure: genuine [N] citations must not be outvoted by paren
    # false positives (numbered list items, in-range equation numbers)
    md = (
        "Real citations [1] and [2] and [3] appear. "
        "The results are: (1) first point (2) second point (3) third point "
        "(4) fourth point (5) fifth point."
    )
    res = link_citations(md, _numbered(5))
    assert res.style == "numbered-bracket"
    assert [m.text for m in res.markers] == ["[1]", "[2]", "[3]"]  # unknown number -> rejected, not mislinked


# ---------------------------------------------------------------------------
# paren-number style (brunton/PNAS)
# ---------------------------------------------------------------------------


def test_paren_citations_linked():
    md = "by Bongard and Lipson (3) and Schmidt and Lipson (4)."
    res = link_citations(md, _numbered(10, style="{}."))
    assert [(m.text, m.ref_indices) for m in res.markers] == [("(3)", [2]), ("(4)", [3])]


def test_paren_rejects_equation_context():
    md = "in the polynomial basis in Eq. (2). But the method (2) is cited."
    res = link_citations(md, _numbered(5, style="{}."))
    assert len(res.markers) == 1
    assert md[res.markers[0].start - 11 : res.markers[0].end] == "the method (2)"
    assert "(2)" in res.ambiguous


def test_paren_eq_context_accepted_above_max_equation_tag():
    # the brunton case: "...equations (44, 45)" is a genuine citation -- the paper's
    # own \tag'd equations only go up to [2] here, so 44/45 can't be equation numbers
    md = (
        "$$ x = y \\tag{[1]} $$\n\n$$ z \\tag{[2]} $$\n\n"
        "simulations of the Navier-Stokes equations (44, 45)."
    )
    res = link_citations(md, _numbered(50, style="{}."))
    assert [m.text for m in res.markers] == ["(44, 45)"]


def test_paren_eq_context_still_rejected_within_tag_range():
    md = "$$ x \\tag{[1]} $$\n\n$$ y \\tag{[6]} $$\n\nsee equations (3, 4), while (5) is cited."
    res = link_citations(md, _numbered(10, style="{}."))
    assert [m.text for m in res.markers] == ["(5)"]
    assert "(3, 4)" in res.ambiguous


def test_paren_eq_context_rejected_when_no_tags_detected():
    # no machine-readable equation numbering -> the relaxation has no basis; stay strict
    md = "as shown in (7), the governing equations (44, 45) are solved."
    res = link_citations(md, _numbered(50, style="{}."))
    assert [m.text for m in res.markers] == ["(7)"]
    assert "(44, 45)" in res.ambiguous


def test_paren_rejects_standalone_equation_tag_paragraph():
    md = "Some text before.\n\n(3)\n\nMore text (3) citing."
    res = link_citations(md, _numbered(5, style="{}."))
    assert len(res.markers) == 1  # only the in-sentence one
    assert res.markers[0].start > md.index("More text")


def test_paren_rejects_out_of_range_numbers():
    md = "measured at (300) K, but see (5)."
    res = link_citations(md, _numbered(10, style="{}."))
    assert [m.text for m in res.markers] == ["(5)"]


# ---------------------------------------------------------------------------
# author-year style (fmech)
# ---------------------------------------------------------------------------

AY = _author_year([("Bhushan", 2013), ("Marian", 2020), ("Vakis", 2018)])


def test_author_year_parenthetical():
    res = link_citations("as reviewed (Bhushan, 2013).", AY)
    assert [(m.text, m.ref_indices) for m in res.markers] == [("(Bhushan, 2013)", [0])]


def test_author_year_et_al_and_multi():
    res = link_citations("studies (Marian et al., 2020; Vakis et al., 2018) agree.", AY)
    assert len(res.markers) == 1
    assert res.markers[0].ref_indices == [1, 2]


def test_author_year_narrative():
    res = link_citations("Vakis et al. (2018) proposed a framework.", AY)
    assert [(m.text, m.ref_indices) for m in res.markers] == [
        ("Vakis et al. (2018)", [2])
    ]


def test_author_year_matches_across_diacritic_inconsistency():
    # the real fmech case: body prints "Höhn et al. (2011)", bibliography OCR'd the
    # same surname as "Hohn" -- folding must bridge both directions
    entries = _author_year([("Hohn", 2011)])
    res = link_citations("as evidenced by Höhn et al. (2011).", entries)
    assert [(m.text, m.ref_indices) for m in res.markers] == [("Höhn et al. (2011)", [0])]


def test_make_citekey_folds_diacritics():
    assert make_citekey({"authors": [{"family": "Höhn"}], "year": 2011}) == "hohn_2011"


def test_author_year_unknown_surname_left_unlinked():
    res = link_citations("as shown (Nobody, 2019).", AY)
    assert res.markers == [] and "(Nobody, 2019)" in res.ambiguous


def test_author_year_colliding_surname_year_dropped():
    entries = _author_year([("Smith", 2020), ("Smith", 2020)])
    res = link_citations("see (Smith, 2020).", entries)
    assert res.markers == []  # linking either would be a guess


def test_author_year_plain_year_parenthesis_not_linked():
    res = link_citations("first observed in (2013) by someone.", AY)
    assert res.markers == []


# ---------------------------------------------------------------------------
# make_citekey / rewrite_markers
# ---------------------------------------------------------------------------


def test_make_citekey():
    assert make_citekey({"authors": [{"family": "Brunton"}], "year": 2016}) == "brunton_2016"
    assert make_citekey({"authors": [{"family": "O'Brien"}], "year": 1999}) == "obrien_1999"


def test_make_citekey_missing_parts():
    assert make_citekey({"authors": [], "year": 2016}) is None
    assert make_citekey({"authors": [{"family": "X"}]}) is None


def test_rewrite_markers_replaces_back_to_front():
    md = "As shown [1], later [2,3]."
    markers = [Marker(9, 12, "[1]", [0]), Marker(20, 25, "[2,3]", [1, 2])]
    out = rewrite_markers(md, markers, ["kalman_1960", "wiener_1949", "zadeh_1950"])
    assert out == "As shown [kalman_1960], later [wiener_1949; zadeh_1950]."


def test_rewrite_markers_skips_marker_with_missing_citekey():
    md = "See [1] and [2]."
    markers = [Marker(4, 7, "[1]", [0]), Marker(12, 15, "[2]", [1])]
    out = rewrite_markers(md, markers, ["kalman_1960", None])
    assert out == "See [kalman_1960] and [2]."  # [2] untouched, never guessed


def test_link_citations_empty_bibliography():
    res = link_citations("some text [1]", [])
    assert res.markers == [] and res.uncited == []
