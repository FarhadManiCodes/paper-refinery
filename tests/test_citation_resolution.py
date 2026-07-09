"""Tests for API verification/resolution (layer 2/3). Pure/mocked -- no live network:
provider calls are monkeypatched (the provider functions themselves live in
citation_providers.py and are imported by name into citation_resolution.py, so
monkeypatching them on the ``cr`` module still reaches ``verify_and_resolve``)."""

from __future__ import annotations

import pytest

from paper_refinery import citation_resolution as cr
from paper_refinery.config import CitationConfig


def _cfg(**kw) -> CitationConfig:
    kw.setdefault("s2_min_interval_s", 0.0)  # no throttling delays in tests
    kw.setdefault("max_workers", 2)
    return CitationConfig(**kw)


# ---------------------------------------------------------------------------
# title_similarity
# ---------------------------------------------------------------------------


def test_source_confident_strong_title_accepts_despite_year_author_mismatch():
    # regression: a near-exact title identifies the (specific, known) source on its own;
    # supplied year/authors that drift (preprint year, different name) must NOT veto it, or
    # passing metadata would lose the fast-path a bare title-only lookup would have won.
    cfg = _cfg()
    title = "HYCO: A Formalism for Hybrid-Cooperative PDE Modelling"
    src = cr.SourcePaper(title=title, year=2026, authors=[{"family": "Liverani"}])
    candidate = {"title": title, "year": 2023, "authors": [{"family": "Zuazua"}]}
    assert cr._source_confident(src, candidate, cfg)


def test_source_confident_rejects_unrelated_title():
    cfg = _cfg()
    src = cr.SourcePaper(title="A New Approach to Linear Filtering", year=1960)
    candidate = {"title": "Deep residual learning for image recognition", "year": 1960}
    assert not cr._source_confident(src, candidate, cfg)


def test_source_from_meta_maps_bundle_and_prefers_its_title():
    meta = {
        "doi": "10.1/x",
        "title": "Authoritative Title",
        "year": 2021,
        "authors": [{"family": "Smith", "given": "J"}],
    }
    sp = cr.source_from_meta(meta, fallback_title="ocr guess")
    assert (sp.doi, sp.title, sp.year) == ("10.1/x", "Authoritative Title", 2021)
    assert sp.authors == [{"family": "Smith", "given": "J"}]


def test_source_from_meta_falls_back_to_ocr_title_when_bundle_has_none():
    # empty/None bundle -> only the OCR fallback title; nothing else set (fast-path still tries)
    sp = cr.source_from_meta(None, fallback_title="ocr guess")
    assert sp.title == "ocr guess"
    assert sp.doi is None and sp.year is None and sp.authors is None
    # a bundle without a title keeps the fallback but takes the rest
    sp2 = cr.source_from_meta({"doi": "10.1/y"}, fallback_title="ocr guess")
    assert sp2.doi == "10.1/y" and sp2.title == "ocr guess"


def test_title_similarity_near_identical_scores_high():
    a = "Discovering governing equations from data by sparse identification"
    b = "Discovering Governing Equations from Data by Sparse Identification."
    assert cr.title_similarity(a, b) > 0.95


def test_title_similarity_unrelated_scores_low():
    assert (
        cr.title_similarity(
            "A new approach to optimal filtering", "Deep residual learning for image recognition"
        )
        < 0.5
    )


# ---------------------------------------------------------------------------
# verify_and_resolve
# ---------------------------------------------------------------------------

S2_PAPER = {
    "title": "A New Approach to Linear Filtering and Prediction Problems",
    "year": 1960,
    "abstract": "The classical filtering problem...",
    "externalIds": {"DOI": "10.1115/1.3662552"},
    "publicationTypes": ["JournalArticle"],
}

EXTRACTED = {
    "title": "A new approach to linear filtering and prediction problems",
    "year": 1960,
    "volume": "82",
    "page": "35-45",
}


def test_doi_first_skips_similarity_entirely(monkeypatch):
    # candidate title is nothing like the extracted one -- a DOI match wins regardless
    monkeypatch.setattr(
        cr, "s2_by_doi", lambda doi, cfg: {**S2_PAPER, "title": "Utterly Different"}
    )
    monkeypatch.setattr(cr, "s2_search", lambda t, c: pytest.fail("must not title-search"))
    out = cr.verify_and_resolve(dict(EXTRACTED), "Kalman... doi:10.1115/1.3662552", _cfg())
    assert out["verified"] and out["match"] == "doi"
    assert out["title"] == "Utterly Different"  # provider data wins
    assert out["volume"] == "82"  # provider lacks it -> guess kept


def test_falls_back_to_title_search_when_doi_lookup_fails(monkeypatch):
    monkeypatch.setattr(cr, "s2_by_doi", lambda doi, cfg: None)
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: dict(S2_PAPER))
    out = cr.verify_and_resolve(dict(EXTRACTED), "text with doi 10.1115/1.3662552", _cfg())
    assert out["verified"] and out["match"] == "semanticscholar"


def test_provider_chain_order_crossref_s2_openalex(monkeypatch):
    # CrossRef leads (published-record dates); S2 second; OpenAlex last
    calls = []
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: calls.append("crossref"))
    monkeypatch.setattr(cr, "s2_search", lambda t, c: calls.append("s2"))
    monkeypatch.setattr(
        cr,
        "openalex_search",
        lambda t, c: (
            calls.append("openalex"),
            {
                "display_name": EXTRACTED["title"],
                "publication_year": 1960,
                "doi": "https://doi.org/10.1115/1.3662552",
                "type": "article",
            },
        )[1],
    )
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi here", _cfg())
    assert calls == ["crossref", "s2", "openalex"]
    assert out["verified"] and out["match"] == "openalex"


def test_rejects_low_similarity_match(monkeypatch):
    monkeypatch.setattr(
        cr, "s2_search", lambda t, c: {**S2_PAPER, "title": "Completely Unrelated Work"}
    )
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert not out["verified"] and out["match"] is None
    assert out["title"] == EXTRACTED["title"]  # guess kept entirely


def test_rejects_year_mismatch_beyond_tolerance(monkeypatch):
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "year": 1975})
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert not out["verified"]


def test_accepts_year_within_tolerance(monkeypatch):
    # confirmed live: S2 reports brunton-2016's arXiv year (2015) -- one off is fine
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "year": 1961})
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert out["verified"]


def test_provider_year_below_printed_is_kept_out(monkeypatch):
    # user decision: never pull the year backward -- a provider year below the printed
    # one is the preprint's (S2 merges preprint+published, reports the earliest year)
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "year": 1959})
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert out["verified"] and out["year"] == 1960  # printed year wins


def test_provider_year_above_printed_is_taken(monkeypatch):
    # the paper cited the preprint; the provider knows the later published version
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "year": 1961})
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert out["verified"] and out["year"] == 1961


def test_missing_extracted_year_skips_year_check(monkeypatch):
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: dict(S2_PAPER))
    extracted = {k: v for k, v in EXTRACTED.items() if k != "year"}
    out = cr.verify_and_resolve(extracted, "no doi", _cfg())
    assert out["verified"] and out["year"] == 1960  # provider fills the gap


def test_unverified_when_no_title_to_search(monkeypatch):
    monkeypatch.setattr(cr, "s2_search", lambda t, c: pytest.fail("nothing to search with"))
    out = cr.verify_and_resolve({}, "garbled OCR line with no doi", _cfg())
    assert not out["verified"]


def test_crossref_match_without_abstract_triggers_s2_followup(monkeypatch):
    crossref_item = {
        "DOI": "10.1115/1.3662552",
        "title": [EXTRACTED["title"]],
        "issued": {"date-parts": [[1960]]},
        "type": "journal-article",
        # no abstract (CrossRef's common case)
    }
    followups = []
    monkeypatch.setattr(cr, "s2_search", lambda t, c: None)
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: crossref_item)
    monkeypatch.setattr(
        cr, "s2_by_doi", lambda doi, cfg: (followups.append(doi), dict(S2_PAPER))[1]
    )
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi in raw text", _cfg())
    assert out["verified"] and out["match"] == "crossref"
    assert followups == ["10.1115/1.3662552"]
    assert out["abstract"] == S2_PAPER["abstract"]


def test_relaxed_tier_accepts_with_year_and_surname_corroboration(monkeypatch):
    # OCR-garbled title (sim between relaxed and strict) + exact year + same surname
    garbled = {
        "title": "A new aproach to linear fltering and predction problms extra junk",
        "year": 1960,
        "authors": [{"family": "Kalman", "given": "R. E."}],
    }
    s2_hit = {**S2_PAPER, "authors": [{"name": "R. E. Kalman"}]}
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: s2_hit)
    sim = cr.title_similarity(garbled["title"], S2_PAPER["title"])
    assert 0.75 <= sim < 0.90  # fixture sanity: really in the relaxed band
    out = cr.verify_and_resolve(garbled, "no doi", _cfg())
    assert out["verified"]


def test_relaxed_tier_rejects_without_surname_match(monkeypatch):
    garbled = {
        "title": "A new aproach to linear fltering and predction problms extra junk",
        "year": 1960,
        "authors": [{"family": "Wiener"}],  # wrong surname -> corroboration fails
    }
    monkeypatch.setattr(
        cr, "s2_search", lambda t, c: {**S2_PAPER, "authors": [{"name": "R. E. Kalman"}]}
    )
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    out = cr.verify_and_resolve(garbled, "no doi", _cfg())
    assert not out["verified"]


def test_relaxed_tier_surname_match_is_diacritic_folded():
    extracted = {"title": "T", "year": 2011, "authors": [{"family": "Hohn"}]}
    candidate = {"title": "T", "year": 2011, "authors": [{"family": "Höhn"}]}
    assert cr._first_family(extracted) == cr._first_family(candidate) == "hohn"


def test_published_version_preferred_over_acceptable_preprint(monkeypatch):
    # CrossRef (first in chain) returns an acceptable PREPRINT record -- the chain
    # must keep going and take S2's published version instead
    crossref_preprint = {
        "DOI": "10.48550/arXiv.1509.03580",
        "title": [EXTRACTED["title"]],
        "issued": {"date-parts": [[1960]]},
        "type": "posted-content",
    }
    s2_published = {**S2_PAPER, "authors": []}
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: crossref_preprint)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: s2_published)
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert out["verified"] and out["match"] == "semanticscholar"
    assert out["doi"] == "10.1115/1.3662552"


def test_preprint_used_as_fallback_when_no_published_version(monkeypatch):
    preprint = {
        "title": EXTRACTED["title"],
        "year": 1960,
        "externalIds": {"DOI": "10.48550/arXiv.1509.03580"},
        "publicationTypes": None,
    }
    monkeypatch.setattr(cr, "s2_search", lambda t, c: preprint)
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert out["verified"] and out["match"] == "semanticscholar"
    assert out["doi"].startswith("10.48550/")


def test_unverified_carries_best_near_miss(monkeypatch):
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "title": "Unrelated A"})
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(
        cr,
        "openalex_search",
        lambda t, c: {
            "display_name": "A new approach to linear filtering wrong year",
            "publication_year": 1999,
        },
    )
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", _cfg())
    assert not out["verified"]
    assert out["near_miss"]["provider"] == "openalex"  # the higher-similarity reject
    assert out["near_miss"]["year"] == 1999


def test_verified_overwrites_authors_from_provider(monkeypatch):
    extracted = {**EXTRACTED, "authors": [{"family": "Kalmn", "given": "R."}]}  # OCR-garbled
    monkeypatch.setattr(
        cr, "s2_by_doi", lambda doi, cfg: {**S2_PAPER, "authors": [{"name": "R. E. Kalman"}]}
    )
    out = cr.verify_and_resolve(extracted, "doi:10.1115/1.3662552", _cfg())
    assert out["verified"]
    assert out["authors"] == [{"family": "Kalman", "given": "R. E."}]


# ---------------------------------------------------------------------------
# format_resolution_report
# ---------------------------------------------------------------------------


def test_report_summarizes_verified_and_unverified():
    extracted = [
        {"title": "Paper A", "year": 2020, "authors": [{"family": "Ay"}]},
        {"title": "Paper B", "year": 2021},
    ]
    resolved = [
        {
            "number": "1",
            "title": "Paper A",
            "year": 2020,
            "doi": "10.1/a",
            "abstract": "abs",
            "authors": [{"family": "Ay"}],
            "verified": True,
            "match": "doi",
        },
        {
            "number": "2",
            "title": "Paper B",
            "year": 2021,
            "verified": False,
            "match": None,
            "raw_text": "2 Paper B...",
            "near_miss": {
                "provider": "crossref",
                "similarity": 0.81,
                "title": "Paper B-ish",
                "year": 2021,
            },
        },
    ]
    report = cr.format_resolution_report(extracted, resolved)
    assert "resolved 1/2 references (doi: 1)" in report
    assert "+doi+abstract" in report
    assert "UNVERIFIED (1)" in report
    assert "best reject: crossref sim 0.81" in report


# ---------------------------------------------------------------------------
# infer_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,provider_type,expected",
    [
        ("semanticscholar", "JournalArticle", "article-journal"),
        ("crossref", "proceedings-article", "paper-conference"),
        ("crossref", "posted-content", "article"),
        ("openalex", "book-chapter", "chapter"),
    ],
)
def test_infer_type_provider_mapping(source, provider_type, expected):
    assert cr.infer_type({"source": source, "provider_type": provider_type}, "") == expected


def test_infer_type_container_title_heuristic():
    assert cr.infer_type({"container_title": "Journal of Things"}, "") == "article-journal"


def test_infer_type_arxiv_heuristic():
    assert cr.infer_type({}, "Someone, arXiv preprint arXiv:1509.03580") == "article"


def test_infer_type_safe_default():
    assert cr.infer_type({}, "no signals at all") == "article"


def test_infer_type_unknown_provider_type_falls_through_to_heuristics():
    entry = {"source": "crossref", "provider_type": "some-new-type", "container_title": "J."}
    assert cr.infer_type(entry, "") == "article-journal"


# ---------------------------------------------------------------------------
# resolve_references
# ---------------------------------------------------------------------------


def test_resolve_references_merges_by_position(monkeypatch):
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda ext, raw, cfg: {**ext, "verified": False, "match": None}
    )
    extracted = [{"title": "A"}, {"title": "B"}]
    raw = [
        {"page": 9, "number": "1", "text": "1 First raw."},
        {"page": 9, "number": None, "text": "2 Second raw."},
    ]
    out = cr.resolve_references(extracted, raw, _cfg())
    assert [e["title"] for e in out] == ["A", "B"]  # order preserved
    assert out[0]["page"] == 9 and out[0]["raw_text"] == "1 First raw."
    assert out[0]["number"] == "1"
    assert out[1]["number"] == "2"  # recovered from the raw text's leading marker
    assert all("type" in e for e in out)


def test_resolve_references_pads_short_extraction(monkeypatch):
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda ext, raw, cfg: {**ext, "verified": False, "match": None}
    )
    raw = [{"page": 1, "number": None, "text": "1. Only raw."}]
    out = cr.resolve_references([], raw, _cfg())
    assert len(out) == 1 and out[0]["raw_text"] == "1. Only raw."


def test_resolve_references_empty():
    assert cr.resolve_references([], [], _cfg()) == []


# ---------------------------------------------------------------------------
# S2 bulk-references fast-path
# ---------------------------------------------------------------------------


def test_source_references_none_when_disabled_or_no_source():
    assert cr._source_references(None, _cfg(), 5) is None
    off = _cfg(s2_bulk_references=False)
    assert cr._source_references(cr.SourcePaper(doi="10.1/x"), off, 5) is None


def test_source_references_by_doi_is_trusted(monkeypatch):
    monkeypatch.setattr(
        cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}) if kw.get("doi") else None
    )
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [{"title": "Ref A"}])
    # n_refs=1 == len(s2) -> S2 covered it, no OpenAlex fill
    assert cr._source_references(cr.SourcePaper(doi="10.1/x"), _cfg(), 1) == [{"title": "Ref A"}]


def test_source_references_title_rejected_when_wrong_paper(monkeypatch):
    # S2's title hit is a DIFFERENT paper -> not confident -> never fetch its references
    monkeypatch.setattr(
        cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "A Completely Different Paper"})
    )
    monkeypatch.setattr(
        cr, "s2_references", lambda pid, cfg: pytest.fail("must not fetch wrong paper's refs")
    )
    assert cr._source_references(cr.SourcePaper(title="My Precise Paper Title"), _cfg(), 5) is None


def test_source_references_title_accepted_when_confident(monkeypatch):
    monkeypatch.setattr(
        cr,
        "s2_paper_id",
        lambda cfg, **kw: ("PID", {"title": "My Precise Paper Title", "year": 2020}),
    )
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [{"title": "X"}])
    src = cr.SourcePaper(title="My Precise Paper Title", year=2020)  # title-only: no OpenAlex fill
    assert cr._source_references(src, _cfg(), 1) == [{"title": "X"}]


def test_source_references_fills_from_openalex_when_s2_short(monkeypatch):
    # publisher-elided: S2 serves nothing; OpenAlex (by DOI) fills the pool
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [])
    oa = [{"title": "OA Ref", "doi": "10.5678/o", "source": "openalex"}]
    monkeypatch.setattr(cr, "openalex_references", lambda doi, cfg: oa)
    out = cr._source_references(cr.SourcePaper(doi="10.1/x"), _cfg(), 5)
    assert out == oa


def test_source_references_no_openalex_when_s2_covers(monkeypatch):
    s2 = [{"title": f"R{i}"} for i in range(6)]  # len 6 >= n_refs 5 -> S2 covered it
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: s2)
    monkeypatch.setattr(
        cr, "openalex_references", lambda doi, cfg: pytest.fail("S2 covered it; no OpenAlex")
    )
    assert len(cr._source_references(cr.SourcePaper(doi="10.1/x"), _cfg(), 5)) == 6


def test_dedup_candidates_by_doi_then_title():
    cands = [
        {"title": "Paper A", "doi": "10.1/a"},
        {"title": "Paper A (dup DOI)", "doi": "10.1/A"},  # same DOI (folded) -> dropped
        {"title": "Paper B", "doi": None},
        {"title": "Paper B", "doi": None},  # same title, no DOI -> dropped
    ]
    out = cr._dedup_candidates(cands)
    assert [c["title"] for c in out] == ["Paper A", "Paper B"]


def test_match_in_bulk_by_shared_doi():
    bulk = [
        {"title": "Wrong", "doi": "10.9999/z", "source": "semanticscholar"},
        {"title": "Right", "doi": "10.1234/abc", "source": "semanticscholar"},
    ]
    out = cr._match_in_bulk({"title": "x"}, "... see doi:10.1234/abc here", bulk, _cfg())
    assert out is not None and out["doi"] == "10.1234/abc" and out["match"] == "bulk"


def test_resolve_references_fastpath_matches_bulk_without_per_entry_search(monkeypatch):
    bulk = [cr.normalize_s2(dict(S2_PAPER))]  # the kalman paper, normalized
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: bulk)
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda *a, **k: pytest.fail("fast-path must not per-entry search")
    )
    raw = [
        {"page": 1, "number": "1", "text": "1. Kalman RE. A new approach to linear filtering..."}
    ]
    out = cr.resolve_references([dict(EXTRACTED)], raw, _cfg(), source=cr.SourcePaper(doi="10.1/x"))
    assert out[0]["verified"] is True
    assert out[0]["match"] == "bulk"
    assert out[0]["doi"] == "10.1115/1.3662552"  # enriched from the bulk candidate
    assert out[0]["number"] == "1"  # printed number preserved (from OCR, not S2)


def test_resolve_references_fastpath_falls_back_for_unmatched(monkeypatch):
    bulk = [cr.normalize_s2({"title": "Totally Unrelated Work", "year": 1900, "externalIds": {}})]
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: bulk)
    seen = []

    def fake_verify(ext, raw_text, cfg):
        seen.append(raw_text)
        return {**ext, "verified": False, "match": None}

    monkeypatch.setattr(cr, "verify_and_resolve", fake_verify)
    raw = [
        {"page": 1, "number": "1", "text": "1. Kalman RE. A new approach to linear filtering..."}
    ]
    out = cr.resolve_references([dict(EXTRACTED)], raw, _cfg(), source=cr.SourcePaper(doi="10.1/x"))
    assert seen  # no bulk match -> fell back to the per-entry path
    assert out[0]["verified"] is False
