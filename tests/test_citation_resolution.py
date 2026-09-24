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


@pytest.fixture(autouse=True)
def _no_extra_candidates(monkeypatch):
    """The multi-hit fallback searches are live HTTP; tests that want them patch them."""
    for name in ("crossref_search_more", "s2_search_more", "openalex_search_more"):
        monkeypatch.setattr(cr, name, lambda t, c, n: [])
    # the OpenAlex source and DOI lookups are live HTTP too
    monkeypatch.setattr(cr, "openalex_by_doi", lambda doi, cfg: None)
    monkeypatch.setattr(cr, "openalex_source", lambda cfg, doi=None, title=None: None)
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: [])


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


def test_to_papis_citations_maps_verified_refs_and_skips_the_rest():
    refs = [
        {
            "title": "Alpha",
            "doi": "10.1/a",
            "year": 2007,
            "authors": [{"family": "Smith", "given": "J"}],
            "container_title": "J. X",
            "volume": "34",
            "verified": True,
        },
        {"title": "Unverified guess", "verified": False},  # skipped by default
        {"raw_text": "junk", "verified": True},  # no title/doi -> skipped
    ]
    out = cr.to_papis_citations(refs)
    assert out == [
        {
            "article-title": "Alpha",
            "DOI": "10.1/a",
            "author": "Smith",
            "year": "2007",
            "journal-title": "J. X",
            "volume": "34",
        }
    ]
    # --all includes unverified entries that still have a title/doi
    assert len(cr.to_papis_citations(refs, verified_only=False)) == 2


def test_papis_citations_round_trips_through_the_normalizer():
    # to_papis_citations is the inverse of _normalize_caller_references
    refs = [
        {
            "title": "Alpha Beta",
            "doi": "10.1/a",
            "year": 2007,
            "authors": [{"family": "Smith"}],
            "verified": True,
        }
    ]
    back = cr._normalize_caller_references(cr.to_papis_citations(refs))
    assert back[0]["title"] == "Alpha Beta" and back[0]["doi"] == "10.1/a"
    assert back[0]["year"] == 2007 and back[0]["authors"] == [{"family": "Smith"}]


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


OPENALEX_PUBLISHED = {
    "display_name": EXTRACTED["title"],
    "publication_year": 1960,
    "doi": "https://doi.org/10.1115/1.3662552",
    "type": "article",
}


def test_title_search_order_is_configurable(monkeypatch):
    # OpenAlex first (fast with a key): an acceptable published hit ends the chain there
    calls = []
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: calls.append("crossref"))
    monkeypatch.setattr(cr, "s2_search", lambda t, c: calls.append("s2"))
    monkeypatch.setattr(
        cr, "openalex_search", lambda t, c: (calls.append("openalex"), OPENALEX_PUBLISHED)[1]
    )
    cfg = _cfg(title_search_order=["openalex", "crossref", "semanticscholar"])
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi here", cfg)
    assert calls == ["openalex"]
    assert out["verified"] and out["match"] == "openalex"


def test_openalex_first_preprint_still_yields_to_a_published_record(monkeypatch):
    # the order must not trade accuracy for speed: OpenAlex's arXiv record is only a
    # fallback, and CrossRef's published record (with its own year) wins
    preprint = {**OPENALEX_PUBLISHED, "doi": "https://doi.org/10.48550/arXiv.1509.03580"}
    crossref_published = {
        "DOI": "10.1115/1.3662552",
        "title": [EXTRACTED["title"]],
        "issued": {"date-parts": [[1960]]},
        "type": "journal-article",
    }
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: preprint)
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: crossref_published)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: pytest.fail("published hit ends the chain"))
    monkeypatch.setattr(cr, "s2_by_doi", lambda doi, cfg: None)  # CrossRef abstract follow-up
    cfg = _cfg(title_search_order=["openalex", "crossref", "semanticscholar"])
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi", cfg)
    assert out["verified"] and out["match"] == "crossref"
    assert out["doi"] == "10.1115/1.3662552"


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
    # default order (S2 before OpenAlex): S2 serves nothing, OpenAlex by DOI fills the pool
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [])
    oa = [{"title": "OA Ref", "doi": "10.5678/o", "source": "openalex"}]
    monkeypatch.setattr(
        cr, "openalex_source", lambda cfg, doi=None, title=None: ({"title": "Src"}, ["W1"])
    )
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: oa)
    assert cr._source_references(cr.SourcePaper(doi="10.1/x"), _cfg(), 5) == oa


def test_source_references_stop_once_the_pool_covers_the_bibliography(monkeypatch):
    s2 = [{"title": f"R{i}"} for i in range(6)]  # 6 >= 5 printed -> OpenAlex not asked
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: s2)
    monkeypatch.setattr(
        cr, "openalex_source", lambda cfg, **kw: pytest.fail("S2 covered it; no OpenAlex")
    )
    assert len(cr._source_references(cr.SourcePaper(doi="10.1/x"), _cfg(), 5)) == 6


def test_source_references_ask_openalex_first_by_doi_or_title(monkeypatch):
    # OpenAlex first (paid, keyed): by DOI when known, else by title; S2 never asked
    cfg = _cfg(title_search_order=["openalex", "crossref"])
    asked = []
    monkeypatch.setattr(
        cr,
        "openalex_source",
        lambda cfg, doi=None, title=None: (
            asked.append(("doi", doi) if doi else ("title", title)),
            ({"title": "My Precise Book Title", "year": 2020}, ["W1", "W2"]),
        )[1],
    )
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: [{"title": i} for i in ids])
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: pytest.fail("S2 not in the order"))
    assert len(cr._source_references(cr.SourcePaper(doi="10.1/x"), cfg, 2)) == 2
    by_title = cr.SourcePaper(title="My Precise Book Title", year=2020)
    assert len(cr._source_references(by_title, cfg, 2)) == 2
    assert asked == [("doi", "10.1/x"), ("title", "My Precise Book Title")]


def test_default_order_asks_openalex_by_title_when_s2_is_short(monkeypatch):
    # default order (crossref, semanticscholar, openalex): S2 first; with no DOI, OpenAlex
    # is now asked by title to fill the gap
    monkeypatch.setattr(
        cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "My Precise Book Title"})
    )
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [])
    monkeypatch.setattr(
        cr,
        "openalex_source",
        lambda cfg, doi=None, title=None: (
            ({"title": "My Precise Book Title"}, ["W1"])
            if title
            else pytest.fail("no DOI: by title")
        ),
    )
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: [{"title": "Ref"}])
    out = cr._source_references(cr.SourcePaper(title="My Precise Book Title"), _cfg(), 3)
    assert out == [{"title": "Ref"}]


def test_an_openalex_list_that_hydrates_nothing_is_flagged(monkeypatch, caplog):
    caplog.set_level("INFO", logger=cr.__name__)
    cfg = _cfg(title_search_order=["openalex"])
    monkeypatch.setattr(
        cr, "openalex_source", lambda cfg, doi=None, title=None: ({"title": "B"}, ["W1", "W2"])
    )
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: [{"title": "only one"}])
    cr._source_references(cr.SourcePaper(doi="10.1/b"), cfg, 5)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]  # 1 of 2: fine
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: [])
    cr._source_references(cr.SourcePaper(doi="10.1/b"), cfg, 5)
    assert [r for r in caplog.records if r.levelname == "WARNING"]  # none of 2: warn


def test_source_references_reject_an_openalex_title_hit_for_another_work(monkeypatch):
    cfg = _cfg(title_search_order=["openalex"])
    monkeypatch.setattr(
        cr,
        "openalex_source",
        lambda cfg, doi=None, title=None: ({"title": "A Completely Different Book"}, ["W1"]),
    )
    monkeypatch.setattr(cr, "openalex_hydrate", lambda ids, cfg: pytest.fail("wrong work"))
    assert cr._source_references(cr.SourcePaper(title="My Precise Book Title"), cfg, 5) is None


def test_source_references_logs_every_step(monkeypatch, caplog):
    # a failed fast path used to fall back silently to per-reference search
    caplog.set_level("INFO", logger=cr.__name__)
    monkeypatch.setattr(cr, "s2_paper_id", lambda cfg, **kw: ("PID", {"title": "Src"}))
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: [{"title": "Ref A"}])
    src = cr.SourcePaper(arxiv="2502.00963")
    cr._source_references(src, _cfg(), 1, label="p.pdf")
    assert "p.pdf: reference list: semanticscholar by arXiv id: 1 references" in caplog.text
    assert "p.pdf: reference list: 1 candidates; matching locally" in caplog.text

    # a throttled list fetch (None) must not read as "no list"
    caplog.clear()
    monkeypatch.setattr(cr, "s2_references", lambda pid, cfg: None)
    assert cr._source_references(src, _cfg(), 1, label="p.pdf") is None
    (warning,) = [r for r in caplog.records if r.levelname == "WARNING"]
    assert "found, but the list fetch failed" in warning.getMessage()
    assert "reference list: none available; searching each of 1 references" in caplog.text

    caplog.clear()
    book = cr.SourcePaper(doi="10.1/book")
    monkeypatch.setattr(
        cr, "openalex_source", lambda cfg, doi=None, title=None: ({"title": "Book"}, [])
    )
    cfg = _cfg(title_search_order=["openalex", "crossref"])
    assert cr._source_references(book, cfg, 3, label="b.pdf") is None
    assert 'b.pdf: reference list: openalex by DOI: found "Book" (None), no references' in (
        caplog.text
    )


def test_dedup_candidates_by_doi_then_title():
    cands = [
        {"title": "Paper A", "doi": "10.1/a"},
        {"title": "Paper A (dup DOI)", "doi": "10.1/A"},  # same DOI (folded) -> dropped
        {"title": "Paper B", "doi": None},
        {"title": "Paper B", "doi": None},  # same title, no DOI -> dropped
    ]
    out = cr._dedup_candidates(cands)
    assert [c["title"] for c in out] == ["Paper A", "Paper B"]


def test_normalize_caller_references_handles_crossref_and_refinery_shapes():
    entries = [
        {"article-title": "T1", "DOI": "10.1/a", "year": "2007", "author": "Smith"},  # crossref
        {"title": "T2", "doi": "10.2/b", "year": 2010, "authors": [{"family": "Doe"}]},  # refinery
        {"volume-title": "Book Title", "year": "2008"},  # book, no doi -> kept (has a title)
        {"year": "1999"},  # neither title nor doi -> dropped
        "not a dict",  # ignored
    ]
    out = cr._normalize_caller_references(entries)
    assert [c["title"] for c in out] == ["T1", "T2", "Book Title"]
    assert out[0]["doi"] == "10.1/a" and out[0]["year"] == 2007
    assert out[0]["authors"] == [{"family": "Smith"}] and out[0]["_pool"] == "papis"
    assert out[2]["doi"] is None  # book kept on title alone


def test_resolve_references_uses_caller_references_locally(monkeypatch):
    # papis-supplied references match printed refs with NO network and report match="papis";
    # no source lookup, no per-entry search needed
    monkeypatch.setattr(cr, "s2_paper_id", lambda *a, **k: pytest.fail("no source lookup"))
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda *a, **k: pytest.fail("no per-entry search")
    )
    source = cr.SourcePaper(
        references=[
            {"article-title": "Attention Is All You Need", "DOI": "10.5/aiayn", "year": "2017"}
        ]
    )
    raw = [{"page": 1, "number": "1", "text": "[1] Vaswani et al. Attention is all you need."}]
    out = cr.resolve_references(
        [{"title": "Attention is all you need"}], raw, _cfg(), source=source
    )
    assert out[0]["verified"] and out[0]["match"] == "papis"
    assert out[0]["doi"] == "10.5/aiayn" and out[0]["year"] == 2017


def test_resolve_references_skips_s2_when_caller_covers_all(monkeypatch):
    # when the caller's references resolve EVERY printed ref, the S2/OpenAlex source fetch is
    # skipped entirely -- zero network ("don't re-resolve what papis already has")
    monkeypatch.setattr(
        cr,
        "_source_references",
        lambda *a, **k: pytest.fail("must not fetch S2 when papis covers all"),
    )
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda *a, **k: pytest.fail("no per-entry search")
    )
    source = cr.SourcePaper(
        references=[
            {"article-title": "Alpha Study", "DOI": "10.1/a"},
            {"article-title": "Beta Study", "DOI": "10.2/b"},
        ]
    )
    raw = [
        {"page": 1, "number": "1", "text": "[1] Alpha study. 2020."},
        {"page": 1, "number": "2", "text": "[2] Beta study. 2021."},
    ]
    out = cr.resolve_references(
        [{"title": "Alpha Study"}, {"title": "Beta Study"}], raw, _cfg(), source=source
    )
    assert all(o["match"] == "papis" and o["verified"] for o in out)


def test_resolve_references_fetches_s2_only_for_caller_gaps(monkeypatch):
    # a ref the caller lacks triggers the S2 fetch (once); the caller-covered ref stays "papis"
    calls = {"s2": 0}

    def fake_source_refs(src, cfg, n, **kw):
        calls["s2"] += 1
        return [{"title": "Gamma Study", "doi": "10.3/c", "source": "semanticscholar"}]

    monkeypatch.setattr(cr, "_source_references", fake_source_refs)
    monkeypatch.setattr(
        cr, "verify_and_resolve", lambda *a, **k: pytest.fail("bulk should cover it")
    )
    source = cr.SourcePaper(references=[{"article-title": "Alpha Study", "DOI": "10.1/a"}])
    raw = [
        {"page": 1, "number": "1", "text": "[1] Alpha study."},
        {"page": 1, "number": "2", "text": "[2] Gamma study."},
    ]
    out = cr.resolve_references(
        [{"title": "Alpha Study"}, {"title": "Gamma Study"}], raw, _cfg(), source=source
    )
    assert calls["s2"] == 1
    assert out[0]["match"] == "papis" and out[1]["match"] == "bulk"


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


def test_resolve_references_logs_progress(monkeypatch, caplog):
    # a long bibliography must say how far it is, not only when it finishes
    caplog.set_level("INFO", logger=cr.__name__)
    monkeypatch.setattr(
        cr,
        "verify_and_resolve",
        lambda extracted, raw_text, cfg: {"verified": raw_text != "3", "match": "openalex"},
    )
    raw = [{"page": 1, "number": str(i), "text": str(i)} for i in range(1, 6)]
    cr.resolve_references([{}] * 5, raw, _cfg(), label="hastie", progress_every=2)
    lines = [r.getMessage() for r in caplog.records]
    assert "hastie: resolving 5 references" in lines
    progress = [m for m in lines if m.startswith("hastie: resolved")]
    assert [m.split(" references")[0] for m in progress] == [
        "hastie: resolved 2/5",
        "hastie: resolved 4/5",
        "hastie: resolved 5/5",
    ]
    assert progress[-1].startswith("hastie: resolved 5/5 references (4 verified: openalex 4), ")
    assert progress[-1].endswith("; lookups: none")  # the stub made no provider calls

    # a total that is a multiple of the interval still gets exactly one final line
    caplog.clear()
    cr.resolve_references([{}] * 4, raw[:4], _cfg(), label="even", progress_every=2)
    finals = [r.getMessage() for r in caplog.records if "resolved 4/4" in r.getMessage()]
    assert len(finals) == 1
    assert caplog.records[0].getMessage() == "even: resolving 4 references"


def test_format_lookups_lists_answers_then_failed_attempts():
    from collections import Counter

    counts = Counter(
        {
            ("openalex", "ok"): 610,
            ("openalex", "cached"): 400,
            ("openalex", "failed"): 3,
            ("openalex", "429"): 5,
            ("openalex", "missing"): 2,
            ("semanticscholar", "timeout"): 2,
            ("crossref", "ok"): 0,
        }
    )
    assert cr.format_lookups(counts) == (
        "openalex 610 ok, 400 cached, 2 missing, 3 failed [429x5]; semanticscholar 0 ok [timeoutx2]"
    )
    assert cr.format_lookups(Counter()) == "none"


def test_printed_doi_follows_the_order_openalex_first(monkeypatch):
    asked = []
    monkeypatch.setattr(
        cr,
        "openalex_by_doi",
        lambda doi, cfg: (asked.append("openalex"), OPENALEX_PUBLISHED)[1],
    )
    monkeypatch.setattr(cr, "s2_by_doi", lambda doi, cfg: pytest.fail("OpenAlex answered"))
    cfg = _cfg(title_search_order=["openalex", "crossref", "semanticscholar"])
    out = cr.verify_and_resolve(dict(EXTRACTED), "doi 10.1115/1.3662552", cfg)
    assert asked == ["openalex"] and out["match"] == "doi"


def test_each_reference_is_logged_with_its_outcome(monkeypatch, caplog):
    caplog.set_level("INFO", logger=cr.__name__)

    def fake(extracted, raw_text, cfg):
        if raw_text == "1":
            return {**extracted, "verified": True, "match": "openalex"}
        return {
            **extracted,
            "verified": False,
            "near_miss": {"provider": "crossref", "similarity": 0.62, "title": "Other"},
        }

    monkeypatch.setattr(cr, "verify_and_resolve", fake)
    extracted = [
        {"title": "Convex Optimization", "year": 2004, "authors": [{"family": "Boyd"}]},
        {"title": "A Method", "year": 1983, "authors": [{"family": "Nesterov"}]},
    ]
    raw = [{"page": 1, "number": str(i), "text": str(i)} for i in (1, 2)]
    cr.resolve_references(extracted, raw, _cfg(), label="b.pdf")
    lines = [r.getMessage() for r in caplog.records]
    assert 'b.pdf: [1/2] Boyd 2004 "Convex Optimization" -> verified: openalex (1.00)' in lines
    assert 'b.pdf: [2/2] Nesterov 1983 "A Method" -> unverified (best: crossref 0.62 "Other")' in (
        lines
    )
    caplog.clear()
    cr.resolve_references(extracted, raw, _cfg(log_each_reference=False), label="b.pdf")
    assert not any("[1/2]" in r.getMessage() for r in caplog.records)


def test_a_title_of_page_numbers_is_never_searched(monkeypatch):
    # an index line's "title" "94" once matched a record exactly (2026-09-24)
    monkeypatch.setattr(cr, "_resolve_by_title", lambda *a: pytest.fail("not searchable"))
    out = cr.verify_and_resolve({"title": "94"}, "Weisberg, S. 94", _cfg())
    assert out["verified"] is False
    assert cr._searchable("Hints")  # a real one-word title still is


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


# ---------------------------------------------------------------------------
# tier-1 author check (_authors_disagree)
# ---------------------------------------------------------------------------


def _tier1(extracted_authors, candidate_authors, title="Compressive sensing"):
    extracted = {"title": title, "year": 2007, "authors": extracted_authors}
    candidate = {"title": "Compressed sensing", "year": 2006, "authors": candidate_authors}
    return cr._acceptable(extracted, candidate, _cfg())


def test_tier1_rejects_near_identical_title_by_different_authors():
    # live 2026-09-24: Baraniuk's "Compressive sensing" resolved to Donoho's paper
    assert cr.title_similarity("Compressive sensing", "Compressed sensing") >= 0.90
    assert not _tier1([{"family": "Baraniuk", "given": "R. G."}], [{"family": "Donoho"}])


def test_tier1_accepts_when_any_author_matches_in_any_order():
    assert _tier1([{"family": "Tao", "given": "T."}], [{"family": "Candès"}, {"family": "Tao"}])


def test_tier1_author_check_tolerates_ocr_spelling_and_diacritics():
    assert _tier1([{"family": "Ptluri", "given": "S."}], [{"family": "Potluri"}])
    assert _tier1([{"family": "Candes", "given": "E."}], [{"family": "Candès"}])


@pytest.mark.parametrize(
    "printed, listed",
    [
        ([], [{"family": "Donoho"}]),  # ditto marks: no author printed
        ([{"family": "Kolmogorov"}], [{"family": "Колмогоров"}]),  # other script
        ([{"family": "Baraniuk"}], []),  # provider record without authors
        ([{"family": "OpenAI"}], [{"family": "Achiam", "given": "Josh"}]),  # organisation
    ],
)
def test_tier1_author_check_passes_when_nothing_can_be_compared(printed, listed):
    assert _tier1(printed, listed)


def test_provider_name_suffix_is_not_the_surname():
    from paper_refinery.citation_providers import _split_full_name

    assert _split_full_name("Martin L. King Jr.") == {"family": "King", "given": "Martin L."}
    assert _split_full_name("John Smith III")["family"] == "Smith"


# ---------------------------------------------------------------------------
# fallback to further title-search hits
# ---------------------------------------------------------------------------

TURBULENCE = {
    "title": "Two-dimensional turbulence",
    "year": 2012,
    "authors": [{"family": "Boffetta", "given": "G."}, {"family": "Ecke", "given": "R. E."}],
}


def _s2_hit(title, year, *families):
    return {"title": title, "year": year, "authors": [{"name": f"A. {f}"} for f in families]}


def test_rejected_top_hit_falls_back_to_the_next_matching_one(monkeypatch):
    # live 2026-09-24: a generic title's top hit is someone else's paper
    wrong = _s2_hit("Two-dimensional turbulence", 2012, "Kraichnan", "Montgomery")
    right = _s2_hit("Two-Dimensional Turbulence", 2012, "Boffetta", "Ecke")
    asked = []
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: wrong)
    monkeypatch.setattr(cr, "s2_search_more", lambda t, c, n: asked.append(n) or [wrong, right])
    out = cr.verify_and_resolve(dict(TURBULENCE), "no doi", _cfg())
    assert out["verified"] and asked == [5]
    assert [a["family"] for a in out["authors"]][:2] == ["Boffetta", "Ecke"]


def test_accepted_top_hit_never_fetches_more(monkeypatch):
    right = _s2_hit("Two-Dimensional Turbulence", 2012, "Boffetta", "Ecke")
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: right)
    monkeypatch.setattr(cr, "s2_search_more", lambda t, c, n: pytest.fail("no extra request"))
    assert cr.verify_and_resolve(dict(TURBULENCE), "no doi", _cfg())["verified"]


def test_search_candidates_one_disables_the_fallback(monkeypatch):
    wrong = _s2_hit("Two-dimensional turbulence", 2012, "Kraichnan", "Montgomery")
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: wrong)
    monkeypatch.setattr(cr, "s2_search_more", lambda t, c, n: pytest.fail("disabled"))
    assert not cr.verify_and_resolve(dict(TURBULENCE), "no doi", _cfg(search_candidates=1))[
        "verified"
    ]


def test_single_hit_search_keeps_its_cached_url(monkeypatch):
    from paper_refinery import citation_providers as cp

    urls = []
    monkeypatch.setattr(cp, "_get_json", lambda url, cfg, **kw: urls.append(url) or None)
    cfg = _cfg(mailto="")
    (
        cp.s2_search("A title", cfg),
        cp.crossref_search("A title", cfg),
        cp.openalex_search("A title", cfg),
    )
    assert (
        urls[0].endswith("&limit=1")
        and urls[1].endswith("&rows=1")
        and urls[2].endswith("&per-page=1")
    )


def test_fallback_that_finds_nothing_leaves_the_reference_unverified(monkeypatch):
    wrong = _s2_hit("Two-dimensional turbulence", 2012, "Kraichnan", "Montgomery")
    monkeypatch.setattr(cr, "crossref_search", lambda t, c: None)
    monkeypatch.setattr(cr, "openalex_search", lambda t, c: None)
    monkeypatch.setattr(cr, "s2_search", lambda t, c: wrong)
    out = cr.verify_and_resolve(dict(TURBULENCE), "no doi", _cfg())  # autouse stub: []
    assert not out["verified"]
