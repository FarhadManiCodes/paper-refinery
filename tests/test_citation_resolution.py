"""Tests for API verification/resolution (layer 2/3). Pure/mocked -- no live network:
provider calls are monkeypatched; normalizer fixtures mirror the live-confirmed shapes
(2026-07-03 smoke test)."""

from __future__ import annotations

import pytest

from paper_refinery import citation_resolution as cr
from paper_refinery.config import CitationConfig


def _cfg(**kw) -> CitationConfig:
    kw.setdefault("s2_min_interval_s", 0.0)  # no throttling delays in tests
    kw.setdefault("max_workers", 2)
    return CitationConfig(**kw)


# ---------------------------------------------------------------------------
# extract_doi
# ---------------------------------------------------------------------------


def test_extract_doi_bare():
    assert extract("... vol 12. doi:10.1073/pnas.1517384113") == "10.1073/pnas.1517384113"


def extract(text):
    return cr.extract_doi(text)


def test_extract_doi_url_form():
    assert extract("https://doi.org/10.3389/fmech.2021.655266 accessed") == (
        "10.3389/fmech.2021.655266"
    )


def test_extract_doi_strips_ocr_glued_punctuation():
    assert extract("10.1000/xyz123.") == "10.1000/xyz123"
    assert extract("(10.1000/xyz123),") == "10.1000/xyz123"


def test_extract_doi_none_when_absent():
    assert extract("Smith, J. (2020) A paper. Journal 12:1-10.") is None


# ---------------------------------------------------------------------------
# reconstruct_openalex_abstract
# ---------------------------------------------------------------------------


def test_reconstruct_openalex_abstract():
    inverted = {"models": [2], "Sparse": [0], "identify": [1], "dynamics.": [3, 4]}
    assert cr.reconstruct_openalex_abstract(inverted) == (
        "Sparse identify models dynamics. dynamics."
    )


def test_reconstruct_openalex_abstract_empty():
    assert cr.reconstruct_openalex_abstract(None) is None
    assert cr.reconstruct_openalex_abstract({}) is None


# ---------------------------------------------------------------------------
# title_similarity
# ---------------------------------------------------------------------------


def test_title_similarity_near_identical_scores_high():
    a = "Discovering governing equations from data by sparse identification"
    b = "Discovering Governing Equations from Data by Sparse Identification."
    assert cr.title_similarity(a, b) > 0.95


def test_title_similarity_unrelated_scores_low():
    assert cr.title_similarity("A new approach to optimal filtering", "Deep residual learning for image recognition") < 0.5


# ---------------------------------------------------------------------------
# normalizers (fixtures mirror live-confirmed shapes)
# ---------------------------------------------------------------------------


def test_normalize_s2():
    paper = {
        "title": "T",
        "year": 2015,
        "abstract": "Plain text abstract.",
        "externalIds": {"DOI": "10.1/x", "ArXiv": "1509.03580"},
        "publicationTypes": ["JournalArticle"],
    }
    out = cr._normalize_s2(paper)
    assert out == {
        "title": "T",
        "year": 2015,
        "doi": "10.1/x",
        "abstract": "Plain text abstract.",
        "authors": [],
        "source": "semanticscholar",
        "provider_type": "JournalArticle",
    }


def test_normalize_s2_splits_author_display_names():
    paper = {"title": "T", "authors": [{"name": "Steven L. Brunton"}, {"name": "Kutz"}]}
    out = cr._normalize_s2(paper)
    assert out["authors"] == [
        {"family": "Brunton", "given": "Steven L."},
        {"family": "Kutz", "given": None},
    ]


def test_normalize_crossref_authors_clean_family_given():
    item = {
        "title": ["T"],
        "author": [
            {"family": "Liverani", "given": "L."},
            {"name": "Some Consortium"},  # organization: no family -> skipped
        ],
    }
    out = cr._normalize_crossref(item)
    assert out["authors"] == [{"family": "Liverani", "given": "L."}]


def test_normalize_openalex_authors_from_authorships():
    work = {
        "display_name": "T",
        "authorships": [{"author": {"display_name": "Lu Lu"}}, {"author": {}}],
    }
    out = cr._normalize_openalex(work)
    assert out["authors"] == [{"family": "Lu", "given": "Lu"}]


def test_normalize_s2_tolerates_nulls():
    out = cr._normalize_s2({"title": "T", "publicationTypes": None, "externalIds": None})
    assert out["doi"] is None and out["provider_type"] is None


def test_normalize_crossref():
    item = {
        "DOI": "10.1073/pnas.1517384113",
        "title": ["Discovering governing equations"],  # title is a LIST (confirmed live)
        "issued": {"date-parts": [[2016, 3, 28]]},
        "type": "journal-article",
        "abstract": "<jats:p>An abstract.</jats:p>",
    }
    out = cr._normalize_crossref(item)
    assert out["title"] == "Discovering governing equations"
    assert out["year"] == 2016
    assert out["abstract"] == "An abstract."  # JATS tags stripped
    assert out["provider_type"] == "journal-article"


def test_normalize_openalex():
    work = {
        "display_name": "T",
        "publication_year": 2016,
        "doi": "https://doi.org/10.1073/pnas.1517384113",  # full URL (confirmed live)
        "type": "article",
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
    }
    out = cr._normalize_openalex(work)
    assert out["doi"] == "10.1073/pnas.1517384113"
    assert out["abstract"] == "Hello world"
    assert out["provider_type"] == "article"


def test_normalizers_pass_none_through():
    assert cr._normalize_s2(None) is None
    assert cr._normalize_crossref(None) is None
    assert cr._normalize_openalex(None) is None


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
    monkeypatch.setattr(cr, "s2_by_doi", lambda doi, cfg: {**S2_PAPER, "title": "Utterly Different"})
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
        lambda t, c: (calls.append("openalex"), {
            "display_name": EXTRACTED["title"],
            "publication_year": 1960,
            "doi": "https://doi.org/10.1115/1.3662552",
            "type": "article",
        })[1],
    )
    out = cr.verify_and_resolve(dict(EXTRACTED), "no doi here", _cfg())
    assert calls == ["crossref", "s2", "openalex"]
    assert out["verified"] and out["match"] == "openalex"


def test_rejects_low_similarity_match(monkeypatch):
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "title": "Completely Unrelated Work"})
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
    monkeypatch.setattr(cr, "s2_search", lambda t, c: {**S2_PAPER, "authors": [{"name": "R. E. Kalman"}]})
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
        cr, "openalex_search",
        lambda t, c: {"display_name": "A new approach to linear filtering wrong year",
                      "publication_year": 1999},
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
# response cache
# ---------------------------------------------------------------------------


def test_get_json_caches_success_and_replays_without_fetch(tmp_path, monkeypatch):
    cfg = _cfg(api_cache_dir=str(tmp_path))
    calls = []

    class FakeResp:
        status = 200

        def read(self):
            return b'{"ok": 1}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return FakeResp()

    monkeypatch.setattr(cr.urllib.request, "urlopen", fake_urlopen)
    assert cr._get_json("https://x.test/a", cfg) == {"ok": 1}
    assert cr._get_json("https://x.test/a", cfg) == {"ok": 1}  # cache hit
    assert len(calls) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_get_json_does_not_cache_failure(tmp_path, monkeypatch):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=1)

    def fake_urlopen(req, timeout=None):
        raise ValueError("boom")  # non-retryable

    monkeypatch.setattr(cr.urllib.request, "urlopen", fake_urlopen)
    assert cr._get_json("https://x.test/fail", cfg) is None
    assert list(tmp_path.iterdir()) == []  # a transient error must not stick


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
            "number": "1", "title": "Paper A", "year": 2020, "doi": "10.1/a",
            "abstract": "abs", "authors": [{"family": "Ay"}],
            "verified": True, "match": "doi",
        },
        {
            "number": "2", "title": "Paper B", "year": 2021, "verified": False,
            "match": None, "raw_text": "2 Paper B...",
            "near_miss": {"provider": "crossref", "similarity": 0.81,
                          "title": "Paper B-ish", "year": 2021},
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
