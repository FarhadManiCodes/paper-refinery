"""Tests for the provider HTTP layer (DOI extraction, response cache, normalizers).
Pure/mocked -- no live network: normalizer fixtures mirror the live-confirmed shapes
(2026-07-03 smoke test)."""

from __future__ import annotations

from paper_refinery import citation_providers as cp
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
    return cp.extract_doi(text)


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
    assert cp.reconstruct_openalex_abstract(inverted) == (
        "Sparse identify models dynamics. dynamics."
    )


def test_reconstruct_openalex_abstract_empty():
    assert cp.reconstruct_openalex_abstract(None) is None
    assert cp.reconstruct_openalex_abstract({}) is None


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
    out = cp.normalize_s2(paper)
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
    out = cp.normalize_s2(paper)
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
    out = cp.normalize_crossref(item)
    assert out["authors"] == [{"family": "Liverani", "given": "L."}]


def test_normalize_openalex_authors_from_authorships():
    work = {
        "display_name": "T",
        "authorships": [{"author": {"display_name": "Lu Lu"}}, {"author": {}}],
    }
    out = cp.normalize_openalex(work)
    assert out["authors"] == [{"family": "Lu", "given": "Lu"}]


def test_normalize_s2_tolerates_nulls():
    out = cp.normalize_s2({"title": "T", "publicationTypes": None, "externalIds": None})
    assert out["doi"] is None and out["provider_type"] is None


def test_normalize_crossref():
    item = {
        "DOI": "10.1073/pnas.1517384113",
        "title": ["Discovering governing equations"],  # title is a LIST (confirmed live)
        "issued": {"date-parts": [[2016, 3, 28]]},
        "type": "journal-article",
        "abstract": "<jats:p>An abstract.</jats:p>",
    }
    out = cp.normalize_crossref(item)
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
    out = cp.normalize_openalex(work)
    assert out["doi"] == "10.1073/pnas.1517384113"
    assert out["abstract"] == "Hello world"
    assert out["provider_type"] == "article"


def test_normalizers_pass_none_through():
    assert cp.normalize_s2(None) is None
    assert cp.normalize_crossref(None) is None
    assert cp.normalize_openalex(None) is None


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

    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    assert cp._get_json("https://x.test/a", cfg) == {"ok": 1}
    assert cp._get_json("https://x.test/a", cfg) == {"ok": 1}  # cache hit
    assert len(calls) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_get_json_does_not_cache_failure(tmp_path, monkeypatch):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=1)

    def fake_urlopen(req, timeout=None):
        raise ValueError("boom")  # non-retryable

    monkeypatch.setattr(cp.urllib.request, "urlopen", fake_urlopen)
    assert cp._get_json("https://x.test/fail", cfg) is None
    assert list(tmp_path.iterdir()) == []  # a transient error must not stick


# ---------------------------------------------------------------------------
# source-paper lookup + bulk references (fast-path building blocks)
# ---------------------------------------------------------------------------


def _patch_get_json(monkeypatch, result):
    # the HTTP/cache layer is covered above; here we only exercise URL logic + parsing
    monkeypatch.setattr(cp, "_get_json", lambda *a, **k: result)


def test_s2_paper_id_by_doi_returns_id_and_candidate(monkeypatch):
    _patch_get_json(
        monkeypatch,
        {
            "paperId": "P1",
            "title": "Source Paper",
            "year": 2016,
            "authors": [{"name": "Jane Roe"}],
            "externalIds": {"DOI": "10.1/x"},
            "publicationTypes": ["JournalArticle"],
        },
    )
    result = cp.s2_paper_id(_cfg(), doi="10.1/x")
    assert result is not None
    pid, cand = result
    assert pid == "P1"
    assert cand["title"] == "Source Paper" and cand["year"] == 2016 and cand["doi"] == "10.1/x"


def test_s2_paper_id_by_arxiv(monkeypatch):
    _patch_get_json(monkeypatch, {"paperId": "P9", "title": "Preprint", "authors": []})
    result = cp.s2_paper_id(_cfg(), arxiv="1234.5678")
    assert result is not None and result[0] == "P9" and result[1]["title"] == "Preprint"


def test_s2_paper_id_by_title_returns_candidate_for_corroboration(monkeypatch):
    _patch_get_json(monkeypatch, {"data": [{"paperId": "P2", "title": "Src", "year": 2020}]})
    result = cp.s2_paper_id(_cfg(), title="Src")
    assert result is not None and result[0] == "P2"
    assert result[1]["title"] == "Src" and result[1]["year"] == 2020


def test_s2_paper_id_miss_returns_none(monkeypatch):
    _patch_get_json(monkeypatch, None)
    assert cp.s2_paper_id(_cfg(), doi="10.1/x") is None
    _patch_get_json(monkeypatch, {"data": []})
    assert cp.s2_paper_id(_cfg(), title="Nope") is None


def test_s2_references_normalizes_and_skips_null_cited(monkeypatch):
    _patch_get_json(
        monkeypatch,
        {
            "data": [
                {
                    "citedPaper": {
                        "title": "Ref A",
                        "year": 2020,
                        "externalIds": {"DOI": "10.1/a"},
                        "abstract": "abs A",
                        "authors": [{"name": "Jane Roe"}],
                        "publicationTypes": ["JournalArticle"],
                    }
                },
                {"citedPaper": None},  # S2 sometimes carries an unresolved reference
            ]
        },
    )
    out = cp.s2_references("P1", _cfg())
    assert out is not None and len(out) == 1
    assert out[0]["title"] == "Ref A"
    assert out[0]["doi"] == "10.1/a"
    assert out[0]["abstract"] == "abs A"
    assert out[0]["authors"] == [{"family": "Roe", "given": "Jane"}]
    assert out[0]["source"] == "semanticscholar"


def test_s2_references_fetch_failure_returns_none(monkeypatch):
    _patch_get_json(monkeypatch, None)
    assert cp.s2_references("P1", _cfg()) is None


def test_openalex_references_hydrates_referenced_works(monkeypatch):
    def fake_get(url, cfg, **kw):
        if "referenced_works" in url and "filter=" not in url:  # the id-list call
            return {"referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"]}
        return {  # the hydration call
            "results": [
                {
                    "display_name": "Ref One",
                    "publication_year": 2019,
                    "doi": "https://doi.org/10.1234/a",
                    "authorships": [],
                    "type": "article",
                }
            ]
        }

    monkeypatch.setattr(cp, "_get_json", fake_get)
    out = cp.openalex_references("10.1/x", _cfg())
    assert out is not None and out[0]["title"] == "Ref One"
    assert out[0]["doi"] == "10.1234/a" and out[0]["source"] == "openalex"


def test_openalex_references_none_when_no_referenced_works(monkeypatch):
    monkeypatch.setattr(cp, "_get_json", lambda *a, **k: {"referenced_works": []})
    assert cp.openalex_references("10.1/x", _cfg()) is None
