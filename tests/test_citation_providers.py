"""Tests for the provider HTTP layer (DOI extraction, response cache, normalizers).
Pure/mocked -- no live network: normalizer fixtures mirror the live-confirmed shapes
(2026-07-03 smoke test)."""

from __future__ import annotations

import urllib.error

import pytest

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


def test_get_json_attempts_overrides_the_retry_budget(tmp_path, monkeypatch):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=2, api_retry_base_delay=0.0)
    seen = []
    monkeypatch.setattr(cp, "call_with_backoff", lambda fn, attempts, delay: seen.append(attempts))
    cp._get_json("https://x.test/a", cfg)
    cp._get_json("https://x.test/b", cfg, attempts=7)
    assert seen == [2, 7]


@pytest.fixture
def patient(monkeypatch):
    # the circuit breaker is process-wide; each test starts patient
    cp._BULK_PATIENCE.set()
    yield
    cp._BULK_PATIENCE.set()


def test_source_list_calls_use_the_bulk_retry_budget(monkeypatch, patient):
    # one reference-list call replaces dozens of searches, so it waits longer;
    # a single title search keeps the ordinary budget
    cfg = _cfg(bulk_retry_attempts=9)
    monkeypatch.setenv(cfg.openalex_api_key_env, "k")
    seen = []

    def fake(url, cfg, headers=None, before_fetch=None, attempts=None):
        seen.append(attempts)
        return {"paperId": "P", "title": "T", "data": [], "referenced_works": ["W1"]}

    monkeypatch.setattr(cp, "_get_json", fake)
    cp.s2_paper_id(cfg, arxiv="2502.00963")
    cp.s2_references("P", cfg)
    cp.openalex_references("10.1/x", cfg)  # source lookup + one hydration batch
    cp.openalex_source(cfg, title="Some Title")
    assert seen == [9] * 5
    seen.clear()
    cp.s2_paper_id(cfg, title="Some Title")  # a keyless S2 search: ordinary budget
    assert seen == [None]
    seen.clear()
    cp.crossref_search("Some Title", cfg)
    assert seen == [None]


def test_keyless_openalex_list_gets_no_long_wait(monkeypatch, patient):
    # without a key OpenAlex answers 429 for lack of budget; waiting never fixes that
    cfg = _cfg(bulk_retry_attempts=9)
    monkeypatch.delenv(cfg.openalex_api_key_env, raising=False)
    seen = []
    monkeypatch.setattr(
        cp, "_get_json", lambda url, cfg, attempts=None, **kw: seen.append(attempts)
    )
    cp.openalex_references("10.1/x", cfg)
    assert seen == [None]


def test_exhausted_list_call_trips_the_breaker_until_one_succeeds(tmp_path, monkeypatch, patient):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=2, bulk_retry_attempts=7)
    throttled = urllib.error.HTTPError("https://api.semanticscholar.org/x", 429, "", {}, None)

    def failing(fn, attempts, delay):
        raise throttled

    monkeypatch.setattr(cp, "call_with_backoff", failing)
    assert cp.s2_references("P1", cfg) is None
    assert cp._list_attempts(cfg) == 2  # later list calls: ordinary budget

    seen = []
    monkeypatch.setattr(
        cp, "call_with_backoff", lambda fn, attempts, delay: seen.append(attempts) or {"data": []}
    )
    assert cp.s2_references("P2", cfg) == []
    assert seen == [2]
    assert cp._list_attempts(cfg) == 7  # a list call succeeding restores the patience


def test_non_retryable_list_failure_keeps_patience(tmp_path, monkeypatch, patient):
    # a 404 (paper unknown to S2) is an answer, not an outage
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=2, bulk_retry_attempts=7)
    missing = urllib.error.HTTPError("https://api.semanticscholar.org/x", 404, "", {}, None)

    def failing(fn, attempts, delay):
        raise missing

    monkeypatch.setattr(cp, "call_with_backoff", failing)
    assert cp._get_json("https://api.semanticscholar.org/a", cfg, attempts=7) is None
    assert cp._list_attempts(cfg) == 7


@pytest.fixture
def fresh_stats():
    # reset in place: citation_resolution holds the same PROVIDER_STATS object
    cp.reset_provider_state()
    yield cp
    cp.reset_provider_state()


def _http_error(url, code, body=b""):
    import io

    return urllib.error.HTTPError(url, code, "", {}, io.BytesIO(body))


def test_failed_attempts_warn_once_per_provider_and_class(fresh_stats, caplog):
    # rate limits and exhausted budgets used to be logged only at DEBUG
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg()
    s2 = f"{cfg.s2_api_base}/paper/search?query=x"
    for _ in range(3):
        cp._note_failed_attempt(s2, _http_error(s2, 429), cfg)
    cp._note_failed_attempt(s2, TimeoutError("timed out"), cfg)
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 2
    assert warnings[0].startswith("semanticscholar lookup failed (HTTP 429")
    assert "timeout" in warnings[1]
    counts = cp.PROVIDER_STATS.snapshot()
    assert counts[("semanticscholar", "429")] == 3
    assert counts[("semanticscholar", "timeout")] == 1


def test_openalex_budget_exhaustion_is_named(fresh_stats, caplog):
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg(mailto="me@example.org")
    url = f"{cfg.openalex_api_base}/works?search=x&mailto=me%40example.org"
    body = b'{"error": "Insufficient budget", "message": "daily budget used"}'
    cp._note_failed_attempt(url, _http_error(url, 429, body), cfg)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert msg.startswith("OpenAlex budget exhausted (HTTP 429: ")
    assert "Insufficient budget" in msg
    assert "example.org" not in msg  # the contact address never reaches a log line


def test_a_404_is_counted_as_missing_not_as_a_failure(tmp_path, monkeypatch, fresh_stats, caplog):
    # an unknown (often OCR-mangled) DOI is S2's answer, not a provider problem
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=3, api_retry_base_delay=0.0)
    url = f"{cfg.s2_api_base}/paper/DOI:10.1/nope"

    def not_found(req, timeout=None):
        raise _http_error(url, 404, b'{"error": "Paper with id DOI:10.1/nope not found"}')

    monkeypatch.setattr(cp.urllib.request, "urlopen", not_found)
    assert cp._get_json(url, cfg) is None
    counts = cp.PROVIDER_STATS.snapshot()
    assert counts[("semanticscholar", "missing")] == 1  # not retried, not failed
    assert counts[("semanticscholar", "failed")] == 0
    assert caplog.records == []


def test_error_warning_masks_contact_address_and_keys(fresh_stats, monkeypatch, caplog):
    caplog.set_level("WARNING", logger=cp.__name__)
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-key-123")
    cfg = _cfg(mailto="me@example.org")
    url = f"{cfg.crossref_api_base}/works?query=x&mailto=me%40example.org"
    echo = b"<html>502 for /works?query=x&mailto=me%40example.org from me@example.org "
    echo += b"secret-key-123</html>"
    cp._note_failed_attempt(url, _http_error(url, 502, echo), cfg)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert "crossref lookup failed (HTTP 502" in msg
    assert "example.org" not in msg and "secret-key-123" not in msg
    assert "***" in msg


def test_a_secret_crossing_the_cut_is_masked_not_truncated(fresh_stats, caplog):
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg(mailto="someone.long@example.org")
    url = f"{cfg.crossref_api_base}/works?query=x"
    body = b"x" * 150 + b" someone.long@example.org"  # crosses the 160-character cut
    cp._note_failed_attempt(url, _http_error(url, 502, body), cfg)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert "someone" not in msg


def test_a_secret_crossing_the_read_limit_is_dropped(fresh_stats, caplog):
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg(mailto="someone.long@example.org")
    url = f"{cfg.crossref_api_base}/works?query=x"
    # mostly whitespace, so collapsing would pull a cut-off secret back into view
    body = b" " * (cp._BODY_READ - 10) + b"someone.long@example.org"
    cp._note_failed_attempt(url, _http_error(url, 502, body), cfg)
    (msg,) = [r.getMessage() for r in caplog.records]
    assert "someone" not in msg


def test_rejected_openalex_key_warns_once_not_twice(fresh_stats, monkeypatch, caplog):
    caplog.set_level("WARNING", logger=cp.__name__)
    monkeypatch.setenv("OPENALEX_API_KEY", "bad")
    cfg = _cfg()
    url = f"{cfg.openalex_api_base}/works?search=x"
    cp._note_failed_attempt(url, _http_error(url, 401), cfg)
    assert caplog.records == []  # left to _warn_once_if_key_rejected
    assert cp.PROVIDER_STATS.snapshot()[("openalex", "4xx")] == 1


def test_a_rate_limited_provider_cools_down_then_returns(
    tmp_path, monkeypatch, fresh_stats, caplog
):
    # keyless S2 answering 429 must not make every later lookup sit through backoff
    caplog.set_level("WARNING", logger=cp.__name__)
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=1, provider_cooldown_s=60)
    s2 = f"{cfg.s2_api_base}/paper/search?query="
    now = [1000.0]
    monkeypatch.setattr(cp.time, "monotonic", lambda: now[0])
    fetches = []

    def throttled(req, timeout=None):
        fetches.append(req.full_url)
        raise _http_error(req.full_url, 429)

    monkeypatch.setattr(cp.urllib.request, "urlopen", throttled)
    for q in "abc":
        assert cp._get_json(s2 + q, cfg) is None
    assert any("skipping it for 1 min" in r.getMessage() for r in caplog.records)
    assert cp._get_json(s2 + "d", cfg) is None  # skipped: no request at all
    assert len(fetches) == 3
    assert cp.PROVIDER_STATS.snapshot()[("semanticscholar", "skipped")] == 1

    now[0] += 61  # cooldown over: asked again, and a success clears the streak
    monkeypatch.setattr(cp, "call_with_backoff", lambda fn, attempts, delay: {"data": []})
    assert cp._get_json(s2 + "e", cfg) == {"data": []}
    assert cp.PROVIDER_STATS.snapshot()[("semanticscholar", "ok")] == 1


def test_s2_can_be_asked_once_without_backoff(tmp_path, monkeypatch, fresh_stats):
    # S2 last in the order, asked once: a 429 costs one request, not a backoff ladder
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=5, s2_retry_attempts=1)
    seen = []
    monkeypatch.setattr(
        cp, "call_with_backoff", lambda fn, attempts, delay: seen.append(attempts) or {}
    )
    cp._get_json(f"{cfg.s2_api_base}/paper/search?query=x", cfg)
    cp._get_json(f"{cfg.crossref_api_base}/works?query=x", cfg)
    cp._get_json(f"{cfg.s2_api_base}/paper/P/references", cfg, attempts=7)  # a list call
    assert seen == [1, 5, 7]


def test_a_404_or_success_resets_the_rate_limit_streak(tmp_path, monkeypatch, fresh_stats):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=1, provider_cooldown_s=60)
    s2 = f"{cfg.s2_api_base}/paper/search?query="
    codes = iter([429, 429, 404, 429, 429])

    def respond(req, timeout=None):
        raise _http_error(req.full_url, next(codes))

    monkeypatch.setattr(cp.urllib.request, "urlopen", respond)
    for q in "abcde":
        cp._get_json(s2 + q, cfg)
    assert cp.PROVIDER_STATS.snapshot()[("semanticscholar", "skipped")] == 0


def test_get_json_counts_ok_cached_and_failed(tmp_path, monkeypatch, fresh_stats):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=1)
    url = f"{cfg.crossref_api_base}/works?query=x"
    monkeypatch.setattr(cp, "call_with_backoff", lambda fn, attempts, delay: {"ok": 1})
    cp._get_json(url, cfg)
    cp._get_json(url, cfg)  # cache hit

    def failing(fn, attempts, delay):
        raise ValueError("boom")

    monkeypatch.setattr(cp, "call_with_backoff", failing)
    cp._get_json(f"{cfg.crossref_api_base}/works?query=y", cfg)
    counts = cp.PROVIDER_STATS.snapshot()
    assert counts[("crossref", "ok")] == 1
    assert counts[("crossref", "cached")] == 1
    assert counts[("crossref", "failed")] == 1


def test_every_failed_attempt_is_counted_through_the_retry_loop(tmp_path, monkeypatch, fresh_stats):
    cfg = _cfg(api_cache_dir=str(tmp_path), api_retry_attempts=3, api_retry_base_delay=0.0)
    url = f"{cfg.crossref_api_base}/works?query=z"

    def throttled(req, timeout=None):
        raise _http_error(url, 429)

    monkeypatch.setattr(cp.urllib.request, "urlopen", throttled)
    assert cp._get_json(url, cfg) is None
    counts = cp.PROVIDER_STATS.snapshot()
    assert counts[("crossref", "429")] == 3
    assert counts[("crossref", "failed")] == 1


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


# ---------------------------------------------------------------------------
# polite-pool contact (mailto)
# ---------------------------------------------------------------------------


def test_mailto_goes_to_crossref_and_openalex_but_not_semantic_scholar(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("REFINERY_MAILTO", "me@example.org")
    urls = []
    monkeypatch.setattr(cp, "_get_json", lambda url, cfg, **kw: urls.append(url) or None)
    cfg = CitationConfig(s2_min_interval_s=0.0)
    cp.crossref_search("A title", cfg), cp.openalex_search("A title", cfg)
    cp.s2_search("A title", cfg)
    crossref, openalex, s2 = urls
    assert crossref.endswith("&mailto=me%40example.org")
    assert openalex.endswith("&mailto=me%40example.org")
    assert "mailto" not in s2
    assert "me@example.org" in cp._user_agent(cfg, crossref)
    assert "me@example.org" in cp._user_agent(cfg, openalex)
    assert "me@example.org" not in cp._user_agent(cfg, s2)


def test_config_mailto_wins_over_the_environment(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("REFINERY_MAILTO", "env@example.org")
    assert cp._mailto(CitationConfig(mailto="cfg@example.org"), "openalex") == "cfg@example.org"
    assert cp._mailto(CitationConfig(), "openalex") == "env@example.org"


def test_mailto_can_be_limited_to_openalex(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("REFINERY_MAILTO", "me@example.org")
    urls = []
    monkeypatch.setattr(cp, "_get_json", lambda url, cfg, **kw: urls.append(url) or None)
    cfg = CitationConfig(mailto_providers=["openalex"])
    cp.crossref_search("A title", cfg), cp.openalex_search("A title", cfg)
    crossref, openalex = urls
    assert "mailto" not in crossref and "me@example.org" not in cp._user_agent(cfg, crossref)
    assert openalex.endswith("&mailto=me%40example.org")
    assert "me@example.org" in cp._user_agent(cfg, openalex)


def test_mailto_is_not_part_of_the_cache_key():
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    cfg = CitationConfig(api_cache_dir="/tmp/x")
    plain = "https://api.crossref.org/works?query.bibliographic=T&rows=1"
    assert cp._cache_path(plain, cfg) == cp._cache_path(plain + "&mailto=me%40example.org", cfg)


def test_openalex_key_is_a_bearer_header_for_openalex_only(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("OPENALEX_API_KEY", "k3y")
    cfg = CitationConfig()
    for url in ("https://api.openalex.org/works?search=T", "https://api.openalex.org/works/doi:1"):
        assert cp._auth_headers(url, cfg) == {"Authorization": "Bearer k3y"}
    for other in ("https://api.crossref.org/works?rows=1", "https://api.semanticscholar.org/x"):
        assert cp._auth_headers(other, cfg) == {}


def test_openalex_key_never_reaches_a_url_or_another_host(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("OPENALEX_API_KEY", "k3y")
    sent = []
    monkeypatch.setattr(cp, "_cache_path", lambda url, cfg: None)
    monkeypatch.setattr(cp, "call_with_backoff", lambda fn, *a: fn())

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def urlopen(req, timeout):
        sent.append((req.full_url, req.get_header("Authorization")))
        return Resp()

    monkeypatch.setattr(cp.urllib.request, "urlopen", urlopen)  # unkeyed requests
    monkeypatch.setattr(cp._KEYED_OPENER, "open", urlopen)  # keyed requests
    cfg = CitationConfig()
    cp._get_json("https://api.openalex.org/works?search=T", cfg)
    cp._get_json("https://api.crossref.org/works?q=T", cfg)
    assert sent == [
        ("https://api.openalex.org/works?search=T", "Bearer k3y"),
        ("https://api.crossref.org/works?q=T", None),
    ]


def test_no_openalex_header_without_a_key(monkeypatch):
    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    assert cp._auth_headers("https://api.openalex.org/works?search=T", CitationConfig()) == {}


def test_a_redirect_to_another_host_drops_the_key():
    import io
    import urllib.request

    from paper_refinery import citation_providers as cp

    req = urllib.request.Request(
        "https://api.openalex.org/works", headers={"Authorization": "Bearer k"}
    )
    handler = cp._KeepCredentialsOnHost()
    elsewhere = handler.redirect_request(
        req, io.BytesIO(), 302, "Found", {}, "https://cdn.example.org/x"
    )
    same_host = handler.redirect_request(
        req, io.BytesIO(), 302, "Found", {}, "https://api.openalex.org/y"
    )
    assert not elsewhere.has_header("Authorization")
    assert same_host.get_header("Authorization") == "Bearer k"


def test_a_rejected_key_is_reported_once(monkeypatch, caplog):
    import logging
    import urllib.error

    from paper_refinery import citation_providers as cp
    from paper_refinery.config import CitationConfig

    monkeypatch.setenv("OPENALEX_API_KEY", "bad")
    monkeypatch.setattr(cp, "_REJECTED_KEY_WARNED", set())
    err = urllib.error.HTTPError("https://api.openalex.org/works", 401, "Unauthorized", {}, None)
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            cp._warn_once_if_key_rejected(
                "https://api.openalex.org/works?search=T", err, CitationConfig()
            )
    assert caplog.text.count("OpenAlex rejected the API key") == 1
