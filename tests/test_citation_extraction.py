"""Tests for citation_extraction.py (pure/mocked; the real Gemini call is live-only)."""

from __future__ import annotations

import pytest

from paper_refinery.citation_extraction import (
    Author,
    ExtractedReference,
    _sanitize_citation_keys,
    extract_references,
    make_client,
)
from paper_refinery.config import CitationConfig

# ---------------------------------------------------------------------------
# schema sanity
# ---------------------------------------------------------------------------


def test_extracted_reference_requires_only_title():
    ref = ExtractedReference(title="Paper A")
    assert ref.title == "Paper A"
    assert ref.authors == []
    assert ref.year is None


def test_extracted_reference_model_dump_excludes_none():
    ref = ExtractedReference(title="Paper A", year=2020)
    assert ref.model_dump(exclude_none=True) == {"title": "Paper A", "authors": [], "year": 2020}


def test_author_given_is_optional():
    a = Author(family="Smith")
    assert a.given is None


# ---------------------------------------------------------------------------
# make_client
# ---------------------------------------------------------------------------


def test_make_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        make_client(CitationConfig())


# ---------------------------------------------------------------------------
# extract_references
# ---------------------------------------------------------------------------


def test_extract_references_returns_empty_for_no_input():
    assert extract_references([], CitationConfig()) == []


def test_extract_references_uses_response_parsed():
    parsed = [
        ExtractedReference(title="Paper A", authors=[Author(family="Smith", given="J.")]),
        ExtractedReference(title="Paper B", year=2020),
    ]

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return type("R", (), {"parsed": parsed})()

    out = extract_references(["ref one", "ref two"], CitationConfig(), client=FakeClient())
    assert out == [
        {"title": "Paper A", "authors": [{"family": "Smith", "given": "J."}]},
        {"title": "Paper B", "authors": [], "year": 2020},
    ]


def test_extract_references_pads_when_model_returns_fewer_items():
    parsed = [ExtractedReference(title="Only One")]

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return type("R", (), {"parsed": parsed})()

    with pytest.warns(UserWarning, match="got 1 items for 2 input"):
        out = extract_references(["ref one", "ref two"], CitationConfig(), client=FakeClient())
    assert out == [{"title": "Only One", "authors": []}, {}]


def test_extract_references_retries_transient_then_succeeds():
    parsed = [ExtractedReference(title="Paper A")]
    calls = []

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                calls.append(1)
                if len(calls) < 3:
                    raise ConnectionError("transient")  # retryable (see retry.py)
                return type("R", (), {"parsed": parsed})()

    cfg = CitationConfig(retry_attempts=4, retry_base_delay=0.0)
    out = extract_references(["ref one"], cfg, client=FakeClient())
    assert out == [{"title": "Paper A", "authors": []}]
    assert len(calls) == 3


def test_extract_references_raises_after_exhausting_retries():
    calls = []

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                calls.append(1)
                raise ConnectionError("still down")

    cfg = CitationConfig(retry_attempts=2, retry_base_delay=0.0)
    with pytest.raises(ConnectionError, match="still down"):
        extract_references(["ref one"], cfg, client=FakeClient())
    assert len(calls) == 2


def test_extract_references_fails_fast_on_non_retryable_error():
    # a bad key / bad request won't get better by waiting -- no backoff burn
    calls = []

    class FakeAuthError(Exception):
        code = 401

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                calls.append(1)
                raise FakeAuthError("invalid api key")

    cfg = CitationConfig(retry_attempts=4, retry_base_delay=10.0)  # delay never slept
    with pytest.raises(FakeAuthError):
        extract_references(["ref one"], cfg, client=FakeClient())
    assert len(calls) == 1


def test_extract_references_sends_schema_and_deterministic_config():
    seen = {}

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                seen["model"] = model
                seen["config"] = config
                return type("R", (), {"parsed": [ExtractedReference(title="Paper A")]})()

    extract_references(["ref one"], CitationConfig(), client=FakeClient())
    assert seen["model"] == "gemini-3.1-flash-lite"
    assert seen["config"].temperature == 0.0
    assert seen["config"].response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# _sanitize_citation_keys (fabricated-key guard)
# ---------------------------------------------------------------------------


def test_sanitize_drops_fabricated_keys_for_markerless_bibliography():
    # the live fmech failure: Gemini invented "1."/"2."/... for an author-year
    # bibliography that prints no markers -- style inference then flips to numbered
    raws = ["Bianchini, C. (2017). Windage losses.", "Boness, R. J. (1989). Churning."]
    items = [{"citation_key": "1.", "title": "A"}, {"citation_key": "2.", "title": "B"}]
    out = _sanitize_citation_keys(raws, items)
    assert all("citation_key" not in item for item in out)


def test_sanitize_keeps_keys_confirmed_by_raw_text_head():
    raws = ["[6] Liverani, L. (2025).", "3. Bongard J (2007).", "2 L. A. Zadeh (1950)."]
    items = [{"citation_key": "[6]"}, {"citation_key": "3."}, {"citation_key": "2"}]
    out = _sanitize_citation_keys(raws, items)
    assert [i.get("citation_key") for i in out] == ["[6]", "3.", "2"]


def test_sanitize_drops_key_whose_number_mismatches_raw_head():
    out = _sanitize_citation_keys(["12. Smith, J."], [{"citation_key": "13."}])
    assert "citation_key" not in out[0]


def test_sanitize_leaves_absent_and_non_numeric_keys_alone():
    raws = ["Smith, J. (2020).", "Jones, K. (2021)."]
    items = [{}, {"citation_key": "Jones2021"}]
    out = _sanitize_citation_keys(raws, items)
    assert out[0] == {} and out[1]["citation_key"] == "Jones2021"
