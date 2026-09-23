"""Tests for citation_extraction.py (pure/mocked; the real Gemini call is live-only)."""

from __future__ import annotations

import logging

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

    raws = ["[1] J. Smith, Paper A, 2019.", "[2] Paper B, 2020."]
    out = extract_references(raws, CitationConfig(), client=FakeClient())
    assert out == [
        {"title": "Paper A", "authors": [{"family": "Smith", "given": "J."}]},
        {"title": "Paper B", "authors": [], "year": 2020},
    ]


def test_extract_references_never_guesses_position_for_a_short_unnumbered_response(caplog):
    # 1 row for 2 lines and no line numbers: which line it belongs to is unknowable, so
    # both slots stay empty rather than risk attaching it to the wrong reference
    parsed = [ExtractedReference(title="Only One")]

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return type("R", (), {"parsed": parsed})()

    with caplog.at_level(logging.WARNING):
        out = extract_references(["ref one", "ref two"], CitationConfig(), client=FakeClient())
    assert "got 1 items for 2 input" in caplog.text
    assert out == [{}, {}]


def _numbered_client(skip: set[str], calls: list[list[str]], fail_retry: bool = False):
    """Fake Gemini that numbers its rows like the real one and skips titles in *skip*
    on the first call only (the live failure: one line dropped from a 50-line batch)."""
    import re as _re

    ref_line = _re.compile(r"^(\d+)\. (ref \d+)$")

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                lines = [
                    (int(m.group(1)), m.group(2))
                    for ln in contents.splitlines()
                    if (m := ref_line.match(ln))
                ]
                calls.append([t for _, t in lines])
                if len(calls) > 1 and fail_retry:
                    raise ValueError("retry broke")
                rows = [
                    ExtractedReference(line=n, title=t)
                    for n, t in lines
                    if len(calls) > 1 or t not in skip
                ]
                return type("R", (), {"parsed": rows})()

    return FakeClient()


def test_skipped_line_does_not_shift_later_references_and_is_retried():
    calls: list[list[str]] = []
    raws = [f"ref {i}" for i in range(5)]
    out = extract_references(raws, CitationConfig(), client=_numbered_client({"ref 1"}, calls))

    assert [item.get("title") for item in out] == raws  # nothing shifted
    assert calls == [raws, ["ref 1"]]  # the retry asked for the skipped line only
    assert all("line" not in item for item in out)  # alignment detail never leaks out


def test_failed_retry_keeps_the_batch_and_leaves_only_the_skipped_line_empty(caplog):
    calls: list[list[str]] = []
    raws = [f"ref {i}" for i in range(4)]
    with caplog.at_level(logging.WARNING):
        out = extract_references(
            raws, CitationConfig(), client=_numbered_client({"ref 2"}, calls, fail_retry=True)
        )

    assert [item.get("title") for item in out] == ["ref 0", "ref 1", None, "ref 3"]
    assert "retry of 1 line(s) failed" in caplog.text


def test_align_by_line_ignores_out_of_range_and_duplicate_lines():
    from paper_refinery.citation_extraction import _align_by_line

    rows = [
        ExtractedReference(line=2, title="B"),
        ExtractedReference(line=2, title="B again"),
        ExtractedReference(line=9, title="nowhere"),
        ExtractedReference(line=1, title="A"),
    ]
    assert [item.get("title") for item in _align_by_line(rows, ["A a", "B b", "C c"])] == [
        "A",
        "B",
        None,
    ]


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
    out = extract_references(["[1] Paper A, 2020."], cfg, client=FakeClient())
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


def test_extract_references_batches_and_concatenates_in_order():
    # the live Ferziger failure: 495 refs in one call overran the output-token cap ->
    # unparseable JSON -> 0 extracted. Batching keeps each call small; the batches must
    # re-concatenate in the original reference order.
    import re as _re

    seen_batches = []
    # only the appended reference-listing lines ("N. ref M"), never _PROMPT's own
    # numbered instruction lines
    ref_line = _re.compile(r"^\d+\. (ref \d+)$")

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                titles = [m.group(1) for ln in contents.splitlines() if (m := ref_line.match(ln))]
                seen_batches.append(titles)
                return type("R", (), {"parsed": [ExtractedReference(title=t) for t in titles]})()

    raws = [f"ref {i}" for i in range(10)]
    cfg = CitationConfig(extract_batch_size=3, max_workers=4)
    out = extract_references(raws, cfg, client=FakeClient())

    assert [item["title"] for item in out] == raws  # order preserved end to end
    assert len(seen_batches) == 4  # 10 refs / batch 3 -> 3+3+3+1
    assert sorted(len(b) for b in seen_batches) == [1, 3, 3, 3]


def test_extract_references_batch_failure_degrades_that_batch_only():
    import re as _re

    ref_line = _re.compile(r"^\d+\. (ref \d+)$")

    # one bad batch (unparseable response) must pad to {} for its slots without
    # misaligning the good batches around it
    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                titles = [m.group(1) for ln in contents.splitlines() if (m := ref_line.match(ln))]
                if "ref 3" in titles:  # the second batch (indices 3,4,5) returns junk
                    return type("R", (), {"parsed": None})()
                return type("R", (), {"parsed": [ExtractedReference(title=t) for t in titles]})()

    raws = [f"ref {i}" for i in range(6)]
    cfg = CitationConfig(extract_batch_size=3, max_workers=4)
    out = extract_references(raws, cfg, client=FakeClient())

    assert [item.get("title") for item in out] == [
        "ref 0",
        "ref 1",
        "ref 2",
        None,  # batch 2 failed -> {} for each of its three slots, still positioned right
        None,
        None,
    ]


def test_extract_references_batch_that_raises_degrades_that_batch_only(caplog):
    # a batch raising (non-retryable error, or exhausted retries) must not abort the
    # whole extraction -- degrade just that batch to {}, keeping the others. Same
    # "one bad unit never fails the whole" pattern as enrich.py.
    import re as _re

    ref_line = _re.compile(r"^\d+\. (ref \d+)$")

    class FakeAuthError(Exception):
        code = 401  # non-retryable per retry.py

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                titles = [m.group(1) for ln in contents.splitlines() if (m := ref_line.match(ln))]
                if "ref 3" in titles:  # the second batch raises instead of returning
                    raise FakeAuthError("boom")
                return type("R", (), {"parsed": [ExtractedReference(title=t) for t in titles]})()

    raws = [f"ref {i}" for i in range(6)]
    cfg = CitationConfig(extract_batch_size=3, max_workers=4, retry_attempts=1)
    with caplog.at_level(logging.WARNING):
        out = extract_references(raws, cfg, client=FakeClient())

    assert "a batch of 3 references failed" in caplog.text
    assert [item.get("title") for item in out] == [
        "ref 0",
        "ref 1",
        "ref 2",
        None,  # the raising batch -> {} x3, others unaffected
        None,
        None,
    ]


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


def test_self_numbered_rows_after_a_skip_are_caught_by_the_title_check():
    # the model skips "Beta" but numbers its rows 1,2,3 by its own count, so line
    # numbers alone would shift Gamma and Delta; the title check drops those two and
    # the retry of the three empty lines fills them correctly
    raws = ["[1] Alpha networks.", "[2] Beta filters.", "[3] Gamma control.", "[4] Delta flows."]
    responses = [
        [
            ExtractedReference(line=1, title="Alpha networks"),
            ExtractedReference(line=2, title="Gamma control"),
            ExtractedReference(line=3, title="Delta flows"),
        ],
        [
            ExtractedReference(line=1, title="Beta filters"),
            ExtractedReference(line=2, title="Gamma control"),
            ExtractedReference(line=3, title="Delta flows"),
        ],
    ]

    class FakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return type("R", (), {"parsed": responses.pop(0)})()

    out = extract_references(raws, CitationConfig(), client=FakeClient())
    assert [item.get("title") for item in out] == [
        "Alpha networks",
        "Beta filters",
        "Gamma control",
        "Delta flows",
    ]
