"""Tests for the shared retry policy (what's transient, how backoff behaves)."""

from __future__ import annotations

import pytest

from paper_refinery.retry import call_with_backoff, is_retryable


class _CodedError(Exception):
    """Stand-in for google-genai's APIError, which carries HTTP status on .code."""

    def __init__(self, code: int):
        super().__init__(f"http {code}")
        self.code = code


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [429, 500, 503])
def test_retryable_status_codes(code):
    assert is_retryable(_CodedError(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_non_retryable_status_codes(code):
    assert not is_retryable(_CodedError(code))


def test_transport_errors_are_retryable():
    assert is_retryable(ConnectionError("reset"))
    assert is_retryable(TimeoutError("timed out"))


def test_plain_exceptions_are_not_retryable():
    assert not is_retryable(ValueError("bad input"))
    assert not is_retryable(RuntimeError("bug"))


def test_status_code_attribute_also_recognized():
    # requests/httpx-style exceptions carry .status_code instead of .code
    exc = Exception("boom")
    exc.status_code = 503
    assert is_retryable(exc)


# ---------------------------------------------------------------------------
# call_with_backoff
# ---------------------------------------------------------------------------


def test_backoff_retries_transient_then_succeeds():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise _CodedError(429)
        return "ok"

    assert call_with_backoff(fn, attempts=4, base_delay=0.0) == "ok"
    assert len(calls) == 3


def test_backoff_raises_after_final_attempt():
    calls = []

    def fn():
        calls.append(1)
        raise _CodedError(500)

    with pytest.raises(_CodedError):
        call_with_backoff(fn, attempts=3, base_delay=0.0)
    assert len(calls) == 3


def test_backoff_fails_fast_on_non_retryable():
    calls = []

    def fn():
        calls.append(1)
        raise _CodedError(401)

    with pytest.raises(_CodedError):
        call_with_backoff(fn, attempts=4, base_delay=10.0)  # delay must never be slept
    assert len(calls) == 1
