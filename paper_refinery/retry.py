"""Retry policy shared by the Gemini call sites (figures.py, citation_extraction.py).

One place for two decisions that used to be duplicated (and wrong) at each call site:
what counts as transient, and how to back off. The old loops retried *every* exception,
so a bad API key (401) burned all attempts x exponential backoff (~30s of silence)
before surfacing an error that no amount of waiting could fix.
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def is_retryable(exc: Exception) -> bool:
    """True for transient failures worth a backoff retry.

    Retryable: rate limits (429), server-side errors (5xx), and transport-level drops
    (connection reset, timeout). Everything else -- 400 bad request, 401/403 bad or
    missing key, 404 wrong model name -- fails immediately. google-genai's APIError
    carries the HTTP status on ``.code``; the ``status_code`` fallback covers
    requests/httpx-style exceptions from any future non-Gemini call site.
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def call_with_backoff(fn: Callable[[], T], attempts: int, base_delay: float) -> T:
    """Call ``fn`` with exponential backoff (base, 2x, 4x, ...) on retryable errors.

    Non-retryable errors and the final attempt's error propagate unchanged.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1 or not is_retryable(exc):
                raise
            time.sleep(base_delay * (2**i))
    raise AssertionError("unreachable: attempts >= 1 always returns or raises")
