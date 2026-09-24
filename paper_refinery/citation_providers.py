"""Talk to CrossRef, Semantic Scholar, and OpenAlex; normalize their very different
response shapes into one common candidate shape.

Plain Python: no LLM, stdlib HTTP + urllib only. Response shapes were confirmed live
(2026-07-03) against all three providers -- see the field notes on each normalizer.
citation_resolution.py owns what to trust from a provider's answer (acceptance bar,
merge semantics); this module only knows how to ask and how to read the reply.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from .config import CitationConfig
from .disk_cache import cache_path, read_json, write_json
from .retry import call_with_backoff, is_retryable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DOI extraction (from the RAW OCR text, not the extractor's guess)
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)


def extract_doi(text: str) -> str | None:
    """First DOI printed in ``text``, with OCR-glued trailing punctuation stripped."""
    m = _DOI_RE.search(text or "")
    if not m:
        return None
    return m.group(0).rstrip(".,;:)]}\"'") or None


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

_S2_THROTTLE_LOCK = threading.Lock()
_s2_last_call = 0.0


def _s2_throttle(cfg: CitationConfig) -> None:
    """Serialize S2 calls to >= cfg.s2_min_interval_s apart, across worker threads.

    Unauthenticated S2 hard-rate-limits bursts (confirmed live: back-to-back calls
    429). Retry-with-backoff would survive that, but pacing up front avoids burning
    retry budget on a limit we know about.
    """
    global _s2_last_call
    with _S2_THROTTLE_LOCK:
        wait = _s2_last_call + cfg.s2_min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _s2_last_call = time.monotonic()


_MAILTO_PARAM_RE = re.compile(r"&mailto=[^&]*")


def _mailto(cfg: CitationConfig, provider: str) -> str:
    """The contact for ``provider`` ("crossref" or "openalex"), or "" if it gets none."""
    if provider not in cfg.mailto_providers:
        return ""
    return cfg.mailto or os.environ.get(cfg.mailto_env or "", "")


def _without_mailto(url: str) -> str:
    """The URL as cached and logged: the contact address is not part of a lookup's identity."""
    return _MAILTO_PARAM_RE.sub("", url)


class _KeepCredentialsOnHost(urllib.request.HTTPRedirectHandler):
    """urllib copies every header, Authorization included, to a redirect target -- even on
    another host. A keyed request drops the key when a redirect leaves its host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            if urllib.parse.urlsplit(newurl).hostname != old_host:
                new.remove_header("Authorization")
        return new


_KEYED_OPENER = urllib.request.build_opener(_KeepCredentialsOnHost)
_REJECTED_KEY_WARNED: set[str] = set()


def _warn_once_if_key_rejected(url: str, exc: Exception, cfg: CitationConfig) -> None:
    """A revoked or mistyped key fails every lookup fast and silently (401 is not
    retried): say so once per process instead of quietly losing a provider."""
    provider = _provider_of(url, cfg)
    code = getattr(exc, "code", None)
    rejected = provider == "openalex" and code in (401, 403) and _auth_headers(url, cfg)
    if rejected and provider not in _REJECTED_KEY_WARNED:
        _REJECTED_KEY_WARNED.add(provider)
        logger.warning(
            "OpenAlex rejected the API key in $%s (HTTP %s): OpenAlex lookups will fail",
            cfg.openalex_api_key_env,
            code,
        )


def _auth_headers(url: str, cfg: CitationConfig) -> dict:
    """OpenAlex's API key as the bearer header its docs recommend -- never in the URL,
    so it cannot reach the cache key, a log line, or any other provider."""
    key = os.environ.get(cfg.openalex_api_key_env or "", "")
    if key and _provider_of(url, cfg) == "openalex":
        return {"Authorization": f"Bearer {key}"}
    return {}


def _provider_of(url: str, cfg: CitationConfig) -> str | None:
    if url.startswith(cfg.crossref_api_base):
        return "crossref"
    if url.startswith(cfg.openalex_api_base):
        return "openalex"
    return None


def _provider_label(url: str, cfg: CitationConfig) -> str:
    """Provider name for logs and counts; unlike ``_provider_of`` (which decides who gets
    the key and contact address) it also names S2, and falls back to the host."""
    if url.startswith(cfg.s2_api_base):
        return "semanticscholar"
    return _provider_of(url, cfg) or urllib.parse.urlsplit(url).hostname or "unknown"


class ProviderStats:
    """Process-wide, thread-safe lookup counts per provider: ``ok`` and ``cached`` answers,
    ``missing`` (a 404: the provider does not know that id -- an answer, not a failure),
    ``failed`` lookups (retries exhausted), and each failed *attempt* by class (``429``,
    ``5xx``, ``4xx``, ``timeout``, ``error``). Callers diff two ``snapshot()``s to report
    one document; documents refined concurrently share the counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Counter[tuple[str, str]] = Counter()

    def add(self, provider: str, outcome: str) -> None:
        with self._lock:
            self._counts[(provider, outcome)] += 1

    def snapshot(self) -> Counter[tuple[str, str]]:
        with self._lock:
            return Counter(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


PROVIDER_STATS = ProviderStats()

# A provider whose ordinary lookups keep ending in 429 is skipped for a while instead of
# making every later lookup sit through its backoff (keyless S2, 2026-09-24).
_COOLDOWN_AFTER = 3  # consecutive lookups exhausted on 429
_cooldown_lock = threading.Lock()
_rate_limited_streak: Counter[str] = Counter()
_cooling_until: dict[str, float] = {}


def _cooling_down(provider: str) -> bool:
    with _cooldown_lock:
        until = _cooling_until.get(provider)
        if until is None:
            return False
        if time.monotonic() < until:
            return True
        del _cooling_until[provider]
    logger.info("%s: cooldown over; asking it again", provider)
    return False


def _after_lookup(provider: str, cfg: CitationConfig, rate_limited: bool) -> None:
    """Track consecutive 429-exhausted lookups per provider; start a cooldown at the
    threshold. Any other outcome (success, 404, other error) resets the streak."""
    with _cooldown_lock:
        if not rate_limited:
            _rate_limited_streak[provider] = 0
            return
        _rate_limited_streak[provider] += 1
        if _rate_limited_streak[provider] < _COOLDOWN_AFTER or cfg.provider_cooldown_s <= 0:
            return
        _rate_limited_streak[provider] = 0
        _cooling_until[provider] = time.monotonic() + cfg.provider_cooldown_s
    logger.warning(
        "%s: rate-limited on %d lookups in a row; skipping it for %.0f min",
        provider,
        _COOLDOWN_AFTER,
        cfg.provider_cooldown_s / 60,
    )


def reset_provider_state() -> None:
    """Clear counts, warnings and cooldowns (tests, and long-lived callers)."""
    PROVIDER_STATS.reset()
    with _cooldown_lock:
        _rate_limited_streak.clear()
        _cooling_until.clear()
    with _IMPATIENT_LOCK:
        _IMPATIENT.clear()
    _ERROR_CLASSES_WARNED.clear()


_ERROR_CLASSES_WARNED: set[tuple[str, str]] = set()
_ERROR_WARN_LOCK = threading.Lock()


def _error_class(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code == 404:
            return "missing"
        return "429" if code == 429 else "5xx" if code >= 500 else "4xx"
    reason = getattr(exc, "reason", exc)
    if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
        return "timeout"
    return "error"


def _secret_values(cfg: CitationConfig) -> list[str]:
    values = [cfg.mailto, os.environ.get(cfg.mailto_env or "", "")]
    values += [os.environ.get(cfg.openalex_api_key_env or "", "")]
    values += [os.environ.get(cfg.s2_api_key_env or "", "")]
    out = []
    for v in values:
        if v and len(v) >= 4:
            out += [v, urllib.parse.quote(v), urllib.parse.quote(v, safe="")]
    return sorted(set(out), key=len, reverse=True)


def _scrub(text: str, cfg: CitationConfig) -> str:
    for value in _secret_values(cfg):
        text = text.replace(value, "***")
    return text


_BODY_READ = 2048
_SECRET_MARGIN = 256  # longer than any contact address or key


def _error_detail(exc: BaseException, cfg: CitationConfig) -> str:
    """Status plus the provider's own message (OpenAlex's "Insufficient budget"), short.
    An error page may echo the request, so the contact address and keys are masked."""
    code = getattr(exc, "code", None)
    body = ""
    if code is not None and hasattr(exc, "read"):
        try:
            raw = exc.read(_BODY_READ)
        except Exception:
            raw = b""
        # a read that hit the limit may end inside a secret, which then would not match
        # the mask: drop that tail. Mask before shortening, for the same reason.
        if len(raw) == _BODY_READ:
            raw = raw[: _BODY_READ - _SECRET_MARGIN]
        body = _scrub(raw.decode("utf-8", "replace"), cfg)
    body = " ".join(body.split())[:160]
    head = f"HTTP {code}" if code is not None else type(exc).__name__
    return _scrub(f"{head}: {body}" if body else f"{head} ({exc})", cfg)


def _note_failed_attempt(url: str, exc: BaseException, cfg: CitationConfig) -> None:
    """Count every failed attempt, and warn the first time each (provider, class) occurs
    in this process -- rate limits and exhausted budgets used to show only at DEBUG."""
    provider, cls = _provider_label(url, cfg), _error_class(exc)
    PROVIDER_STATS.add(provider, cls)
    if cls == "missing":
        return  # a 404 is the provider's answer ("no such id"), not a failure
    keyed = bool(_auth_headers(url, cfg))
    if keyed and cls == "4xx" and getattr(exc, "code", None) in (401, 403):
        return  # _warn_once_if_key_rejected names this one
    with _ERROR_WARN_LOCK:
        if (provider, cls) in _ERROR_CLASSES_WARNED:
            return
        _ERROR_CLASSES_WARNED.add((provider, cls))
    detail = _error_detail(exc, cfg)
    if provider == "openalex" and cls == "429" and "budget" in detail.lower():
        logger.warning(
            "OpenAlex budget exhausted (%s): its lookups fail until the daily budget resets "
            "or credit is added; later ones are counted in the progress lines",
            detail,
        )
        return
    logger.warning(
        "%s lookup failed (%s) for %s; retryable errors are retried, and later %s "
        "failures from %s are counted in the progress lines",
        provider,
        detail,
        _scrub(_without_mailto(url), cfg),
        cls,
        provider,
    )


def _cache_path(url: str, cfg: CitationConfig) -> Path | None:
    key = _without_mailto(url)
    return cache_path(cfg.api_cache_dir, hashlib.sha256(key.encode()).hexdigest())


# Reference-list calls get cfg.bulk_retry_attempts, per provider, only while that provider's
# last list call did not exhaust its retries on a retryable error: a sustained outage then
# costs one long wait per process, not one per paper, and one provider succeeding does not
# re-arm another's wait. A successful list call from the provider restores its patience.
_IMPATIENT: set[str] = set()
_IMPATIENT_LOCK = threading.Lock()


def _list_attempts(cfg: CitationConfig, provider: str) -> int:
    if provider == "semanticscholar" and cfg.s2_retry_attempts > 0:
        return cfg.s2_retry_attempts  # asked once (or as set) everywhere, lists included
    with _IMPATIENT_LOCK:
        impatient = provider in _IMPATIENT
    return cfg.api_retry_attempts if impatient else cfg.bulk_retry_attempts


def _record_failure(url: str, exc: Exception, cfg: CitationConfig, attempts: int | None) -> None:
    """Book-keeping for a lookup whose retries are exhausted: counts, the key-rejected
    warning, and tripping the list-call breaker."""
    if _error_class(exc) != "missing":  # a 404 was already counted as an answer
        PROVIDER_STATS.add(_provider_label(url, cfg), "failed")
    if attempts is None:
        _after_lookup(_provider_label(url, cfg), cfg, rate_limited=_error_class(exc) == "429")
    _warn_once_if_key_rejected(url, exc, cfg)
    logger.debug("provider fetch failed for %s: %r", _without_mailto(url), exc)
    if attempts and attempts > cfg.api_retry_attempts and is_retryable(exc):
        with _IMPATIENT_LOCK:
            _IMPATIENT.add(_provider_label(url, cfg))
        logger.warning(
            "%s kept failing after %d attempts; its reference-list calls use the ordinary "
            "retry budget until one succeeds",
            _provider_label(url, cfg),
            attempts,
        )


def _ordinary_attempts(provider: str, cfg: CitationConfig) -> int:
    if provider == "semanticscholar" and cfg.s2_retry_attempts > 0:
        return cfg.s2_retry_attempts
    return cfg.api_retry_attempts


def _get_json(
    url: str,
    cfg: CitationConfig,
    headers: dict | None = None,
    before_fetch=None,
    attempts: int | None = None,
) -> dict | None:
    """GET ``url`` as JSON with an on-disk cache and retry/backoff; ``None`` (never
    raising) on final failure.

    Cache first: every successful response is stored under a sha256-of-URL key, so
    iterating on matching logic replays at zero API cost -- the keyless providers are a
    shared resource, use them once. Failures are deliberately NOT cached (a transient
    error must not stick). A reference that can't be verified just stays unverified --
    one provider being down must not fail the whole run. urllib's HTTPError carries the
    status on ``.code``, so retry.is_retryable's 429/5xx-vs-4xx split applies as-is.
    ``attempts`` overrides ``cfg.api_retry_attempts`` for calls worth waiting longer for.
    """
    cache = _cache_path(url, cfg)
    if cache:
        cached = read_json(cache)  # None on a cache miss or a corrupt entry alike
        if isinstance(cached, dict):
            PROVIDER_STATS.add(_provider_label(url, cfg), "cached")
            return cached

    provider = _provider_label(url, cfg)
    if _cooling_down(provider):  # list calls too: a cooling provider makes nobody wait
        PROVIDER_STATS.add(provider, "skipped")
        return None

    base_headers = {"User-Agent": _user_agent(cfg, url), **_auth_headers(url, cfg)}
    base_headers.update(headers or {})

    def fetch():
        req = urllib.request.Request(url, headers=base_headers)
        opener = _KEYED_OPENER.open if "Authorization" in base_headers else urllib.request.urlopen
        try:
            with opener(req, timeout=cfg.request_timeout_s) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            _note_failed_attempt(url, exc, cfg)
            raise

    if before_fetch is not None:
        before_fetch()  # e.g. the S2 throttle -- only on a real fetch, never a cache hit
    try:
        data = call_with_backoff(
            fetch, attempts or _ordinary_attempts(provider, cfg), cfg.api_retry_base_delay
        )
    except Exception as exc:
        _record_failure(url, exc, cfg, attempts)
        return None
    PROVIDER_STATS.add(provider, "ok")
    if attempts is None:
        _after_lookup(provider, cfg, rate_limited=False)
    else:  # a list call succeeded, whatever its budget
        with _IMPATIENT_LOCK:
            _IMPATIENT.discard(provider)
    if cache and data is not None:
        write_json(cache, data)
    return data


def _user_agent(cfg: CitationConfig, url: str) -> str:
    ua = "paper-refinery/0.1"
    if (provider := _provider_of(url, cfg)) and (mailto := _mailto(cfg, provider)):
        ua += f" (mailto:{mailto})"  # CrossRef/OpenAlex polite pool only
    return ua


def _s2_headers(cfg: CitationConfig) -> dict:
    key = os.environ.get(cfg.s2_api_key_env or "", "")
    return {"x-api-key": key} if key else {}


# ---------------------------------------------------------------------------
# Provider calls (one GET each)
# ---------------------------------------------------------------------------

_S2_FIELDS = "title,year,abstract,authors,externalIds,publicationTypes"


def openalex_by_doi(doi: str, cfg: CitationConfig) -> dict | None:
    """One OpenAlex work by DOI (an exact lookup), in ``normalize_openalex``'s input shape."""
    url = f"{cfg.openalex_api_base}/works/doi:{urllib.parse.quote(doi)}?select={_OPENALEX_FIELDS}"
    if mailto := _mailto(cfg, "openalex"):
        url += f"&mailto={urllib.parse.quote(mailto)}"
    return _get_json(url, cfg)


def s2_by_doi(doi: str, cfg: CitationConfig) -> dict | None:
    url = f"{cfg.s2_api_base}/paper/DOI:{urllib.parse.quote(doi)}?fields={_S2_FIELDS}"
    return _get_json(url, cfg, headers=_s2_headers(cfg), before_fetch=lambda: _s2_throttle(cfg))


def s2_search_more(title: str, cfg: CitationConfig, n: int) -> list[dict]:
    """Top ``n`` title-search hits, best first. With ``n=1`` the URL is exactly the one
    ``s2_search`` has always used, so its cached responses stay valid."""
    url = (
        f"{cfg.s2_api_base}/paper/search"
        f"?query={urllib.parse.quote(title)}&fields={_S2_FIELDS}&limit={n}"
    )
    data = _get_json(url, cfg, headers=_s2_headers(cfg), before_fetch=lambda: _s2_throttle(cfg))
    return (data or {}).get("data") or []


def s2_search(title: str, cfg: CitationConfig) -> dict | None:
    hits = s2_search_more(title, cfg, 1)
    return hits[0] if hits else None


_SOURCE_FIELDS = "title,year,authors,externalIds,publicationTypes"


def s2_paper_id(
    cfg: CitationConfig,
    *,
    doi: str | None = None,
    arxiv: str | None = None,
    title: str | None = None,
) -> tuple[str, dict] | None:
    """Resolve a SOURCE paper to ``(paper_id, candidate)``.

    An external id (``doi`` or ``arxiv``) uses S2's exact graph lookup -- no search endpoint,
    so it dodges the keyless-search 429s -- and is trusted by construction. Otherwise a
    one-hit ``title`` search: ``candidate`` (``normalize_s2`` of the hit -- title/year/authors/
    doi) lets the caller CORROBORATE a title match on the same acceptance bar it uses for
    references, since a title alone can hit the wrong paper. ``None`` on a miss/fetch failure.
    """
    if doi:
        ref = f"DOI:{urllib.parse.quote(doi)}"
    elif arxiv:
        ref = f"ARXIV:{urllib.parse.quote(arxiv)}"
    else:
        ref = None
    if ref:
        data = _get_json(
            f"{cfg.s2_api_base}/paper/{ref}?fields={_SOURCE_FIELDS}",
            cfg,
            headers=_s2_headers(cfg),
            before_fetch=lambda: _s2_throttle(cfg),
            attempts=_list_attempts(cfg, "semanticscholar"),
        )
        cand = normalize_s2(data)
        return (data["paperId"], cand) if (data and data.get("paperId") and cand) else None
    if title:
        data = _get_json(
            f"{cfg.s2_api_base}/paper/search"
            f"?query={urllib.parse.quote(title)}&fields={_SOURCE_FIELDS}&limit=1",
            cfg,
            headers=_s2_headers(cfg),
            before_fetch=lambda: _s2_throttle(cfg),
        )
        hits = (data or {}).get("data") or []
        if hits and hits[0].get("paperId"):
            cand = normalize_s2(hits[0])
            if cand:
                return hits[0]["paperId"], cand
    return None


def s2_references(paper_id: str, cfg: CitationConfig) -> list[dict] | None:
    """The source paper's cited references as normalized candidates, in ONE bulk call.

    Each entry is ``normalize_s2``'d (title/year/doi/abstract/authors/type), confirmed live
    to include the abstract, so no per-entry follow-up is needed. ``None`` on fetch failure.
    The list is S2's reference set for the paper -- **not** in printed-bibliography order
    (confirmed live), so callers match by content, never by position. Capped at 1000 refs
    (papers beyond that are vanishingly rare; the overflow just falls back per-entry).
    """
    url = (
        f"{cfg.s2_api_base}/paper/{urllib.parse.quote(paper_id)}/references"
        f"?fields={_S2_FIELDS}&limit=1000"
    )
    data = _get_json(
        url,
        cfg,
        headers=_s2_headers(cfg),
        before_fetch=lambda: _s2_throttle(cfg),
        attempts=_list_attempts(cfg, "semanticscholar"),
    )
    if data is None:
        return None
    candidates: list[dict] = []
    for item in data.get("data") or []:
        cand = normalize_s2((item or {}).get("citedPaper"))
        if cand:
            candidates.append(cand)
    return candidates


def crossref_search_more(title: str, cfg: CitationConfig, n: int) -> list[dict]:
    """Top ``n`` hits, best first; ``n=1`` reproduces ``crossref_search``'s cached URL."""
    q = urllib.parse.quote(title)
    url = f"{cfg.crossref_api_base}/works?query.bibliographic={q}&rows={n}"
    if mailto := _mailto(cfg, "crossref"):
        url += f"&mailto={urllib.parse.quote(mailto)}"
    data = _get_json(url, cfg)
    return ((data or {}).get("message") or {}).get("items") or []


def crossref_search(title: str, cfg: CitationConfig) -> dict | None:
    items = crossref_search_more(title, cfg, 1)
    return items[0] if items else None


_OPENALEX_WILDCARD_RE = re.compile(r"[?*]")


def _openalex_query(title: str) -> str:
    """A title as an OpenAlex search: ``?`` and ``*`` are wildcards there, and the default
    (stemmed) search rejects them with HTTP 400 ("Robust principal component analysis?",
    2026-09-24), so they become spaces. A title without them keeps its cached URL."""
    if not _OPENALEX_WILDCARD_RE.search(title):
        return title
    return " ".join(_OPENALEX_WILDCARD_RE.sub(" ", title).split())


def openalex_search_more(title: str, cfg: CitationConfig, n: int) -> list[dict]:
    """Top ``n`` hits, best first; ``n=1`` reproduces ``openalex_search``'s cached URL."""
    query = urllib.parse.quote(_openalex_query(title))
    url = f"{cfg.openalex_api_base}/works?search={query}&per-page={n}"
    if mailto := _mailto(cfg, "openalex"):
        url += f"&mailto={urllib.parse.quote(mailto)}"
    data = _get_json(url, cfg)
    return (data or {}).get("results") or []


def openalex_search(title: str, cfg: CitationConfig) -> dict | None:
    results = openalex_search_more(title, cfg, 1)
    return results[0] if results else None


_OPENALEX_FIELDS = "id,display_name,publication_year,authorships,doi,abstract_inverted_index,type"


_OPENALEX_SOURCE_FIELDS = "id,display_name,publication_year,authorships,doi,type,referenced_works"


def openalex_source(
    cfg: CitationConfig, doi: str | None = None, title: str | None = None
) -> tuple[dict, list[str]] | None:
    """Identify a SOURCE work in OpenAlex -- exactly by ``doi``, else by its top ``title``
    search hit (the caller corroborates that one) -- and return ``(candidate, ids)``: the
    normalized record and the OpenAlex ids of the works it cites. ``None`` on a miss or a
    failed fetch."""
    # keyless OpenAlex answers 429 for lack of budget, which waiting never fixes
    attempts = (
        _list_attempts(cfg, "openalex") if os.environ.get(cfg.openalex_api_key_env or "") else None
    )
    if doi:
        url = (
            f"{cfg.openalex_api_base}/works/doi:{urllib.parse.quote(doi)}"
            f"?select={_OPENALEX_SOURCE_FIELDS}"
        )
    elif title:
        url = (
            f"{cfg.openalex_api_base}/works?search={urllib.parse.quote(_openalex_query(title))}"
            f"&per-page=1&select={_OPENALEX_SOURCE_FIELDS}"
        )
    else:
        return None
    if mailto := _mailto(cfg, "openalex"):
        url += f"&mailto={urllib.parse.quote(mailto)}"
    data = _get_json(url, cfg, attempts=attempts)
    work = data if doi else ((data or {}).get("results") or [None])[0]
    cand = normalize_openalex(work)
    if not cand:
        return None
    ids = [w.rsplit("/", 1)[-1] for w in (work or {}).get("referenced_works") or []]
    return cand, ids


def openalex_hydrate(ids: list[str], cfg: CitationConfig) -> list[dict]:
    """The works behind OpenAlex ``ids`` as normalized candidates, in batched calls
    (<=100 ids each, OpenAlex's OR-filter cap), abstracts reconstructed."""
    attempts = (
        _list_attempts(cfg, "openalex") if os.environ.get(cfg.openalex_api_key_env or "") else None
    )
    candidates: list[dict] = []
    for id_batch in itertools.batched(ids, 100):
        batch = "|".join(id_batch)
        url = (
            f"{cfg.openalex_api_base}/works?filter=openalex_id:{batch}"
            f"&per-page=100&select={_OPENALEX_FIELDS}"
        )
        if mailto := _mailto(cfg, "openalex"):
            url += f"&mailto={urllib.parse.quote(mailto)}"
        data = _get_json(url, cfg, attempts=attempts)
        for work in (data or {}).get("results") or []:
            cand = normalize_openalex(work)
            if cand:
                candidates.append(cand)
    return candidates


def openalex_references(doi: str, cfg: CitationConfig) -> list[dict] | None:
    """The source paper's references from OpenAlex (``referenced_works``, hydrated) as
    normalized candidates -- a bulk source for when S2's list is publisher-elided
    (ASME/IEEE...) or sparse. ``None`` when the work isn't found or lists no references.
    """
    found = openalex_source(cfg, doi=doi)
    return (openalex_hydrate(found[1], cfg) or None) if found and found[1] else None


# ---------------------------------------------------------------------------
# Normalizers: each provider's shape -> one common candidate shape
# ---------------------------------------------------------------------------


def reconstruct_openalex_abstract(inverted_index: dict | None) -> str | None:
    """OpenAlex returns ``abstract_inverted_index: {word: [positions...]}`` (a
    copyright-driven design), not plain text -- place each word at its position(s)
    and join in order."""
    if not inverted_index:
        return None
    placed = [(pos, word) for word, positions in inverted_index.items() for pos in positions]
    return " ".join(word for _, word in sorted(placed)) or None


_JATS_TAG_RE = re.compile(r"<[^>]+>")


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _split_full_name(full: str | None) -> dict | None:
    """Best-effort {family, given} from a display name ("Steven L. Brunton").

    S2 and OpenAlex give only full display names (CrossRef alone has proper
    family/given fields); last-token-as-family is the standard approximation and is
    what citekeys need. Multi-word surnames ("van der Berg") come out imperfect --
    the diff report surfaces author replacements so such cases are visible.
    """
    tokens = (full or "").split()
    # a generational suffix is not the surname: "Martin L. King Jr." -> King
    while len(tokens) > 1 and tokens[-1].rstrip(".,").lower() in _NAME_SUFFIXES:
        tokens.pop()
    if not tokens:
        return None
    return {"family": tokens[-1], "given": " ".join(tokens[:-1]) or None}


def normalize_s2(paper: dict | None) -> dict | None:
    # confirmed live: `abstract` is plain text; DOI under externalIds; publicationTypes
    # is a list like ["JournalArticle"] (or null); authors carry only a full `name`
    if not paper:
        return None
    types = paper.get("publicationTypes") or []
    authors = [
        a for a in (_split_full_name(p.get("name")) for p in paper.get("authors") or []) if a
    ]
    return {
        "title": paper.get("title"),
        "year": paper.get("year"),
        "doi": (paper.get("externalIds") or {}).get("DOI"),
        "abstract": paper.get("abstract"),
        "authors": authors,
        "source": "semanticscholar",
        "provider_type": types[0] if types else None,
    }


def normalize_crossref(item: dict | None) -> dict | None:
    # confirmed live: `title` is a LIST; year sits at issued.date-parts[0][0];
    # `abstract` (when present at all) is JATS XML -- tags stripped here. Authors are
    # the one clean family/given source of the three providers.
    if not item:
        return None
    titles = item.get("title") or []
    date_parts = (item.get("issued") or {}).get("date-parts") or []
    year = date_parts[0][0] if date_parts and date_parts[0] else None
    abstract = item.get("abstract")
    if abstract:
        abstract = _JATS_TAG_RE.sub("", abstract).strip() or None
    authors = [
        {"family": a["family"], "given": a.get("given")}
        for a in item.get("author") or []
        if a.get("family")  # organizations carry `name` instead; skip them
    ]
    return {
        "title": titles[0] if titles else None,
        "year": year,
        "doi": item.get("DOI"),
        "abstract": abstract,
        "authors": authors,
        "source": "crossref",
        "provider_type": item.get("type"),
    }


def normalize_openalex(work: dict | None) -> dict | None:
    # confirmed live: `doi` is a full https://doi.org/ URL; abstract only as inverted
    # index; authors under authorships[].author.display_name
    if not work:
        return None
    doi = (work.get("doi") or "").removeprefix("https://doi.org/") or None
    authors = [
        a
        for a in (
            _split_full_name((s.get("author") or {}).get("display_name"))
            for s in work.get("authorships") or []
        )
        if a
    ]
    return {
        "title": work.get("display_name") or work.get("title"),
        "year": work.get("publication_year"),
        "doi": doi,
        "abstract": reconstruct_openalex_abstract(work.get("abstract_inverted_index")),
        "authors": authors,
        "source": "openalex",
        "provider_type": work.get("type"),
    }
