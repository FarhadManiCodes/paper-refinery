"""Talk to CrossRef, Semantic Scholar, and OpenAlex; normalize their very different
response shapes into one common candidate shape.

Plain Python: no LLM, stdlib HTTP + urllib only. Response shapes were confirmed live
(2026-07-03) against all three providers -- see the field notes on each normalizer.
citation_resolution.py owns what to trust from a provider's answer (acceptance bar,
merge semantics); this module only knows how to ask and how to read the reply.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .config import CitationConfig
from .disk_cache import cache_path, read_json, write_json
from .retry import call_with_backoff

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


def _cache_path(url: str, cfg: CitationConfig) -> Path | None:
    return cache_path(cfg.api_cache_dir, hashlib.sha256(url.encode()).hexdigest())


def _get_json(
    url: str, cfg: CitationConfig, headers: dict | None = None, before_fetch=None
) -> dict | None:
    """GET ``url`` as JSON with an on-disk cache and retry/backoff; ``None`` (never
    raising) on final failure.

    Cache first: every successful response is stored under a sha256-of-URL key, so
    iterating on matching logic replays at zero API cost -- the keyless providers are a
    shared resource, use them once. Failures are deliberately NOT cached (a transient
    error must not stick). A reference that can't be verified just stays unverified --
    one provider being down must not fail the whole run. urllib's HTTPError carries the
    status on ``.code``, so retry.is_retryable's 429/5xx-vs-4xx split applies as-is.
    """
    cache = _cache_path(url, cfg)
    if cache:
        cached = read_json(cache)  # None on a cache miss or a corrupt entry alike
        if isinstance(cached, dict):
            return cached

    base_headers = {"User-Agent": _user_agent(cfg)}
    base_headers.update(headers or {})

    def fetch():
        req = urllib.request.Request(url, headers=base_headers)
        with urllib.request.urlopen(req, timeout=cfg.request_timeout_s) as resp:
            return json.loads(resp.read())

    if before_fetch is not None:
        before_fetch()  # e.g. the S2 throttle -- only on a real fetch, never a cache hit
    try:
        data = call_with_backoff(fetch, cfg.api_retry_attempts, cfg.api_retry_base_delay)
    except Exception as exc:
        logger.debug("provider fetch failed for %s: %r", url, exc)
        return None
    if cache and data is not None:
        write_json(cache, data)
    return data


def _user_agent(cfg: CitationConfig) -> str:
    ua = "paper-refinery/0.1"
    if cfg.mailto:
        ua += f" (mailto:{cfg.mailto})"  # CrossRef/OpenAlex polite pool
    return ua


def _s2_headers(cfg: CitationConfig) -> dict:
    key = os.environ.get(cfg.s2_api_key_env or "", "")
    return {"x-api-key": key} if key else {}


# ---------------------------------------------------------------------------
# Provider calls (one GET each)
# ---------------------------------------------------------------------------

_S2_FIELDS = "title,year,abstract,authors,externalIds,publicationTypes"


def s2_by_doi(doi: str, cfg: CitationConfig) -> dict | None:
    url = f"{cfg.s2_api_base}/paper/DOI:{urllib.parse.quote(doi)}?fields={_S2_FIELDS}"
    return _get_json(url, cfg, headers=_s2_headers(cfg), before_fetch=lambda: _s2_throttle(cfg))


def s2_search(title: str, cfg: CitationConfig) -> dict | None:
    url = (
        f"{cfg.s2_api_base}/paper/search"
        f"?query={urllib.parse.quote(title)}&fields={_S2_FIELDS}&limit=1"
    )
    data = _get_json(url, cfg, headers=_s2_headers(cfg), before_fetch=lambda: _s2_throttle(cfg))
    hits = (data or {}).get("data") or []
    return hits[0] if hits else None


def crossref_search(title: str, cfg: CitationConfig) -> dict | None:
    url = f"{cfg.crossref_api_base}/works?query.bibliographic={urllib.parse.quote(title)}&rows=1"
    if cfg.mailto:
        url += f"&mailto={urllib.parse.quote(cfg.mailto)}"
    data = _get_json(url, cfg)
    items = ((data or {}).get("message") or {}).get("items") or []
    return items[0] if items else None


def openalex_search(title: str, cfg: CitationConfig) -> dict | None:
    url = f"{cfg.openalex_api_base}/works?search={urllib.parse.quote(title)}&per-page=1"
    if cfg.mailto:
        url += f"&mailto={urllib.parse.quote(cfg.mailto)}"
    data = _get_json(url, cfg)
    results = (data or {}).get("results") or []
    return results[0] if results else None


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


def _split_full_name(full: str | None) -> dict | None:
    """Best-effort {family, given} from a display name ("Steven L. Brunton").

    S2 and OpenAlex give only full display names (CrossRef alone has proper
    family/given fields); last-token-as-family is the standard approximation and is
    what citekeys need. Multi-word surnames ("van der Berg") come out imperfect --
    the diff report surfaces author replacements so such cases are visible.
    """
    tokens = (full or "").split()
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
