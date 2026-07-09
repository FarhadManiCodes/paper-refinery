"""Verify and complete extracted references against real bibliographic APIs (layer 2/3).

Takes citation_extraction.py's rough per-reference guesses and checks each against
CrossRef -> Semantic Scholar -> OpenAlex (in that order; CrossRef leads because its
`issued` date is the published record's own, while S2 blends preprint+published): a DOI
printed in the raw OCR text short-circuits to an exact S2 lookup; otherwise a title
search must clear a composite
acceptance bar (title similarity AND year tolerance -- similarity alone is a
false-positive risk, two unrelated papers can share near-identical titles). An accepted
match's fields overwrite the guess wherever the provider actually has data; an
unverifiable reference is returned unverified with the guess kept intact -- never
dropped, never silently trusted.

The actual provider calls (HTTP, caching, response normalization) live in
``citation_providers.py``; this module owns what to trust from an answer, not how to
get one.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TypedDict

from .citation_providers import (
    crossref_search,
    extract_doi,
    normalize_crossref,
    normalize_openalex,
    normalize_s2,
    openalex_references,
    openalex_search,
    s2_by_doi,
    s2_paper_id,
    s2_references,
    s2_search,
)
from .config import CitationConfig
from .references import RawReference
from .text_utils import fold_name, leading_number


@dataclass(slots=True)
class SourcePaper:
    """What we know about the paper being processed, for the S2 bulk-references fast-path.

    An external id (``doi``/``arxiv``) resolves the source exactly (trusted); ``title`` (+
    ``year``/``authors`` when available -- papis fills them from info.yaml, the CLI has the
    OCR'd title) resolves it by a corroborated search. All optional: an empty/None source
    just disables the fast-path and everything runs the per-entry path as before.
    """

    doi: str | None = None
    arxiv: str | None = None
    title: str | None = None
    year: int | None = None
    authors: list[dict] | None = None
    references: list[dict] | None = None  # the paper's own bibliography, supplied by a caller


class SourceMeta(TypedDict, total=False):
    """Optional known metadata about the paper being refined, supplied by a caller that
    already has it (e.g. papis-ask, straight from info.yaml). The external-input sibling of
    ``SourcePaper``. Every key is optional; an absent bundle or key just falls back to what
    refinery derives from OCR, so passing it never creates a dependency -- it only lets
    refinery skip re-deriving what's already known.

    ``references`` is the paper's own bibliography if the caller already has it (papis stores
    it in info.yaml ``citations:``, from CrossRef). Each entry may use refinery's candidate
    shape (``title``/``doi``/``year``/``authors``) or common CrossRef-reference keys
    (``article-title``/``volume-title``/``DOI``/``author``) -- both are accepted. It's matched
    against the printed references locally (no network) before any provider search.
    """

    doi: str | None
    arxiv: str | None
    title: str | None
    year: int | None
    authors: list[dict]  # [{"family": ..., "given": ...}, ...]
    references: list[dict]  # the paper's own bibliography (see above)


def source_from_meta(
    meta: SourceMeta | dict | None, fallback_title: str | None = None
) -> SourcePaper:
    """Build a ``SourcePaper`` from a caller-supplied metadata bundle, falling back to
    ``fallback_title`` (typically refinery's OCR'd H1) when the bundle carries no title. A
    caller's authoritative title/year/authors supersede the OCR guess, making source
    identification for the bulk-references fast-path far more reliable."""
    m = meta or {}
    return SourcePaper(
        doi=m.get("doi"),
        arxiv=m.get("arxiv"),
        title=m.get("title") or fallback_title,
        year=m.get("year"),
        authors=m.get("authors"),
        references=m.get("references"),
    )


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------

_TITLE_JUNK_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_title(title: str | None) -> str:
    return " ".join(_TITLE_JUNK_RE.sub(" ", (title or "").lower()).split())


def title_similarity(a: str | None, b: str | None) -> float:
    return SequenceMatcher(None, _normalize_title(a), _normalize_title(b)).ratio()


def _first_family(entry: dict) -> str | None:
    authors = entry.get("authors") or []
    family = authors[0].get("family") if authors else None
    return fold_name(family) if family else None


def _acceptable(extracted: dict, candidate: dict, cfg: CitationConfig) -> bool:
    """Two-tier acceptance bar for a title-search hit.

    Tier 1: title similarity >= threshold AND year within tolerance (a missing year on
    either side skips the year check -- can't disprove). Title similarity alone was
    explicitly rejected as a false-positive risk, hence the AND.

    Tier 2 (user: the strict bar alone rejects OCR-garbled titles' correct matches):
    weaker title evidence, similarity in [relaxed, threshold), is accepted only with
    STRONGER corroboration -- exact year match AND first-author surname match
    (diacritic-folded). Weaker on one axis, stricter on the others.
    """
    similarity = title_similarity(extracted.get("title"), candidate.get("title"))
    year, cand_year = extracted.get("year"), candidate.get("year")
    if similarity >= cfg.title_similarity_threshold:
        return not (
            year is not None
            and cand_year is not None
            and abs(year - cand_year) > cfg.year_tolerance
        )
    if similarity >= cfg.title_similarity_relaxed:
        family, cand_family = _first_family(extracted), _first_family(candidate)
        return (
            year is not None and year == cand_year and family is not None and family == cand_family
        )
    return False


# ---------------------------------------------------------------------------
# CSL type inference
# ---------------------------------------------------------------------------

_TYPE_MAPS = {
    "semanticscholar": {
        "JournalArticle": "article-journal",
        "Review": "article-journal",
        "Conference": "paper-conference",
        "Book": "book",
        "BookSection": "chapter",
        "Dataset": "dataset",
    },
    "crossref": {
        "journal-article": "article-journal",
        "proceedings-article": "paper-conference",
        "posted-content": "article",  # preprints
        "book": "book",
        "book-chapter": "chapter",
        "monograph": "book",
        "report": "report",
        "dissertation": "thesis",
    },
    "openalex": {
        "article": "article-journal",
        "preprint": "article",
        "book": "book",
        "book-chapter": "chapter",
        "report": "report",
        "dissertation": "thesis",
    },
}


def infer_type(entry: dict, raw_text: str) -> str:
    """Closest CSL type: the accepting provider's own type when available, then raw-text
    heuristics, then a safe generic default (never left unset)."""
    source, provider_type = entry.get("source"), entry.get("provider_type")
    if source and provider_type:
        mapped = _TYPE_MAPS.get(source, {}).get(provider_type)
        if mapped:
            return mapped
    if entry.get("container_title"):
        return "article-journal"
    if "arxiv" in (raw_text or "").lower():
        return "article"
    return "article"


# ---------------------------------------------------------------------------
# Per-entry resolution + the public orchestrator
# ---------------------------------------------------------------------------


def _is_preprint(candidate: dict) -> bool:
    """Preprint by the provider's own type, or by an arXiv DOI (10.48550/...)."""
    if candidate.get("provider_type") in ("preprint", "posted-content"):
        return True
    return (candidate.get("doi") or "").lower().startswith("10.48550/")


def _merge_candidate(out: dict, candidate: dict, match: str) -> dict:
    """Merge an accepted candidate's fields into the extracted guess and stamp verified
    metadata. Provider data wins wherever it exists (authors included); the extractor's
    guess is kept for fields the provider lacks (commonly volume/page). Shared by the
    per-entry resolver and the S2 bulk fast-path so both merge identically.
    """
    for fld in ("title", "year", "doi", "abstract", "authors"):
        if not candidate.get(fld):
            continue
        if fld == "year" and out.get("year") and candidate["year"] < out["year"]:
            # never pull the year backward (user decision): a provider year *below* the
            # printed one is the preprint's (S2 merges preprint+published and reports the
            # earliest); keep the paper's printed (published) year. Years still gate
            # acceptance the same way either side (`_acceptable`).
            continue
        out[fld] = candidate[fld]
    out["source"] = candidate.get("source")
    out["provider_type"] = candidate.get("provider_type")
    return {**out, "verified": True, "match": match}


def _resolve_by_doi(raw_text: str, cfg: CitationConfig) -> tuple[dict, str] | None:
    """DOI-first exact lookup: a DOI printed in the raw OCR text is definitionally correct,
    so its S2 record is accepted with no similarity check. None when there's no usable hit."""
    doi = extract_doi(raw_text)
    if not doi:
        return None
    candidate = normalize_s2(s2_by_doi(doi, cfg))
    if candidate and candidate.get("title"):
        return candidate, "doi"
    return None


def _better_near_miss(
    near_miss: dict | None, provider: str, source_title: str | None, hit: dict
) -> dict:
    """Keep the highest-similarity REJECTED candidate seen so far (a diff-report aid for
    threshold tuning) -- the incoming hit replaces the incumbent only if it scores higher."""
    similarity = title_similarity(source_title, hit.get("title"))
    if near_miss is not None and similarity <= near_miss["similarity"]:
        return near_miss
    return {
        "provider": provider,
        "similarity": round(similarity, 3),
        "title": hit.get("title"),
        "year": hit.get("year"),
    }


def _resolve_by_title(
    out: dict, cfg: CitationConfig
) -> tuple[dict | None, str | None, dict | None]:
    """Title-search fallback across CrossRef -> S2 -> OpenAlex under the two-tier bar.

    CrossRef leads (user decision, 2026-07-05): S2's records blend preprint and published
    versions -- published DOI but the *earliest* (arXiv) year, confirmed live on 3 hyco refs
    pulled back a year -- while CrossRef's ``issued`` is the published record's own date.
    Published-over-preprint: an acceptable *preprint* hit is remembered but doesn't stop the
    chain, so a later provider's published version can still win. Returns
    ``(candidate, match, near_miss)``; candidate/match are None when nothing clears the bar.
    """
    providers = (
        ("crossref", crossref_search, normalize_crossref),
        ("semanticscholar", s2_search, normalize_s2),
        ("openalex", openalex_search, normalize_openalex),
    )
    preprint_fallback: tuple[str, dict] | None = None
    near_miss: dict | None = None
    for name, search, normalize in providers:
        hit = normalize(search(out["title"], cfg))
        if not hit:
            continue
        if not _acceptable(out, hit, cfg):
            near_miss = _better_near_miss(near_miss, name, out.get("title"), hit)
            continue
        if _is_preprint(hit):
            preprint_fallback = preprint_fallback or (name, hit)
            continue
        return hit, name, near_miss
    if preprint_fallback is not None:
        name, hit = preprint_fallback
        return hit, name, near_miss
    return None, None, near_miss


def _fill_crossref_abstract(candidate: dict, match: str, cfg: CitationConfig) -> None:
    """CrossRef rarely carries an abstract -- one S2-by-DOI follow-up just for that field."""
    if match == "crossref" and not candidate.get("abstract") and candidate.get("doi"):
        followup = normalize_s2(s2_by_doi(candidate["doi"], cfg))
        if followup and followup.get("abstract"):
            candidate["abstract"] = followup["abstract"]


def verify_and_resolve(extracted: dict, raw_text: str, cfg: CitationConfig) -> dict:
    """Resolve one reference: DOI-first exact lookup, else CrossRef -> S2 -> OpenAlex title
    search under the two-tier acceptance bar (see ``_resolve_by_doi`` / ``_resolve_by_title``).

    Returns the extracted dict merged with the accepted candidate's fields (provider data wins
    wherever it exists, authors included; extractor's guess kept for fields the provider lacks,
    commonly volume/page), plus ``verified``/``match``/``source``/``provider_type``. Unverified
    -> guess returned untouched with ``verified: False`` and, when any provider produced a
    rejected candidate, a ``near_miss`` record (best-scoring reject) for threshold tuning.
    """
    out = dict(extracted)
    doi_hit = _resolve_by_doi(raw_text, cfg)
    if doi_hit is not None:
        candidate, match, near_miss = doi_hit[0], doi_hit[1], None
    elif out.get("title"):
        candidate, match, near_miss = _resolve_by_title(out, cfg)
    else:
        candidate, match, near_miss = None, None, None

    if candidate is None:
        result = {**out, "verified": False, "match": None}
        if near_miss is not None:
            result["near_miss"] = near_miss
        return result
    assert match is not None  # match is set together with candidate on every accept path

    _fill_crossref_abstract(candidate, match, cfg)
    return _merge_candidate(out, candidate, match)


def _source_confident(source: SourcePaper, candidate: dict, cfg: CitationConfig) -> bool:
    """Is a title-search hit really the source paper? Two-tier, mirroring ``_acceptable``:

    - A **strong** title match (>= threshold) identifies the source on its own. Year/author
      are NOT vetoes here: the source is a specific, known paper, so a near-exact title match
      is almost certainly it, and provided year/authors often drift from a provider's record
      (a preprint year, a name-format difference) -- vetoing on that would false-reject the
      right paper and, crucially, do WORSE than a title-only lookup with no metadata at all.
      Supplying more metadata must never lose the fast-path a bare title would have won.
    - A **borderline** title match (in [relaxed, threshold)) is rescued only when year AND
      first-author surname corroborate it -- that's where caller-supplied metadata earns its
      keep (a garbled OCR title identified via papis's authoritative year+authors).
    """
    similarity = title_similarity(source.title, candidate.get("title"))
    if similarity >= cfg.title_similarity_threshold:
        return True
    if similarity >= cfg.title_similarity_relaxed:
        year, cand_year = source.year, candidate.get("year")
        family = _first_family({"authors": source.authors or []})
        cand_family = _first_family(candidate)
        return bool(
            year is not None and year == cand_year and family and family == cand_family
        )
    return False


def _dedup_candidates(candidates: list[dict]) -> list[dict]:
    """Drop duplicates across bulk sources: same DOI (folded), else same normalized title.
    First occurrence wins (S2 before OpenAlex)."""
    seen_doi: set[str] = set()
    seen_title: set[str] = set()
    out: list[dict] = []
    for c in candidates:
        doi = (c.get("doi") or "").lower()
        title = _normalize_title(c.get("title"))
        if (doi and doi in seen_doi) or (not doi and title and title in seen_title):
            continue
        if doi:
            seen_doi.add(doi)
        if title:
            seen_title.add(title)
        out.append(c)
    return out


def _normalize_caller_references(entries: list[dict]) -> list[dict]:
    """Turn caller-supplied reference entries (``SourceMeta.references``) into bulk candidates.

    Accepts refinery's candidate shape (title/doi/year/authors) OR common CrossRef-reference
    keys (article-title/volume-title/DOI/author-as-string), so a papis wrapper can pass
    info.yaml ``citations:`` almost verbatim. Entries with neither a title nor a DOI are
    dropped (nothing to match on). Each candidate is tagged ``_pool="papis"`` so a hit is
    reported as ``match="papis"`` (vs ``"bulk"`` for the S2 pool).
    """
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        title = (
            e.get("title")
            or e.get("article-title")
            or e.get("volume-title")
            or e.get("unstructured")
        )
        doi = e.get("doi") or e.get("DOI")
        if not (title or doi):
            continue
        raw_year = e.get("year")
        try:
            year = int(str(raw_year)[:4]) if raw_year else None
        except ValueError:
            year = None
        authors = e.get("authors")
        if not authors and e.get("author"):
            a = e["author"]
            authors = [{"family": a}] if isinstance(a, str) else a
        out.append(
            {
                "title": title,
                "doi": doi,
                "year": year,
                "authors": authors,
                "source": "papis",
                "provider_type": None,
                "_pool": "papis",
            }
        )
    return out


def _source_references(
    source: SourcePaper | None, cfg: CitationConfig, n_refs: int
) -> list[dict] | None:
    """The source paper's references as a bulk candidate pool, or None when the fast-path is
    off / the source can't be identified confidently.

    S2 first (an external id is trusted; a title match is gated by ``_source_confident`` so we
    never pull the wrong paper's refs). S2 returns >= the printed count when it truly has the
    references, so a short list means the publisher elided them (ASME/IEEE...) or coverage is
    partial -- fill from OpenAlex (a different licensing regime) when a DOI is known, and dedup
    the combined pool.
    """
    if not cfg.s2_bulk_references or source is None:
        return None
    hit = None
    if source.doi or source.arxiv:
        hit = s2_paper_id(cfg, doi=source.doi, arxiv=source.arxiv)  # exact -> trusted
    elif source.title:
        found = s2_paper_id(cfg, title=source.title)
        if found and _source_confident(source, found[1], cfg):
            hit = found
    s2 = (s2_references(hit[0], cfg) or []) if hit else []
    if len(s2) < n_refs and source.doi:
        openalex = openalex_references(source.doi, cfg) or []
        return _dedup_candidates([*s2, *openalex]) or None
    return s2 or None


def _match_in_bulk(
    extracted: dict, raw_text: str, bulk: list[dict], cfg: CitationConfig
) -> dict | None:
    """Match one extracted reference against the source's bulk reference list -- locally, no
    network: a shared DOI first, else the best title hit clearing ``_acceptable``. Returns the
    merged resolved dict (``match="bulk"``) or None to fall back to a per-entry search.
    """
    doi = extract_doi(raw_text) or extracted.get("doi")
    if doi:
        doi = doi.lower()
        for cand in bulk:
            if (cand.get("doi") or "").lower() == doi:
                return _merge_candidate(dict(extracted), cand, cand.get("_pool", "bulk"))
    best, best_sim = None, 0.0
    for cand in bulk:
        if _acceptable(extracted, cand, cfg):
            s = title_similarity(extracted.get("title"), cand.get("title"))
            if s > best_sim:
                best, best_sim = cand, s
    return _merge_candidate(dict(extracted), best, best.get("_pool", "bulk")) if best else None


def resolve_references(
    extracted: list[dict],
    raw_references: list[RawReference],
    cfg: CitationConfig,
    source: SourcePaper | None = None,
) -> list[dict]:
    """Resolve a whole bibliography: layer-1 output positionally merged with parse.py's raw
    ``[{page, number, text}]`` list, each entry verified concurrently.

    The file's one public orchestrator. Each printed ref is resolved locally (no per-ref
    network) against, in order: (1) the caller's OWN references (papis ``citations:``, tagged
    match="papis"), then (2) the source paper's references fetched ONCE from S2/OpenAlex
    (match="bulk"); whatever neither covers falls back to a per-entry provider search. The S2
    fetch is SKIPPED entirely when the caller's references already cover every printed ref --
    so a paper with complete papis citations resolves with zero network. Deliberately does NOT
    call ``extract_references`` itself -- cli.py owns sequencing. Output entries: ``{page,
    number, raw_text, ...resolved fields..., verified, match, type}``, in input order; a
    missing parse-level ``number`` is recovered from the raw text's leading marker.
    """
    if not raw_references:
        return []
    padded = (list(extracted) + [{}] * len(raw_references))[: len(raw_references)]

    # Pass 1: resolve each printed ref against the caller's own references (papis citations) --
    # local, free, no source lookup, survives S2 throttling.
    caller = _normalize_caller_references(source.references) if source and source.references else []
    caller_hits: list[dict | None] = [
        _match_in_bulk(padded[i], raw_references[i]["text"], caller, cfg) if caller else None
        for i in range(len(raw_references))
    ]
    # Only fetch the S2/OpenAlex source references if the caller pool left gaps -- when papis
    # already covered the whole bibliography we skip the source lookup + fetch entirely.
    provider_bulk = (
        [] if all(caller_hits) else (_source_references(source, cfg, len(raw_references)) or [])
    )

    def resolve_one(i: int) -> dict:
        raw = raw_references[i]
        resolved = caller_hits[i]
        if resolved is None and provider_bulk:
            resolved = _match_in_bulk(padded[i], raw["text"], provider_bulk, cfg)
        if resolved is None:
            resolved = verify_and_resolve(padded[i], raw["text"], cfg)
        entry = {
            "page": raw.get("page"),
            "number": raw.get("number") or leading_number(raw["text"]),
            "raw_text": raw["text"],
            **resolved,
        }
        entry["type"] = infer_type(entry, raw["text"])
        return entry

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        return list(pool.map(resolve_one, range(len(raw_references))))


# ---------------------------------------------------------------------------
# Diff report: "what we have (layer-1 JSON) vs what it should be (verified data)"
# ---------------------------------------------------------------------------


def _fmt_authors(entry: dict, limit: int = 3) -> str:
    families = [a.get("family") or "?" for a in entry.get("authors") or []]
    if not families:
        return "-"
    shown = ", ".join(families[:limit])
    return f"{shown}, +{len(families) - limit}" if len(families) > limit else shown


def format_resolution_report(extracted: list[dict], resolved: list[dict]) -> str:
    """Human-readable per-reference diff of layer-1 guesses vs resolved records.

    For each entry: match route, how far the final title moved from the extracted one,
    year changes, which fields were enriched (doi/abstract/authors), author-list
    replacement. Unverified entries are grouped at the end with their best-scoring
    rejected candidate (``near_miss``) -- the data for judging whether the similarity
    thresholds are too strict.
    """
    lines: list[str] = []
    verified = [r for r in resolved if r.get("verified")]
    by_match: dict[str, int] = {}
    for r in verified:
        by_match[r.get("match") or "?"] = by_match.get(r.get("match") or "?", 0) + 1
    route = ", ".join(f"{k}: {v}" for k, v in sorted(by_match.items()))
    lines.extend((f"resolved {len(verified)}/{len(resolved)} references ({route})", ""))

    for i, r in enumerate(resolved):
        ext = extracted[i] if i < len(extracted) else {}
        num = r.get("number") or "?"
        if not r.get("verified"):
            continue
        similarity = title_similarity(ext.get("title"), r.get("title"))
        year_ext, year_res = ext.get("year"), r.get("year")
        year_note = f"{year_ext}->{year_res}" if year_ext != year_res else f"{year_res}"
        gained = [f for f in ("doi", "abstract") if r.get(f) and not ext.get(f)]
        authors_note = ""
        if r.get("authors") and r.get("authors") != ext.get("authors"):
            authors_note = f" | authors: {_fmt_authors(ext)} -> {_fmt_authors(r)}"
        lines.append(
            f"[{num:>3}] {r.get('match'):<15} title-sim {similarity:.2f} | "
            f"year {year_note}" + (f" | +{'+'.join(gained)}" if gained else "") + authors_note
        )
        lines.append(f"      {(r.get('title') or '')[:90]}")

    unverified = [(i, r) for i, r in enumerate(resolved) if not r.get("verified")]
    if unverified:
        lines.extend(("", f"UNVERIFIED ({len(unverified)}) -- layer-1 guess kept untouched:"))
        for i, r in unverified:
            ext = extracted[i] if i < len(extracted) else {}
            shown = (ext.get("title") or r.get("raw_text") or "")[:80]
            lines.append(f"[{r.get('number') or '?':>3}] {shown}")
            miss = r.get("near_miss")
            if miss:
                lines.append(
                    f"      best reject: {miss['provider']} sim {miss['similarity']} "
                    f"year {miss.get('year')} | {(miss.get('title') or '')[:70]}"
                )
            else:
                lines.append("      no candidate from any provider")
    return "\n".join(lines)
