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
from difflib import SequenceMatcher

from .citation_providers import (
    _normalize_crossref,
    _normalize_openalex,
    _normalize_s2,
    crossref_search,
    extract_doi,
    openalex_search,
    s2_by_doi,
    s2_search,
)
from .config import CitationConfig
from .text_utils import fold_name, leading_number

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


def verify_and_resolve(extracted: dict, raw_text: str, cfg: CitationConfig) -> dict:
    """Resolve one reference: DOI-first exact lookup, else CrossRef -> S2 -> OpenAlex
    title search under the two-tier acceptance bar.

    Published-over-preprint (user decision): an acceptable hit that is itself a
    preprint doesn't stop the provider chain -- it's kept as a fallback while the
    remaining providers are tried for the published version. Returns the extracted
    dict merged with the accepted candidate's fields (provider data wins wherever it
    exists, authors included; extractor's guess kept for fields the provider lacks,
    commonly volume/page), plus ``verified``/``match``/``source``/``provider_type``.
    Unverified -> guess returned untouched with ``verified: False`` and, when any
    provider produced a rejected candidate, a ``near_miss`` record (best-scoring
    reject) for threshold tuning from the diff report.
    """
    out = dict(extracted)
    candidate: dict | None = None
    match: str | None = None

    # a DOI printed in the raw OCR text is definitionally correct -- exact lookup,
    # no similarity check
    doi = extract_doi(raw_text)
    if doi:
        candidate = _normalize_s2(s2_by_doi(doi, cfg))
        if candidate and candidate.get("title"):
            match = "doi"
        else:
            candidate = None

    near_miss: dict | None = None
    if candidate is None and out.get("title"):
        # CrossRef leads (user decision, 2026-07-05): S2's records blend preprint and
        # published versions -- published DOI but the *earliest* (arXiv) year, confirmed
        # live on 3 hyco refs pulled backward a year -- while CrossRef's `issued` is the
        # published record's own date. S2 keeps DOI-first lookups + abstract follow-ups.
        providers = (
            ("crossref", crossref_search, _normalize_crossref),
            ("semanticscholar", s2_search, _normalize_s2),
            ("openalex", openalex_search, _normalize_openalex),
        )
        preprint_fallback: tuple[str, dict] | None = None
        for name, search, normalize in providers:
            hit = normalize(search(out["title"], cfg))
            if not hit:
                continue
            if _acceptable(out, hit, cfg):
                if _is_preprint(hit):
                    # keep looking for the published version; remember this one
                    preprint_fallback = preprint_fallback or (name, hit)
                    continue
                candidate, match = hit, name
                break
            similarity = title_similarity(out.get("title"), hit.get("title"))
            if near_miss is None or similarity > near_miss["similarity"]:
                near_miss = {
                    "provider": name,
                    "similarity": round(similarity, 3),
                    "title": hit.get("title"),
                    "year": hit.get("year"),
                }
        if candidate is None and preprint_fallback is not None:
            match, candidate = preprint_fallback

    if candidate is None:
        result = {**out, "verified": False, "match": None}
        if near_miss is not None:
            result["near_miss"] = near_miss
        return result

    # CrossRef rarely carries an abstract -- one S2-by-DOI follow-up just for that field
    if match == "crossref" and not candidate.get("abstract") and candidate.get("doi"):
        followup = _normalize_s2(s2_by_doi(candidate["doi"], cfg))
        if followup and followup.get("abstract"):
            candidate["abstract"] = followup["abstract"]

    for fld in ("title", "year", "doi", "abstract", "authors"):
        if not candidate.get(fld):
            continue
        if fld == "year" and out.get("year") and candidate["year"] < out["year"]:
            # never pull the year backward (user decision): a provider year *below*
            # the printed one is the preprint's (S2 merges preprint+published and
            # reports the earliest year, confirmed live) -- the paper's own printed
            # year is the published one, keep it. Years still gate acceptance the
            # same way either side (`_acceptable`).
            continue
        out[fld] = candidate[fld]
    out["source"] = candidate.get("source")
    out["provider_type"] = candidate.get("provider_type")
    return {**out, "verified": True, "match": match}


def resolve_references(
    extracted: list[dict], raw_references: list[dict], cfg: CitationConfig
) -> list[dict]:
    """Resolve a whole bibliography: layer-1 output positionally merged with parse.py's
    raw ``[{page, number, text}]`` list, each entry verified concurrently.

    The file's one public orchestrator. Deliberately does NOT call
    ``extract_references`` itself -- cli.py owns sequencing, same as every other stage.
    Output entries: ``{page, number, raw_text, ...resolved fields..., verified, match,
    type}``, in input order. A missing parse-level ``number`` is recovered from the raw
    text's leading marker when possible.
    """
    padded = (list(extracted) + [{}] * len(raw_references))[: len(raw_references)]

    def resolve_one(i: int) -> dict:
        raw = raw_references[i]
        resolved = verify_and_resolve(padded[i], raw["text"], cfg)
        entry = {
            "page": raw.get("page"),
            "number": raw.get("number") or leading_number(raw["text"]),
            "raw_text": raw["text"],
            **resolved,
        }
        entry["type"] = infer_type(entry, raw["text"])
        return entry

    if not raw_references:
        return []
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
    lines.append(f"resolved {len(verified)}/{len(resolved)} references ({route})")
    lines.append("")

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
        lines.append("")
        lines.append(f"UNVERIFIED ({len(unverified)}) -- layer-1 guess kept untouched:")
        for i, r in unverified:
            ext = extracted[i] if i < len(extracted) else {}
            lines.append(
                f"[{r.get('number') or '?':>3}] {(ext.get('title') or r.get('raw_text') or '')[:80]}"
            )
            miss = r.get("near_miss")
            if miss:
                lines.append(
                    f"      best reject: {miss['provider']} sim {miss['similarity']} "
                    f"year {miss.get('year')} | {(miss.get('title') or '')[:70]}"
                )
            else:
                lines.append("      no candidate from any provider")
    return "\n".join(lines)
