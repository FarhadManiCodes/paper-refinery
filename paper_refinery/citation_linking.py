"""Deterministic in-text citation linking + (dry-run) standardization.

Finds where a paper's body cites its references and links each marker to the
bibliography entr(ies) it points at. No LLM, no network -- pure text rules anchored on
two facts earlier stages already established:

1. **The paper's marker style is declared by its own bibliography**, via layer-1
   extraction's ``citation_key`` values ("[1]" -> bracket style; "1."/"(1)" -> plain
   number style; all null -> author-year). Never inferred from the body.
2. **The valid target set is closed**: N references with known numbers, surnames, and
   years. A candidate marker that doesn't resolve entirely into that set is left
   unlinked -- the same leave-don't-guess philosophy as parse.py's reference sort.

Live-confirmed hazards these rules must survive (seen in this project's own samples):
a bracketed math interval ``[0,1]`` alongside real ``[N]`` citations (hyco); PNAS-style
paren citations ``(3, 4)`` colliding with equation numbers (brunton); pure author-year
with no numbers anywhere (fmech).

Body rewriting to papis ``author_year`` citekeys (``rewrite_markers``) is wired into
``cli._refine`` after resolution: citekeys are built from layer-2/3-verified
surnames/years, never layer-1's raw guess -- an OCR-corrupted surname would otherwise
propagate into every citekey in the body. A marker is rewritten only when every
reference it points at has a citekey; anything else is left exactly as printed.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Style inference (from the bibliography's own citation keys, never the body)
# ---------------------------------------------------------------------------

_NUMBERED_KEY_RE = re.compile(r"^\[?\(?\d+[]).]?$")


def infer_marker_style(extracted: list[dict]) -> str:
    """ "numbered" | "author-year", by majority over layer-1 citation_keys.

    Mostly-null keys mean the bibliography prints no markers at all -> author-year.
    Deliberately NOT distinguishing bracket from bare/paren numbering here: the
    bibliography's own marker form does not predict the body's (confirmed on
    kalman-1960 -- bibliography prints bare "1 ", body cites "[29]"). Which numbered
    form the body actually uses is decided by scanning for both; see link_citations.
    """
    keys = [(e.get("citation_key") or "").strip() for e in extracted]
    present = [k for k in keys if k]
    if len(present) * 2 < len(keys) or not present:
        return "author-year"
    numbered = sum(1 for k in present if _NUMBERED_KEY_RE.match(k))
    return "numbered" if numbered * 2 >= len(present) else "author-year"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@dataclass
class Marker:
    """One in-text citation occurrence, linked to bibliography entries."""

    start: int  # offset into the body markdown
    end: int
    text: str  # the marker exactly as printed, e.g. "[7,8]" or "(Smith et al., 2023)"
    ref_indices: list[int]  # 0-based indices into the extracted/references list


@dataclass
class LinkResult:
    style: str
    markers: list[Marker]
    uncited: list[int] = field(default_factory=list)  # ref indices never linked
    ambiguous: list[str] = field(default_factory=list)  # candidates rejected by a guard


_MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$", re.DOTALL)


def _math_spans(markdown: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _MATH_SPAN_RE.finditer(markdown)]


def _inside_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= start and end <= e for s, e in spans)


_KEY_DIGITS_RE = re.compile(r"\d+")


def _number_index(extracted: list[dict]) -> dict[int, int]:
    """Printed reference number -> 0-based list index.

    From each entry's citation_key digits when present (robust to a gap, e.g. a
    bibliography missing entry 2 where position no longer equals number); positional
    fallback (entry i <-> number i+1) otherwise.
    """
    mapping: dict[int, int] = {}
    for i, entry in enumerate(extracted):
        key = entry.get("citation_key") or ""
        m = _KEY_DIGITS_RE.search(key)
        mapping[int(m.group(0)) if m else i + 1] = i
    return mapping


_RANGE_RE = re.compile(r"^(\d{1,3})\s*[-–]\s*(\d{1,3})$")
_MAX_RANGE_SPAN = 50  # a "[12-14]" style range wider than this is not a citation


def _expand_number_group(group: str) -> list[int] | None:
    """ "7, 8" / "12-14" / "3" -> explicit numbers; None if any part isn't citation-like."""
    numbers: list[int] = []
    for part in re.split(r"[,;]", group):
        part = part.strip()
        m = _RANGE_RE.match(part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if not (lo < hi and hi - lo <= _MAX_RANGE_SPAN):
                return None
            numbers.extend(range(lo, hi + 1))
        elif part.isdigit():
            numbers.append(int(part))
        else:
            return None
    return numbers or None


# ---------------------------------------------------------------------------
# Numbered styles
# ---------------------------------------------------------------------------

_BRACKET_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,;]\s*\d{1,3}|\s*[-–]\s*\d{1,3})*)\]")
_PAREN_MARKER_RE = re.compile(r"\((\d{1,3}(?:\s*[,;]\s*\d{1,3}|\s*[-–]\s*\d{1,3})*)\)")
_EQ_BEFORE_RE = re.compile(r"(?:\bEqs?\.?|\bequations?)\s*$", re.IGNORECASE)
_EQ_TAG_RE = re.compile(r"\\tag\{\[?(\d{1,3})\]?\}")
#   display equations OCR'd with an explicit number carry a LaTeX \tag ("\tag{[4]}" in
#   brunton/PNAS) -- their maximum bounds the paper's equation numbering


def _scan_numbered(
    markdown: str, extracted: list[dict], form: str, ambiguous: list[str]
) -> list[Marker]:
    """One bracket- or paren-form scan, under the guards that make linking deterministic.

    Every number in a group must resolve into the bibliography's own number set --
    rejecting hyco's ``[0,1]`` unit-square interval (0 is never a reference) and
    brunton's out-of-range equation numbers in one rule. Markers inside math spans are
    skipped. The paren form additionally rejects "Eq. (12)"-style contexts and markers
    that are an entire paragraph on their own (a stray equation-number region). The
    Eq-context guard is relaxed (user's idea) when the paper's own equation tags prove
    the numbers can't be equations: brunton tags its equations [1]-[6], so
    "Navier-Stokes equations (44, 45)" must be a citation. Only when tags were
    actually detected -- with none, "equation (3)" stays uncheckable and rejected.
    """
    valid = _number_index(extracted)
    spans = _math_spans(markdown)
    # the equation-tag bound belongs to the paren guard alone; don't scan for brackets
    max_eq_tag = (
        max((int(t) for t in _EQ_TAG_RE.findall(markdown)), default=0) if form == "paren" else 0
    )
    pattern = _BRACKET_MARKER_RE if form == "bracket" else _PAREN_MARKER_RE
    markers: list[Marker] = []
    for m in pattern.finditer(markdown):
        if _inside_any(m.start(), m.end(), spans):
            continue
        numbers = _expand_number_group(m.group(1))
        if numbers is None:
            continue
        if any(n not in valid for n in numbers):
            ambiguous.append(m.group(0))
            continue
        if form == "paren":
            before = markdown[max(0, m.start() - 12) : m.start()]
            if _EQ_BEFORE_RE.search(before) and not (
                max_eq_tag and all(n > max_eq_tag for n in numbers)
            ):
                ambiguous.append(m.group(0))
                continue
            # a marker that IS its whole paragraph is an equation tag, not a citation
            para_start = markdown.rfind("\n\n", 0, m.start()) + 2
            para_end = markdown.find("\n\n", m.end())
            para_end = para_end if para_end != -1 else len(markdown)
            if not (
                markdown[para_start : m.start()].strip() or markdown[m.end() : para_end].strip()
            ):
                ambiguous.append(m.group(0))
                continue
        markers.append(Marker(m.start(), m.end(), m.group(0), [valid[n] for n in numbers]))
    return markers


_BRACKET_DECISIVE = 3  # accepted bracket markers >= this decide the form outright


def _find_numbered_markers(
    markdown: str, extracted: list[dict], result: LinkResult
) -> list[Marker]:
    """Scan both numbered forms; bracket evidence is decisive, paren is the fallback.

    A paper never mixes bracket and paren citations, but the bibliography's own marker
    form doesn't predict which one the body uses (kalman: bare-numbered bibliography,
    bracket-citing body). Selection is NOT a plain count comparison -- that failed
    live on kalman: 46 genuine ``[N]`` citations were outvoted by 63 paren false
    positives (in-range numbered list items "(1) ..." and equation references). An
    in-range, out-of-math ``[N]`` has essentially no false-positive source, so a
    handful of accepted bracket markers (``_BRACKET_DECISIVE``) settles it; paren form
    only applies when bracket evidence is essentially absent (brunton/PNAS: zero
    brackets, paren-citing body).
    """
    bracket_amb: list[str] = []
    paren_amb: list[str] = []
    bracket = _scan_numbered(markdown, extracted, "bracket", bracket_amb)
    paren = _scan_numbered(markdown, extracted, "paren", paren_amb)
    if len(bracket) >= _BRACKET_DECISIVE or len(bracket) >= len(paren):
        result.style, result.ambiguous = "numbered-bracket", bracket_amb
        return bracket
    result.style, result.ambiguous = "numbered-paren", paren_amb
    return paren


# ---------------------------------------------------------------------------
# Author-year style
# ---------------------------------------------------------------------------

_SURNAME = r"[A-Z][\w'’-]+"
_PAREN_AY_RE = re.compile(r"\(([^()\n]{0,160}?\d{4}[a-z]?[^()\n]{0,20}?)\)")
_AY_PART_RE = re.compile(
    rf"^(?:e\.g\.,?\s*|see\s+)?({_SURNAME})"
    rf"(?:\s+(?:and|&)\s+{_SURNAME}|\s+et al\.?)?,?\s+(\d{{4}})[a-z]?$"
)
_NARRATIVE_AY_RE = re.compile(
    rf"({_SURNAME})((?:\s+(?:and|&)\s+{_SURNAME})|\s+et al\.?)?\s+\((\d{{4}})[a-z]?\)"
)


def _fold(name: str) -> str:
    """Lowercased, diacritic-folded form for surname comparison.

    OCR is inconsistent about diacritics across a page: confirmed live on fmech, where
    the body prints "Höhn et al. (2011)" but the bibliography region OCR'd the same
    surname as "Hohn" -- an exact match misses a genuine citation.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _author_year_lookup(extracted: list[dict]) -> dict[tuple[str, int], int]:
    """(first-author surname folded, year) -> ref index; colliding pairs are
    dropped entirely (linking one of them would be a guess)."""
    lookup: dict[tuple[str, int], int] = {}
    collisions: set[tuple[str, int]] = set()
    for i, entry in enumerate(extracted):
        authors = entry.get("authors") or []
        family = _fold(authors[0].get("family") or "") if authors else ""
        year = entry.get("year")
        if not family or year is None:
            continue
        key = (family, int(year))
        if key in lookup:
            collisions.add(key)
        lookup[key] = i
    for key in collisions:
        del lookup[key]
    return lookup


def _find_author_year_markers(
    markdown: str, extracted: list[dict], result: LinkResult
) -> list[Marker]:
    lookup = _author_year_lookup(extracted)
    spans = _math_spans(markdown)
    markers: list[Marker] = []
    taken: list[tuple[int, int]] = []

    # parenthetical form first: "(Smith, 2023)", "(A et al., 2020; B, 2021)"
    for m in _PAREN_AY_RE.finditer(markdown):
        if _inside_any(m.start(), m.end(), spans):
            continue
        indices: list[int] = []
        for part in m.group(1).split(";"):
            pm = _AY_PART_RE.match(part.strip())
            if not pm:
                indices = []
                break
            key = (_fold(pm.group(1)), int(pm.group(2)))
            if key not in lookup:
                indices = []
                break
            indices.append(lookup[key])
        if indices:
            markers.append(Marker(m.start(), m.end(), m.group(0), indices))
            taken.append((m.start(), m.end()))
        else:
            result.ambiguous.append(m.group(0))

    # narrative form: "Smith et al. (2023)" -- skip anything overlapping a paren match
    for m in _NARRATIVE_AY_RE.finditer(markdown):
        if _inside_any(m.start(), m.end(), spans):
            continue
        if any(s < m.end() and m.start() < e for s, e in taken):
            continue
        key = (_fold(m.group(1)), int(m.group(3)))
        if key in lookup:
            markers.append(Marker(m.start(), m.end(), m.group(0), [lookup[key]]))
        else:
            result.ambiguous.append(m.group(0))
    markers.sort(key=lambda mk: mk.start)
    return markers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def link_citations(markdown: str, extracted: list[dict]) -> LinkResult:
    """Detect and link every in-text citation marker; the file's main entry point.

    ``extracted`` is layer-1 output (citation_extraction.extract_references), aligned
    with the paper's references. Returns all linked markers plus a coverage view:
    ``uncited`` (references never linked from the body) and ``ambiguous`` (candidates
    a guard rejected, for inspection). Both are diagnostic hints, not error signals --
    an uncited entry can mean a detection miss, an OCR-corrupted marker, a parse-layer
    artifact (fmech's split-URL fragment), or a reference genuinely never cited in the
    text, which legitimately happens. Only reading the entry tells which. The body is
    never modified here.
    """
    result = LinkResult(style=infer_marker_style(extracted), markers=[])
    if not extracted:
        return result
    if result.style == "numbered":
        result.markers = _find_numbered_markers(markdown, extracted, result)
    else:
        result.markers = _find_author_year_markers(markdown, extracted, result)
    cited = {i for mk in result.markers for i in mk.ref_indices}
    result.uncited = [i for i in range(len(extracted)) if i not in cited]
    return result


_CITEKEY_JUNK_RE = re.compile(r"[^a-z0-9]+")


def make_citekey(entry: dict) -> str | None:
    """papis-style ``surname_year`` citekey, or None when either part is missing.

    Callers decide whether ``entry`` is trustworthy -- for real rewriting this should
    be layer-2/3-verified data, not the raw layer-1 guess.
    """
    authors = entry.get("authors") or []
    family = (authors[0].get("family") or "") if authors else ""
    family = _CITEKEY_JUNK_RE.sub("", _fold(family))  # fold, then strip: Höhn -> hohn, not hhn
    year = entry.get("year")
    if not family or year is None:
        return None
    return f"{family}_{year}"


def rewrite_markers(markdown: str, markers: list[Marker], citekeys: list[str | None]) -> str:
    """Replace linked markers with ``[key]`` / ``[key1; key2]`` citekey form.

    Pure function, applied back-to-front so offsets stay valid. A marker is rewritten
    only when EVERY reference it points at has a citekey; anything else stays exactly
    as printed.
    """
    out = markdown
    for mk in sorted(markers, key=lambda m: m.start, reverse=True):
        keys = [citekeys[i] if i < len(citekeys) else None for i in mk.ref_indices]
        if any(k is None for k in keys):
            continue
        out = out[: mk.start] + "[" + "; ".join(keys) + "]" + out[mk.end :]
    return out
