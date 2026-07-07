"""Bibliography repair: turn raw OCR'd reference entries into a trustworthy list.

parse.py routes ``reference_content`` regions into a flat ``[{"page", "number",
"text"}]`` list; every function here exists because a live paper proved that raw list
can't be trusted as-is: PP-DocLayout-V3 mislabels leading entries as body text
(brunton), splits an entry in two at a page break (fmech), tacks copyright back-matter
onto the end (fmech), skips a printed entry entirely (brunton's ref 2), and emits
multi-column bibliographies out of reading order (kalman). The shared philosophy is
leave-don't-guess: each repair fires only on positive evidence and degrades to a loud
warning, never a silent guess.

Consumed only by parse.py (the same relationship markers.py has to the pipeline):
``reclaim_mislabeled_references`` runs inside the per-page region loop, then
``repair_references`` runs the list-level repairs in their one working order, and
``render_references_markdown`` renders the final list for the ``.references.md``
sidecar. Structuring the bibliography into anything beyond plain OCR'd text --
authors/title/venue/DOI, in-text citation linking -- is deliberately a separate, later
concern (citation_extraction/resolution/linking), not this module's job.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TypedDict

from .markers import page_marker
from .text_utils import leading_number

logger = logging.getLogger(__name__)


class RawReference(TypedDict):
    """One raw OCR'd bibliography entry, as parse.py routes ``reference_content`` regions
    out of the body. Consumed by the repairs here and by citation_resolution. ``number``
    is the printed marker's digits when the layout model paired a number region (see
    parse.py's ``_merge_reference_numbers``), else None."""

    page: int
    number: str | None
    text: str


# a bibliography-entry-looking start: "[12] " or "12. " (bracket/dot required -- a bare
# "2 " would match too much ordinary body text to be safe as a reclaim signal)
_MISLABELED_REFERENCE_RE = re.compile(r"^\s*\[?\d{1,3}[\].]\s")


def reclaim_mislabeled_references(
    triples: list[tuple[str, str, dict]],
) -> list[tuple[str, str, dict]]:
    """Reroute body regions that are clearly bibliography entries back into the references.

    Operates on parse.py's per-page dispatch triples ``(kind, text, region)``.
    PP-DocLayout-V3 sometimes mislabels a bibliography's first entr(ies) as plain text --
    confirmed live on brunton-2016.pdf, where "1. Jordan MI, ..." was the last *body*
    paragraph while reference_content detection only began at entry 3. Recovery rule,
    deliberately narrow: on a page that already has detected reference regions, a body
    region is reclaimed iff it starts with a bibliography-entry marker ("[1] " / "1. ",
    see ``_MISLABELED_REFERENCE_RE``) *and* is directly adjacent to a reference region --
    iterated to a fixed point, so a contiguous run of mislabeled entries chains onto the
    reference run one by one. Numbered body text anywhere else on the page is never
    touched.
    """
    if not any(kind == "reference" for kind, _, _ in triples):
        return triples
    out = list(triples)
    changed = True
    while changed:
        changed = False
        for i, (kind, text, region) in enumerate(out):
            if kind != "body" or not _MISLABELED_REFERENCE_RE.match(text):
                continue
            prev_is_ref = i > 0 and out[i - 1][0] == "reference"
            next_is_ref = i + 1 < len(out) and out[i + 1][0] == "reference"
            if prev_is_ref or next_is_ref:
                out[i] = ("reference", text, region)
                changed = True
    return out


def repair_references(
    references: list[RawReference], pdf_path: Path | None = None
) -> list[RawReference]:
    """All list-level repairs, in their one working order.

    Page-break split fragments are rejoined first (so a rejoined entry counts as one),
    trailing back-matter dropped, layout-skipped entries recovered from the PDF text
    layer (``pdf_path`` None skips that), then the contiguity-guarded number sort --
    which a successful recovery may just have unblocked. Unrecovered gaps warn.
    """
    references = _merge_split_references(references)
    references = _drop_trailing_boilerplate(references)
    references = _recover_missing_references(references, pdf_path)
    references = _sort_references_by_number(references)
    _warn_reference_gaps(references)
    return references


_SURNAME_PARTICLES = frozenset("van von de del della der den da di du la le les ter ten te".split())
#   lowercase surname prefixes that legitimately START a bibliography entry
#   ("van Wijk, J. ..."): a lowercase first word alone must not read as a continuation


def _looks_like_continuation(text: str) -> bool:
    """True if ``text`` reads as the tail of a split entry rather than an entry start."""
    stripped = text.lstrip()
    if not stripped or not stripped[0].islower():
        return False
    first_word = stripped.split()[0].rstrip(",.").lower()
    return first_word not in _SURNAME_PARTICLES


def _join_split_entry(head: str, tail: str) -> str:
    """Rejoin a split entry; a mid-URL split is glued back without a space."""
    head, tail = head.rstrip(), tail.lstrip()
    tokens = head.split()
    last_token = tokens[-1] if tokens else ""
    if "://" in last_token or last_token.lower().startswith("www."):
        return head + tail
    return f"{head} {tail}"


def _merge_split_references(references: list[RawReference]) -> list[RawReference]:
    """Fold a page-break continuation fragment back into the entry it belongs to.

    A bibliography entry crossing a page boundary can come back as two regions --
    confirmed live on fmech-07-655266: ref 28 ends page 9 mid-URL
    ("https://journals.sagepub.") and page 10 opens with its tail ("com/home/pij
    Proc. Inst. ..."), which carries the entry's DOI, so the split also costs
    resolution its best signal. Deliberately narrow, to never glue two genuine
    entries: only an entry that (a) is the first on a *later* page than its
    predecessor, (b) has no paired number region and no leading "[N]"/"N." marker of
    its own, and (c) starts continuation-like -- lowercase first word that is not a
    surname particle (see ``_SURNAME_PARTICLES``) -- is merged. Chained fragments
    fold into the same entry one by one.
    """
    out: list[RawReference] = []
    for ref in references:
        prev = out[-1] if out else None
        if (
            prev is not None
            and ref["page"] > prev["page"]
            and ref.get("number") is None
            and leading_number(ref["text"]) is None
            and _looks_like_continuation(ref["text"])
        ):
            prev["text"] = _join_split_entry(prev["text"], ref["text"])
            continue
        # copy so a later in-place `prev["text"]` edit never mutates the caller's entry
        out.append({"page": ref["page"], "number": ref["number"], "text": ref["text"]})
    return out


_REFERENCE_BOILERPLATE_SIGNALS = (
    "conflict of interest",
    "copyright ©",
    "creative commons",
    "open-access article distributed",
)


def _is_reference_boilerplate(text: str) -> bool:
    """True if ``text`` looks like misclassified back-matter rather than an actual
    bibliography entry."""
    lowered = text.lower()
    return any(signal in lowered for signal in _REFERENCE_BOILERPLATE_SIGNALS)


_MAX_TRAILING_BOILERPLATE_CHECK = 3


def _drop_trailing_boilerplate(references: list[RawReference]) -> list[RawReference]:
    """Drop a trailing run of misclassified back-matter entries from the bibliography.

    Checks only the last few entries (up to ``_MAX_TRAILING_BOILERPLATE_CHECK``), and
    only ever removes a *contiguous run starting from the very end* -- stops at the
    first entry that doesn't match, so a genuine reference is never dropped just for
    being near the end of the list (e.g. one whose own title happens to mention
    "copyright"). Sometimes the boilerplate itself splits across more than one entry
    (e.g. "Conflict of Interest: ..." and "Copyright © ..." as two separate regions),
    which is why this checks more than just the single last entry.
    """
    cleaned = list(references)
    checked = 0
    while (
        cleaned
        and checked < _MAX_TRAILING_BOILERPLATE_CHECK
        and _is_reference_boilerplate(cleaned[-1]["text"])
    ):
        cleaned.pop()
        checked += 1
    return cleaned


def _missing_reference_numbers(references: list[RawReference]) -> list[int]:
    """Missing printed numbers of a numbered bibliography; ``[]`` when there is nothing
    trustworthy to report -- keys unclean (author-year style, garbled markers),
    duplicated (two entries garbled to the same number make the expected-set math
    meaningless, same stance as ``_sort_references_by_number``), or simply complete.
    Single source of gap arithmetic for the warning and the text-layer recovery."""
    keys = [_reference_sort_key(ref) for ref in references]
    if not keys or any(key is None for key in keys) or len(set(keys)) != len(keys):
        return []
    nums = [k for k in keys if k is not None]  # all clean & unique past the guard
    return sorted(set(range(1, max(nums) + 1)) - set(nums))


def _warn_reference_gaps(references: list[RawReference]) -> None:
    """Warn (never fix or drop) when a numbered bibliography has holes.

    A gap means the layout model produced no region at all for an entry (confirmed live:
    brunton-2016's entry 2 simply has no region) -- nothing downstream can recover text
    that was never OCR'd, but a silent loss is worse than a loud one. Only fires when
    every entry has a clean numeric key; unnumbered (author-year) styles say nothing.
    Runs after ``_recover_missing_references``, so it only reports what recovery from
    the PDF text layer couldn't fill either.
    """
    missing = _missing_reference_numbers(references)
    if missing:
        logger.warning(
            "numbered bibliography has %d missing entr(ies): %s -- the layout model "
            "likely produced no region for them (unrecoverable here)",
            len(missing),
            missing,
        )


_RECOVERY_PREFIX_CHARS = 40  # of a neighbor entry's text used to anchor into the text layer
_MIN_RECOVERED_CHARS = 20  # a shorter "entry" is a stray number hit, not a reference


def _normalize_layer_text(text: str) -> str:
    """Whitespace-collapsed form shared by the PDF text layer and OCR text so the two
    can be substring-matched; line-break hyphenation is rejoined first."""
    text = re.sub(r"-\n(?=[a-z])", "", text)
    return re.sub(r"\s+", " ", text)


def _splice_missing_from_layer(
    references: list[RawReference], layer_text: str, missing: list[int]
) -> list[RawReference]:
    """Fill numbered-bibliography gaps with entries read from the PDF's own text layer.

    Pure logic half of ``_recover_missing_references`` (which owns the file I/O and
    computes ``missing`` -- guaranteed non-empty with clean, unique keys, see
    ``_missing_reference_numbers``). Every step is anchored on data already trusted,
    and any step failing skips that entry (the gap warning then still fires): the
    missing number's nearest present neighbors are located in the text layer by their
    OCR'd text prefix; the missing entry must start with its own printed marker
    ("[N] "/"N. ") exactly once in the span between them -- zero hits or several (a
    stray "Vol. 2." lookalike) means no guessing. A recovered entry is spliced in
    right after its predecessor; its text starts at the printed marker by
    construction, which doubles as its sort key (``number`` stays None like its
    OCR'd siblings -- setting it would render a doubled "[2] 2. ..." marker), so the
    downstream number sort sees a contiguous run again.
    """
    layer = _normalize_layer_text(layer_text)
    by_key = {_reference_sort_key(ref): ref for ref in references}
    out = list(references)
    for n in missing:
        prev_key = max((k for k in by_key if k is not None and k < n), default=None)
        next_key = min((k for k in by_key if k is not None and k > n), default=None)
        if prev_key is None or next_key is None:
            continue
        prev_prefix = _normalize_layer_text(by_key[prev_key]["text"])[:_RECOVERY_PREFIX_CHARS]
        next_prefix = _normalize_layer_text(by_key[next_key]["text"])[:_RECOVERY_PREFIX_CHARS]
        i_prev, i_next = layer.find(prev_prefix), layer.find(next_prefix)
        if i_prev == -1 or i_next == -1 or i_next <= i_prev:
            continue
        segment = layer[i_prev:i_next]
        starts = [m.start() for m in re.finditer(rf"(?:^|(?<=\s))\[?{n}[\].]\s", segment)]
        if len(starts) != 1:
            continue
        text = segment[starts[0] :].strip()
        if len(text) < _MIN_RECOVERED_CHARS:
            continue
        recovered: RawReference = {"page": by_key[prev_key]["page"], "number": None, "text": text}
        out.insert(out.index(by_key[prev_key]) + 1, recovered)
        by_key[n] = recovered
        logger.warning("recovered missing reference %s from the PDF's embedded text layer", n)
    return out


def _recover_missing_references(
    references: list[RawReference], pdf_path: Path | None
) -> list[RawReference]:
    """Recover numbered-bibliography entries the layout model skipped, from the PDF's
    embedded text layer (confirmed live: brunton-2016's entry 2 is printed in the PDF
    and readable via PyMuPDF, but PP-DocLayout-V3 produces no region for it).

    Born-digital PDFs only -- a scanned PDF has no text layer and falls straight
    through to the gap warning. Never touches unnumbered (author-year) bibliographies
    or ones with unclean keys, and never runs at all when there is no gap.
    """
    if pdf_path is None or not references:
        return references
    missing = _missing_reference_numbers(references)
    if not missing:
        return references
    try:
        import fitz  # PyMuPDF; already present transitively via glmocr

        with fitz.open(pdf_path) as doc:
            layer_text = "\n".join(str(page.get_text()) for page in doc)
    except Exception:  # no text layer / import failure: the gap warning still fires
        return references
    if not layer_text.strip():
        return references
    return _splice_missing_from_layer(references, layer_text, missing)


def _reference_sort_key(ref: RawReference) -> int | None:
    """Best-effort integer ordering key for one reference.

    Prefers the region-paired ``number`` field (see parse.py's
    ``_merge_reference_numbers``), but that pairing is the *uncommon* case in
    practice -- confirmed live on kalman-1960.pdf, where PP-DocLayout-V3 never
    produces a separate reference_number region at all; the marker is just the
    leading digits of the OCR'd text blob itself (e.g. "2 L. A. Zadeh..."). Falls
    back to parsing that leading number directly off the text before giving up.
    Returns ``None`` when neither source yields a clean integer.
    """
    raw = ref.get("number")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            return None
    n = leading_number(ref["text"])
    return int(n) if n is not None else None


def _sort_references_by_number(references: list[RawReference]) -> list[RawReference]:
    """Re-sort references by their best-effort numeric marker, when doing so is
    unambiguous.

    PP-DocLayout-V3's own region ``index`` (used to order regions before merging, see
    parse.py's ``_build_markdown``) isn't always reading order for a multi-column
    bibliography -- confirmed live on kalman-1960.pdf, where two side-by-side columns
    produced entries out of numeric order (2, 1, 3, 4, 5, 7, 6, ...). Since a numbered
    bibliography always prints in ascending order, re-sorting by the parsed number is a
    safe, unambiguous fix -- but only when *every* entry yields a clean integer key (see
    ``_reference_sort_key``) *and* the keys form one contiguous run (a numbered
    bibliography is always 1..n): a duplicate key (two entries garbled to the same
    number) or an outlier (an unnumbered entry whose text happens to start with a year,
    e.g. "2019 IEEE Conference on...") means the keys can't be trusted at all. A style
    with no numbers (author-year, e.g. fmech-07-655266) or a partial/garbled parse is
    left in detected order rather than guessing at a partial sort.
    """
    if not references:
        return references
    keys = [_reference_sort_key(ref) for ref in references]
    if any(key is None for key in keys):
        return references
    nums = [k for k in keys if k is not None]  # all clean past the guard
    if sorted(nums) != list(range(min(nums), min(nums) + len(nums))):
        return references  # duplicates or gaps -> don't trust the keys
    return [ref for _, ref in sorted(zip(nums, references, strict=True), key=lambda pair: pair[0])]


def render_references_markdown(references: list[RawReference]) -> str:
    """Plain markdown rendering of the raw bibliography: one entry per line, grouped
    under each page's own ``<page_number>`` marker, in reading order.

    Zero interpretation -- this is GLM-OCR's own OCR'd text, reassembled only using the
    number/content region pairing already done in parse.py's
    ``_merge_reference_numbers``. No schema, no external tool.
    """
    if not references:
        return ""
    by_page: dict[int, list[RawReference]] = {}
    for ref in references:
        by_page.setdefault(ref["page"], []).append(ref)

    parts: list[str] = []
    for page in sorted(by_page):
        lines = [
            f"[{ref['number']}] {ref['text']}" if ref["number"] else ref["text"]
            for ref in by_page[page]
        ]
        parts.append(page_marker(page))
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts)
