"""Section-aware chunking with a guaranteed soft-window overlap.

Validated policy:
  1. split markdown into sections on its headers (MarkdownNodeParser)
  2. sub-split any section larger than ``max_chars``
  3. merge any section smaller than ``min_chars`` into a neighbour
  4. prepend a guaranteed overlap (soft window ``[overlap_lo, overlap_hi]``) to every
     chunk, snapped to the best nearby boundary: paragraph > sentence > newline > word
  5. resolve each chunk's page range from ``<page_number>`` markers (body only), then
     strip the markers from the text
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

from .config import ChunkConfig
from .markers import PAGE_MARKER_RE

# boundary types in descending preference, with the regex that finds them
_BOUNDARIES = ((3, r"\n\n+"), (2, r"(?<=[.!?])\s+"), (1, r"\n"), (0, r" "))
_MODE = {3: "PARA", 2: "SENT", 1: "NL", 0: "WORD"}


@dataclass(slots=True)
class Chunk:
    """A single chunk emitted by the refinery and ingested by papis-ask."""

    text: str
    index: int
    page_start: int | None = None
    page_end: int | None = None
    overlap_chars: int = 0
    overlap_mode: str = "-"

    def name_for(self, docname: str) -> str:
        """paper-qa-style chunk name carrying the page range (used by papis-ask)."""
        if self.page_start is None:
            return f"{docname} chunk {self.index}"
        if self.page_start == self.page_end:
            return f"{docname} pages {self.page_start}"
        return f"{docname} pages {self.page_start}-{self.page_end}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "overlap_chars": self.overlap_chars,
            "overlap_mode": self.overlap_mode,
        }


def _split_sections(markdown: str) -> list[str]:
    nodes = MarkdownNodeParser().get_nodes_from_documents([Document(text=markdown)])
    # MarkdownNodeParser emits TextNodes (one per section); the isinstance keeps the
    # checker honest about `.text`, which lives on TextNode, not the BaseNode return type
    return [n.text for n in nodes if isinstance(n, TextNode)]


def _subsplit(text: str, cfg: ChunkConfig) -> list[str]:
    # SentenceSplitter is used only to *size* an over-long section; overlap is added
    # deterministically afterwards in `_overlap_before`, so chunk_overlap=0 here.
    sizer = SentenceSplitter(chunk_size=cfg.max_chars // 4, chunk_overlap=0)
    return sizer.split_text(text)


def _merge_small(pieces: list[str], cfg: ChunkConfig) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and (len(piece) < cfg.min_chars or len(merged[-1]) < cfg.min_chars):
            merged[-1] += "\n\n" + piece
        else:
            merged.append(piece)
    return merged


def _overlap_before(prev: str, cfg: ChunkConfig) -> tuple[str, str]:
    """An exact trailing window of ``prev`` (length within the soft window) snapped to
    the best nearby boundary. Returns ``(overlap_text, mode)``; the text is always an
    exact substring of ``prev``."""
    lo, hi, ideal = cfg.overlap_lo, cfg.overlap_hi, cfg.overlap_ideal
    if len(prev) <= lo:
        return prev, "ALL"
    win = prev[-hi:]
    n = len(win)
    pmin, pmax = max(0, n - hi), n - lo  # start positions giving overlap in [lo, hi]
    best: tuple[tuple[int, int], int] | None = None  # (score, position)
    for prio, pattern in _BOUNDARIES:
        for m in re.finditer(pattern, win):
            p = m.end()
            if pmin <= p <= pmax:
                score = (prio, -abs((n - p) - ideal))
                if best is None or score > best[0]:
                    best = (score, p)
    if best is None:  # no boundary in window -> hard cut at the ideal size
        return win[n - ideal :], "HARD"
    (prio, _), p = best
    return win[p:], _MODE[prio]


def _page_range(text: str) -> tuple[int | None, int | None]:
    nums = [int(m.group(1)) for m in PAGE_MARKER_RE.finditer(text)]
    return (min(nums), max(nums)) if nums else (None, None)


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
# at most a single top-level number: "5 References" is back matter, "1.8.4 References" (a
# C++ textbook section on references, live in gottschling-2021) is not
_BACK_MATTER_RE = re.compile(
    r"(?i)^(?:\d+\.?\s*)?(?:references|bibliography|works cited|literature cited"
    r"|reference list|(?:subject |author |name |general )?index)$"
)
# figure blocks survive inside dropped back matter: a figure can sit on a reference page
# (live: dong-2024's Figure 11 follows its REFERENCES heading)
_FIGURE_LINE_RE = re.compile(
    r"^(?:!\[FIGURE|> \*\*Figure description|\**\s*(?:FIGURE|Fig\.?)\s*\d)"
)
_INDEX_LETTER_RE = re.compile(
    r"(?i)^(?:[a-z]|\d+|symbols?|numbers?|numerals?|[a-z0-9]\s*[-–]\s*[a-z0-9])$"
)
_INDEX_ENTRY_RE = re.compile(
    r"^[^\n]{2,90}?,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\.?$"
)
# a citation's opening: "[12] ", "(3) ", "7. " or "Surname, J." / "Surname, Name"
_REFERENCE_START_RE = re.compile(
    r"^(?:(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.)\s*\S|[A-Z][\w'’-]+,\s*(?:[A-Z]\.|[A-Z][a-z]+))"
)
_YEAR_RE = re.compile(r"\b(?:1[89]|20)\d{2}[a-z]?\b")
_MAX_REFERENCE_CHARS = 600  # unheaded runs only: prose with a year is not a citation


def _is_back_matter_paragraph(s: str) -> bool:
    """Inside a headed References/Index section: does this paragraph look like an entry?
    No length cap here -- OCR often joins several references into one paragraph."""
    return bool(_INDEX_ENTRY_RE.match(s) or _REFERENCE_START_RE.match(s))


def _kept_from_dropped(lines: list[str]) -> tuple[list[str], str | None]:
    """What survives a dropped stretch: its figure blocks, and its last page marker, which
    the caller places in front of the next kept content -- left in place it would date the
    text *before* the stretch and stretch that chunk's page range across the whole index."""
    figures: list[str] = []
    for line in lines:
        if _FIGURE_LINE_RE.match(line.strip()):
            figures += ["", line, ""]
    markers = [m.group(0) for line in lines if (m := PAGE_MARKER_RE.search(line))]
    return figures, (markers[-1] if markers else None)


def _emit(out: list[str], item: str, pending: str | None, lines: bool = True) -> str | None:
    """Append ``item`` (a line, or a paragraph when ``lines`` is False); a pending page
    marker goes in front of the first real content -- after it when that content is a
    heading, so the heading's section owns the page."""
    if pending is None or not item.strip():
        out.append(item)
        return pending
    if PAGE_MARKER_RE.fullmatch(item.strip()):
        out.append(item)  # a newer marker supersedes the carried one
        return None
    gap = [""] if lines else []
    item = item.strip("\n")
    if _HEADING_RE.match(item.split("\n", 1)[0]):
        head, _, rest = item.partition("\n")
        out += [head, *gap, pending, *gap] + ([rest] if rest else [])
    else:
        out += [pending, *gap, item]
    return None


def _section_is_back_matter(lines: list[str]) -> bool:
    """Drop a headed section only if its content agrees with its heading: at least half of
    its entry-sized paragraphs look like references or index entries (or it is nearly
    empty). A section merely *titled* "References" or "Index" keeps its prose."""
    paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n")]
    body = [
        p
        for p in paragraphs
        if p
        and not PAGE_MARKER_RE.fullmatch(p)
        and not _FIGURE_LINE_RE.match(p)
        and not (_HEADING_RE.match(p) and _INDEX_LETTER_RE.match(_HEADING_RE.match(p).group(2)))
    ]
    if len(body) <= 1:
        return True
    return sum(_is_back_matter_paragraph(p) for p in body) * 2 >= len(body)


def _drop_back_matter(markdown: str) -> str:
    """Remove reference-list and back-of-book-index sections whose content confirms them.

    A section starts at a heading such as "References", "Bibliography" or "Index" and runs
    to the next heading of the same or a higher level; an index also swallows the letter
    headings ("A", ..., "Z", "Symbols", "1") that follow it at its own level.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    pending: str | None = None
    i = 0
    while i < len(lines):
        heading = _HEADING_RE.match(lines[i])
        title = heading.group(2).strip("*_ ") if heading else ""
        if not heading or not _BACK_MATTER_RE.match(title):
            pending = _emit(out, lines[i], pending)
            i += 1
            continue
        level, is_index = len(heading.group(1)), title.lower().endswith("index")
        j = i + 1
        while j < len(lines):
            nxt = _HEADING_RE.match(lines[j])
            if nxt and len(nxt.group(1)) <= level:
                letter = len(nxt.group(1)) == level and _INDEX_LETTER_RE.match(
                    nxt.group(2).strip("*_ ")
                )
                if not (is_index and letter):
                    break
            j += 1
        section = lines[i + 1 : j]
        if _section_is_back_matter(section):
            figures, marker = _kept_from_dropped(section)
            out += figures
            pending = marker or pending
        else:
            for line in lines[i:j]:
                pending = _emit(out, line, pending)
        i = j
    return "\n".join(out)  # a marker still pending had no content after it: dropped


_MIN_INDEX_RUN, _MIN_REFERENCE_RUN = 20, 10
# inside a run, tolerate this many consecutive short non-matching paragraphs: live index
# entries also read "PRIM, see Patient rule induction method" or "Spline, 186 additive,
# 297-299 ..." (Hastie), which a strict entry pattern misses. Headings never bridge.
_RUN_GAP, _GAP_MAX_CHARS = 2, 200


def _paragraph_kind(paragraph: str) -> str:
    s = paragraph.strip()
    if not s or PAGE_MARKER_RE.fullmatch(s) or _FIGURE_LINE_RE.match(s):
        return "neutral"  # kept (see _kept_from_dropped), and does not end a run
    if _HEADING_RE.match(s):
        return "heading"
    if _INDEX_ENTRY_RE.match(s):
        return "index"
    if len(s) <= _MAX_REFERENCE_CHARS and _REFERENCE_START_RE.match(s) and _YEAR_RE.search(s):
        return "reference"
    return "body"


def _run_end(paragraphs: list[str], kinds: list[str], start: int) -> tuple[int, int]:
    """(end, matches) of the run of ``kinds[start]`` beginning at ``start``: it may bridge
    up to ``_RUN_GAP`` short non-matching paragraphs, and ends at its last match."""
    kind, count, end, gap = kinds[start], 0, start, 0
    for j in range(start, len(paragraphs)):
        if kinds[j] == kind:
            count, gap, end = count + 1, 0, j + 1
        elif kinds[j] == "neutral":
            continue
        elif kinds[j] != "heading" and gap < _RUN_GAP and len(paragraphs[j]) <= _GAP_MAX_CHARS:
            gap += 1
        else:
            break
    return end, count


def _drop_unheaded_runs(markdown: str) -> str:
    """Drop long unbroken runs of index entries ("Lasso, 68, 86-93") or reference entries,
    which some books print with no heading at all. Short runs are left alone: body text
    never holds 10+ consecutive citation-shaped or 20+ index-shaped paragraphs."""
    paragraphs = markdown.split("\n\n")
    kinds = [_paragraph_kind(p) for p in paragraphs]
    out: list[str] = []
    pending: str | None = None
    i = 0
    while i < len(paragraphs):
        if kinds[i] not in ("index", "reference"):
            pending = _emit(out, paragraphs[i], pending, lines=False)
            i += 1
            continue
        end, count = _run_end(paragraphs, kinds, i)
        if count >= (_MIN_INDEX_RUN if kinds[i] == "index" else _MIN_REFERENCE_RUN):
            figures, marker = _kept_from_dropped(paragraphs[i:end])
            out += [f for f in figures if f]  # figure lines, as their own paragraphs
            pending = marker or pending
        else:
            for para in paragraphs[i:end]:
                pending = _emit(out, para, pending, lines=False)
        i = max(end, i + 1)
    return "\n\n".join(out)


def chunk_markdown(markdown: str, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Split page-marked markdown into section-aware, overlapping chunks."""
    cfg = cfg or ChunkConfig()
    if cfg.drop_back_matter:
        markdown = _drop_unheaded_runs(_drop_back_matter(markdown))

    # 1-3: structure (section split -> sub-split big -> merge small)
    pieces: list[str] = []
    for section in _split_sections(markdown):
        pieces += [section] if len(section) <= cfg.max_chars else _subsplit(section, cfg)
    pieces = _merge_small(pieces, cfg)

    # 4-5: overlap + page resolution
    chunks: list[Chunk] = []
    last_page: int | None = None
    for i, body in enumerate(pieces):
        # pages come from the body only, *before* prepending overlap, so a chunk
        # never inherits a page number from the previous chunk's tail.
        ps, pe = _page_range(body)
        if ps is None:
            ps = pe = last_page
        else:
            last_page = pe

        if i == 0:
            text, ov_chars, ov_mode = body, 0, "-"
        else:
            overlap, ov_mode = _overlap_before(pieces[i - 1], cfg)
            text, ov_chars = overlap + "\n\n" + body, len(overlap)

        chunks.append(
            Chunk(
                text=PAGE_MARKER_RE.sub("", text).strip(),
                index=i,
                page_start=ps,
                page_end=pe,
                overlap_chars=ov_chars,
                overlap_mode=ov_mode,
            )
        )
    return chunks
