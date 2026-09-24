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
_BACK_MATTER_RE = re.compile(
    r"(?i)^(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography|works cited|literature cited"
    r"|reference list|(?:subject |author |name |general )?index)$"
)
# figure blocks survive inside dropped back matter: a figure can sit on a reference page
# (live: dong-2024's Figure 11 follows its REFERENCES heading)
_FIGURE_LINE_RE = re.compile(
    r"^(?:!\[FIGURE|> \*\*Figure description|\**\s*(?:FIGURE|Fig\.?)\s*\d)"
)
_INDEX_LETTER_RE = re.compile(r"(?i)^(?:[a-z]|symbols?|numbers?|numerals?|[a-z]\s*[-–]\s*[a-z])$")


def _drop_back_matter(markdown: str) -> str:
    """Remove reference-list and back-of-book-index sections, keeping page markers.

    A section starts at a heading such as "References", "Bibliography" or "Index" and runs
    to the next heading of the same or a higher level; an index also swallows the
    single-letter headings ("A", "B", ..., "Symbols") that follow it at its own level.
    Page markers inside a dropped section are kept, so text after it keeps its pages.
    """
    out: list[str] = []
    dropping_level: int | None = None
    in_index = False
    for line in markdown.split("\n"):
        heading = _HEADING_RE.match(line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip("*_ ")
            if dropping_level is not None and level <= dropping_level:
                if in_index and level == dropping_level and _INDEX_LETTER_RE.match(title):
                    continue  # still inside the index's A-Z run
                dropping_level, in_index = None, False
            if dropping_level is None and _BACK_MATTER_RE.match(title):
                dropping_level, in_index = level, title.lower().endswith("index")
                continue
        if dropping_level is not None:
            if PAGE_MARKER_RE.search(line):
                out.append(PAGE_MARKER_RE.search(line).group(0))
            elif _FIGURE_LINE_RE.match(line.strip()):
                out += ["", line, ""]
            continue
        out.append(line)
    return "\n".join(out)


_INDEX_ENTRY_RE = re.compile(
    r"^[^\n]{2,90}?,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)*\.?$"
)
_REFERENCE_ENTRY_RE = re.compile(
    r"^(?:\[\d{1,4}\]|\(\d{1,4}\)|\d{1,4}\.)?\s*[A-Z][^\n]{10,}\b(?:1[89]|20)\d{2}[a-z]?\b"
)
_MIN_INDEX_RUN, _MIN_REFERENCE_RUN = 20, 10
# inside a run, tolerate this many consecutive short non-matching paragraphs: live index
# entries also read "PRIM, see Patient rule induction method" or "Spline, 186 additive,
# 297-299 ..." (Hastie), which a strict entry pattern misses
_RUN_GAP, _GAP_MAX_CHARS = 2, 200


def _paragraph_kind(paragraph: str) -> str:
    s = paragraph.strip()
    if not s or PAGE_MARKER_RE.fullmatch(s) or _FIGURE_LINE_RE.match(s):
        return "neutral"  # kept, and does not end a run
    if _INDEX_ENTRY_RE.match(s):
        return "index"
    if _REFERENCE_ENTRY_RE.match(s):
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
        elif gap < _RUN_GAP and len(paragraphs[j].strip()) <= _GAP_MAX_CHARS:
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
    drop = [False] * len(paragraphs)
    i = 0
    while i < len(paragraphs):
        if kinds[i] not in ("index", "reference"):
            i += 1
            continue
        end, count = _run_end(paragraphs, kinds, i)
        if count >= (_MIN_INDEX_RUN if kinds[i] == "index" else _MIN_REFERENCE_RUN):
            for k in range(i, end):
                drop[k] = kinds[k] != "neutral"  # page markers and figures stay
        i = max(end, i + 1)
    return "\n\n".join(p for p, d in zip(paragraphs, drop, strict=True) if not d)


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
