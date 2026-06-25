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

from .config import ChunkConfig

_PAGE = re.compile(r"<page_number>\s*(\d+)\s*</page_number>")
# boundary types in descending preference, with the regex that finds them
_BOUNDARIES = ((3, r"\n\n+"), (2, r"(?<=[.!?])\s+"), (1, r"\n"), (0, r" "))
_MODE = {3: "PARA", 2: "SENT", 1: "NL", 0: "WORD"}


@dataclass
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
    return [n.text for n in nodes]


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
    nums = [int(m.group(1)) for m in _PAGE.finditer(text)]
    return (min(nums), max(nums)) if nums else (None, None)


def chunk_markdown(markdown: str, cfg: ChunkConfig | None = None) -> list[Chunk]:
    """Split page-marked markdown into section-aware, overlapping chunks."""
    cfg = cfg or ChunkConfig()

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
                text=_PAGE.sub("", text).strip(),
                index=i,
                page_start=ps,
                page_end=pe,
                overlap_chars=ov_chars,
                overlap_mode=ov_mode,
            )
        )
    return chunks
