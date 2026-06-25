"""Tunable configuration for the refinery pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkConfig:
    """Section-aware chunking with a guaranteed soft-window overlap."""

    max_chars: int = 6000  # sub-split sections larger than this
    min_chars: int = 1000  # merge sections smaller than this into a neighbour
    overlap_lo: int = 300  # soft overlap window: minimum size
    overlap_hi: int = 700  # soft overlap window: maximum size
    overlap_ideal: int = 500  # preferred overlap size within the window


@dataclass
class ParseConfig:
    """LlamaParse options."""

    parse_mode: str = "parse_page_with_agent"  # agentic: best equations/tables (justified cost)
    inline_images: bool = True  # ![alt](src) placeholder at each figure (reliable anchor)
    take_screenshot: bool = True  # full-page renders (page_N.jpg) sent to Gemini per figure
    disable_image_extraction: bool = True  # skip LlamaParse's unreliable per-figure crops
    #   (independent of inline_images: placeholders still appear; only the crop files are skipped)
    api_key_env: str = "LLAMA_API_KEY"
    # Audited: we use only markdown + placeholders + page renders. No figure crops, charts,
    # specialized parsing, vendor models, or HTML tables -- nothing we'd pay for and discard.


@dataclass
class FigureConfig:
    """Gemini figure-description options."""

    model: str = "gemini-3-flash-preview"  # more accurate figure reading than 2.5-flash
    api_key_env: str = "GOOGLE_API_KEY"
    skip_marker: str = "NOT_A_FIGURE"  # Gemini returns this when the figure isn't found
    include_references: bool = False  # also feed in-text "Figure N" mentions as context
    # (off = caption-only context; cross-referencing is a future improvement)
    # The image is a full page; describe only the captioned figure. Trends, not numbers.
    prompt: str = (
        "The attached image is a full page from a scientific paper that may also "
        "contain body text, other figures, or tables. Describe ONLY the figure "
        "identified by the caption below, for search and retrieval: state what is "
        "compared, the variables/axes, and the qualitative trends or conclusions. "
        "Do NOT report precise numeric values read off plotted curves — give ranges "
        "or directions only. Be concise (2-4 sentences). "
        "If the page contains no figure matching that caption, reply with exactly: "
        "NOT_A_FIGURE"
    )


@dataclass
class RefineryConfig:
    """Top-level config bundling each stage."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
