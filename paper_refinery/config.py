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

    parse_mode: str = "parse_page_with_agent"  # agentic: best equations/tables
    save_images: bool = True  # extract figure images to disk
    extract_charts: bool = True  # also extract chart IMAGES (method-comparison plots)
    inline_images: bool = True  # reference figures inline in the markdown
    api_key_env: str = "LLAMA_API_KEY"
    # NOTE: `extract_charts` only saves a chart as an image so figures.py can describe
    # it. It is NOT `specialized_chart_parsing_*`, which fabricates precise numeric
    # tables from plotted curves and is intentionally never enabled.


@dataclass
class FigureConfig:
    """Gemini figure-description options."""

    model: str = "gemini-3-flash-preview"  # more accurate figure reading than 2.5-flash
    api_key_env: str = "GOOGLE_API_KEY"
    # Describe trends/comparisons; never invent numeric values read off curves.
    prompt: str = (
        "Describe this scientific figure for search and retrieval. State what is "
        "compared, the variables/axes, and the qualitative trends or conclusions. "
        "Do NOT report precise numeric values read off plotted curves — give ranges "
        "or directions only. Be concise (2-4 sentences)."
    )


@dataclass
class RefineryConfig:
    """Top-level config bundling each stage."""

    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    parse: ParseConfig = field(default_factory=ParseConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
