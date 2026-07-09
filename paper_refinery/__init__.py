"""paper-refinery — parse, figure-enrich, and chunk papers into RAG-ready chunks."""

from .citation_resolution import SourceMeta
from .cli import RefineResult, refine, refine_many

__version__ = "0.1.0"

__all__ = ["RefineResult", "SourceMeta", "refine", "refine_many", "__version__"]
