"""paper-refinery — parse, figure-enrich, and chunk papers into RAG-ready chunks."""

from .cli import RefineResult, refine, refine_many

__version__ = "0.1.0"

__all__ = ["RefineResult", "refine", "refine_many", "__version__"]
