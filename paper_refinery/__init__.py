"""paper-refinery — parse, figure-enrich, and chunk papers into RAG-ready chunks."""

from .citation_resolution import SourceMeta, to_papis_citations
from .cli import RefineResult, refine, refine_many

__version__ = "0.1.0"

__all__ = [
    "RefineResult",
    "SourceMeta",
    "refine",
    "refine_many",
    "to_papis_citations",
    "__version__",
]
