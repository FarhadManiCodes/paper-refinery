"""paper-refinery — parse, figure-enrich, and chunk papers into RAG-ready chunks."""

from importlib.metadata import version as _version

from .citation_resolution import SourceMeta, to_papis_citations
from .cli import RefineResult, refine, refine_many

# single source of truth is pyproject.toml; a hardcoded copy here drifted (0.2.1 vs 0.3.0)
__version__ = _version("paper-refinery")

__all__ = [
    "RefineResult",
    "SourceMeta",
    "refine",
    "refine_many",
    "to_papis_citations",
    "__version__",
]
