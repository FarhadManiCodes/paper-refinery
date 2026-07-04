"""Tiny disk-cache primitives shared by the figure-description and API-response
caches (figures.py, citation_providers.py).

Each cache is keyed differently and has its own policy for *when* to read/write --
figures.py caches a definitive non-figure verdict as ``{}``, citation_providers.py
deliberately never caches a failure. Only the mechanical parts are identical and
shared here: turning a digest into a path, and safely reading/writing a JSON blob.
"""

from __future__ import annotations

import json
from pathlib import Path


def cache_path(cache_dir: str, digest: str) -> Path | None:
    """Path for one cache entry, or None when caching is disabled (empty ``cache_dir``)."""
    if not cache_dir:
        return None
    return Path(cache_dir).expanduser() / f"{digest}.json"


def read_json(path: Path):
    """The cached value at ``path``, or None if missing or corrupt (never raises)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
