"""Tiny disk-cache primitives shared by the figure-description and API-response
caches (figures.py, citation_providers.py).

Each cache is keyed differently and has its own policy for *when* to read/write --
figures.py caches a definitive non-figure verdict as ``{}``, citation_providers.py
deliberately never caches a failure. Only the mechanical parts are identical and
shared here: turning a digest into a path, and safely reading/writing a JSON blob.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def cache_path(cache_dir: str, digest: str) -> Path | None:
    """Path for one cache entry, or None when caching is disabled (empty ``cache_dir``)."""
    if not cache_dir:
        return None
    return Path(cache_dir).expanduser() / f"{digest}.json"


def read_json(path: Path) -> object | None:
    """The cached value at ``path``, or None if missing or corrupt (never raises)."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: object) -> None:
    """Atomically write ``data`` as JSON to ``path``: serialize to a unique sibling temp
    file, then ``os.replace`` it into place (atomic within the directory's filesystem).

    A reader or a crash therefore never sees a half-written entry -- both caches write
    under a ThreadPoolExecutor, and a unique temp per write means two concurrent writers
    of the same key can't corrupt each other's file. A temp left behind by a killed
    process is inert (readers only ever open ``{digest}.json``); the cache dir is safe to
    delete anytime.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)  # don't leak the temp on serialize/rename failure
        raise
