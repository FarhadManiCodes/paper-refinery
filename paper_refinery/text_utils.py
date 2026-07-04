"""Small text-normalization primitives shared across the citation modules.

Kept separate from ``markers.py`` (specifically the page-boundary marker contract) and
deliberately dependency-free (stdlib only), so every citation module can import this
without pulling in anything heavier.
"""

from __future__ import annotations

import re
import unicodedata

# a bibliography marker at the start of a line/entry: "[12] " / "12. " / "7 "
LEADING_NUMBER_RE = re.compile(r"^\s*\[?(\d+)[\]. ]?\s")


def leading_number(text: str) -> str | None:
    """The leading marker's own printed digits (as a string), or None if absent."""
    m = LEADING_NUMBER_RE.match(text or "")
    return m.group(1) if m else None


def fold_name(name: str) -> str:
    """Lowercased, diacritic-folded form for surname comparison.

    OCR is inconsistent about diacritics across a page -- confirmed live on fmech,
    where the body prints "Höhn et al." but the bibliography region OCR'd the same
    surname as "Hohn". An exact string match would miss a genuine author/citation match.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()
