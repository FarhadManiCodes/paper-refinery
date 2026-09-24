"""Pure helpers of scripts/eval_models.py (the subcommands themselves call Gemini)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_models", Path(__file__).parents[1] / "scripts" / "eval_models.py"
)
em = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(em)

RAW = "[12] J. Smith and A. Doe. Learning to steer models. JMLR, 21(3):1-20, 2020."


def test_faithful_extraction_invents_nothing():
    item = {
        "title": "Learning to steer models",
        "year": 2020,
        "authors": [{"family": "Smith"}, {"family": "Doe"}],
        "volume": "21",
        "page": "1-20",
    }
    assert em.invented_fields(item, RAW) == []


def test_each_fabricated_field_is_flagged():
    item = {
        "title": "Controlling language models with feedback",
        "year": 2019,
        "doi": "10.1/xyz",
        "authors": [{"family": "Jones"}],
        "volume": "44",
    }
    assert em.invented_fields(item, RAW) == ["year", "doi", "volume", "author", "title"]


def test_newer_in_family_compares_versions_within_a_family_only():
    available = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.9-flash", "gemini-4-flash"]
    assert em.newer_in_family("gemini-3.8-flash", available) == [
        "gemini-3.9-flash",
        "gemini-4-flash",
    ]
    assert em.newer_in_family("gemini/gemini-3.5-flash-lite", available) == []
    assert em.newer_in_family("gemini-3-flash-preview", available) == [
        "gemini-3.8-flash",
        "gemini-3.9-flash",
        "gemini-4-flash",
    ]


def test_figure_picks_file_is_well_formed():
    import json

    picks = json.loads(em.FIGURE_PICKS.read_text())
    assert len(picks) == 10 and all({"doc", "figure", "kind"} <= set(p) for p in picks)


def test_typography_is_not_counted_as_invention():
    raw = "[3] C. O\u2019Neil. Weapons of math destruc- tion. Crown, 2016. doi:10.5555/3002861"
    item = {
        "title": "Weapons of math destruction",
        "year": 2016,
        "doi": "10.5555/3002861",
        "authors": [{"family": "O'Neil"}],
    }
    assert em.invented_fields(item, raw) == []
