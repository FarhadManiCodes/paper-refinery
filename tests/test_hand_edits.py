"""A full run must never silently destroy a hand-edited refinery.md."""

from __future__ import annotations

import pytest

from paper_refinery import cli
from paper_refinery.config import RefineryConfig


def _written_by_refinery(work, text="enriched markdown\n"):
    work.mkdir(exist_ok=True)
    (work / "refinery.md").write_text(text)
    cli._record_md_checksum(work, text)


def _backups(work, kind):
    return sorted(p.name for p in work.glob(f"refinery.md.{kind}-*"))


def test_untouched_refinery_md_is_replaced_without_ceremony(tmp_path):
    _written_by_refinery(tmp_path / "w")
    cli._guard_hand_edits(tmp_path / "w", RefineryConfig())
    assert not list((tmp_path / "w").glob("refinery.md.*-*"))


def test_hand_edited_refinery_md_stops_the_run_and_is_backed_up(tmp_path):
    work = tmp_path / "w"
    _written_by_refinery(work)
    (work / "refinery.md").write_text("enriched markdown, fixed by hand\n")
    with pytest.raises(cli.HandEditedMarkdown, match="--from chunk"):
        cli._guard_hand_edits(work, RefineryConfig())
    assert (work / "refinery.md").read_text() == "enriched markdown, fixed by hand\n"
    (backup,) = _backups(work, "hand-edited")
    assert (work / backup).read_text() == "enriched markdown, fixed by hand\n"


def test_overwrite_edits_lets_the_run_proceed_but_keeps_a_copy(tmp_path):
    work = tmp_path / "w"
    _written_by_refinery(work)
    (work / "refinery.md").write_text("edited\n")
    cli._guard_hand_edits(work, RefineryConfig(overwrite_edits=True))
    assert _backups(work, "hand-edited")


def test_refinery_md_from_before_checksums_is_backed_up_and_replaced(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "refinery.md").write_text("old run, maybe edited\n")
    cli._guard_hand_edits(work, RefineryConfig())
    (backup,) = _backups(work, "before")
    assert (work / backup).read_text() == "old run, maybe edited\n"


def test_no_refinery_md_yet_is_fine(tmp_path):
    cli._guard_hand_edits(tmp_path, RefineryConfig())


def test_the_guard_runs_before_any_paid_work(tmp_path, monkeypatch):
    work = tmp_path / "w"
    _written_by_refinery(work)
    (work / "refinery.md").write_text("edited\n")
    monkeypatch.setattr(cli, "_parse_maybe_split", lambda *a, **k: pytest.fail("OCR started"))
    with pytest.raises(cli.HandEditedMarkdown):
        cli._refine(
            tmp_path / "p.pdf", tmp_path / "o.json", tmp_path / "c.json", work, RefineryConfig()
        )


def test_repeated_refusals_keep_a_single_backup(tmp_path):
    work = tmp_path / "w"
    _written_by_refinery(work)
    (work / "refinery.md").write_text("edited\n")
    for _ in range(3):
        with pytest.raises(cli.HandEditedMarkdown):
            cli._guard_hand_edits(work, RefineryConfig())
    assert len(_backups(work, "hand-edited")) == 1
