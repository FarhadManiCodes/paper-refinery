"""`--from review`: write only the sibling <stem>.md review copy from a saved refinery.md."""

import os
import socket

import pytest
from click.testing import CliRunner
from mathunicode import collapse_math_blocks

from paper_refinery import cli
from paper_refinery.chunker import Chunk
from paper_refinery.config import RefineryConfig
from paper_refinery.parse import ParseResult

# multi-line display math (what collapse_math_blocks rewrites) between one-based page markers
REFINERY_MD = (
    "<page_number>1</page_number>\n\n# Title\n\nIntro text.\n\n"
    "$$\n\\int_0^1 x\\,dx\n= \\frac12\n$$\n\n"
    "<page_number>2</page_number>\n\nSecond page, with $a+b$ inline.\n"
)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """Pin the locale/colour environment and make any config, key or network access fatal."""
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("NO_COLOR", "1")
    for var in ("GOOGLE_API_KEY", "ZHIPU_API_KEY", "OPENALEX_API_KEY", "S2_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    def forbidden(*a, **k):
        raise AssertionError("--from review must not touch config, keys or the network")

    monkeypatch.setattr(cli, "load_config", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def _paper(tmp_path, stem="p", md=REFINERY_MD):
    pdf = tmp_path / f"{stem}.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    wd = pdf.with_suffix(".refinery")
    wd.mkdir()
    (wd / "refinery.md").write_text(md)
    return pdf, wd


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mode)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_review_copy_equals_the_full_run_transformation(tmp_path):
    pdf, _ = _paper(tmp_path)
    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert result.exit_code == 0, result.output
    review = pdf.with_suffix(".md").read_text()
    assert review == collapse_math_blocks(REFINERY_MD)
    assert review != REFINERY_MD  # the fixture really exercises the collapse


def test_full_run_and_review_stage_write_identical_review_copy(tmp_path, monkeypatch):
    # run the real full-run wiring (stages mocked) and the review stage on its refinery.md
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(cli, "page_count", lambda pdf: 1)
    monkeypatch.setattr(cli, "load_config", lambda: RefineryConfig())
    monkeypatch.setattr(
        cli, "parse_pdf_cached", lambda p, d, c, force=False: ParseResult(markdown="MD")
    )
    monkeypatch.setattr(cli, "make_client", lambda cfg: object())
    monkeypatch.setattr(cli, "enrich_markdown", lambda parsed, cfg, describe: REFINERY_MD)
    monkeypatch.setattr(cli, "chunk_markdown", lambda md, cfg: [Chunk(md, 0, 1, 1)])
    result = CliRunner().invoke(cli.main, [str(pdf)])
    assert result.exit_code == 0, result.output
    full = pdf.with_suffix(".md").read_bytes()

    pdf.with_suffix(".md").unlink()
    monkeypatch.setattr(cli, "load_config", lambda: pytest.fail("config loaded"))
    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert result.exit_code == 0, result.output
    assert pdf.with_suffix(".md").read_bytes() == full


def test_page_markers_survive_the_review_copy(tmp_path):
    pdf, _ = _paper(tmp_path)
    assert CliRunner().invoke(cli.main, [str(pdf), "--from", "review"]).exit_code == 0
    review = pdf.with_suffix(".md").read_text()
    assert "<page_number>1</page_number>" in review
    assert "<page_number>2</page_number>" in review


def test_only_the_review_copy_is_written(tmp_path):
    pdf, wd = _paper(tmp_path)
    # stand-ins for the artifacts a full run would have written
    pdf.with_suffix(".chunks.json").write_text('{"chunks": []}')
    pdf.with_suffix(".citations.json").write_text('{"references": []}')
    (wd / "resolution_report.txt").write_text("report")
    before = _snapshot(tmp_path)
    md_bytes = (wd / "refinery.md").read_bytes()
    md_mode = (wd / "refinery.md").stat().st_mode

    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert result.exit_code == 0, result.output

    after = _snapshot(tmp_path)
    assert set(after) - set(before) == {"p.md"}  # nothing else created (no .tmp, no checksum)
    for name, state in before.items():
        assert after[name] == state  # nothing modified, refinery.md byte- and mode-identical
    assert (wd / "refinery.md").read_bytes() == md_bytes
    assert (wd / "refinery.md").stat().st_mode == md_mode


def test_missing_refinery_md_fails_with_one_line_and_writes_nothing(tmp_path):
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert result.exit_code != 0
    assert "refinery.md" in result.output and "missing" in result.output
    assert len(result.output.strip().splitlines()) == 1
    assert not pdf.with_suffix(".md").exists()
    assert not pdf.with_suffix(".refinery").exists()


def test_interrupted_write_leaves_no_partial_file(tmp_path, monkeypatch):
    pdf, _ = _paper(tmp_path)
    pdf.with_suffix(".md").write_text("previous review copy")

    def die(src, dst):
        raise OSError("disk went away")

    monkeypatch.setattr(os, "replace", die)
    result = CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert result.exit_code != 0
    assert pdf.with_suffix(".md").read_text() == "previous review copy"  # untouched
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up


def test_interrupted_first_write_creates_no_review_copy(tmp_path, monkeypatch):
    pdf, _ = _paper(tmp_path)
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(OSError("boom")))
    CliRunner().invoke(cli.main, [str(pdf), "--from", "review"])
    assert not pdf.with_suffix(".md").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_batch_writes_each_review_copy(tmp_path):
    a, _ = _paper(tmp_path, "a")
    b, _ = _paper(tmp_path, "b")
    result = CliRunner().invoke(cli.main_many, [str(a), str(b), "--from", "review"])
    assert result.exit_code == 0, result.output
    assert "wrote review copy for 2/2 papers" in result.output
    for p in (a, b):
        assert p.with_suffix(".md").read_text() == collapse_math_blocks(REFINERY_MD)


def test_batch_exits_nonzero_if_any_paper_lacks_refinery_md(tmp_path):
    a, _ = _paper(tmp_path, "a")
    b = tmp_path / "b.pdf"
    b.write_bytes(b"%PDF-1.4 fake")
    result = CliRunner().invoke(cli.main_many, [str(a), str(b), "--from", "review"])
    assert result.exit_code == 1
    assert "wrote review copy for 1/2 papers" in result.output
    assert a.with_suffix(".md").exists() and not b.with_suffix(".md").exists()
