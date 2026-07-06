"""Tests for the parse-stage checkpoint (parse_cache.py): a warm run must skip OCR and
reproduce the parse result, invalidate correctly on pdf/config/version changes, and restore
the raw crops that enrich would otherwise have renamed away.

Driven by a counting fake backend (same shape as test_parse.py's) so "did OCR run again?"
is just ``parser.calls``. No llama-server, no weights, no network.
"""

from __future__ import annotations

from PIL import Image

from paper_refinery import parse_cache
from paper_refinery.backend import OcrBackend
from paper_refinery.config import ParseConfig
from paper_refinery.parse_cache import parse_pdf_cached


def _region(label: str, content: str = "", **extra) -> dict:
    return {"native_label": label, "label": label, "content": content, "index": 0, **extra}


class _FakeResult:
    def __init__(self, json_result, image_files=None):
        self.json_result = json_result
        self.image_files = image_files or {}


class _CountingParser:
    """Records how many times OCR actually ran -- the whole point of the checkpoint."""

    def __init__(self, result):
        self._result = result
        self.calls = 0

    def parse(self, path):
        self.calls += 1
        return self._result


class _FakeServer:
    def kill(self):
        pass


def _backend(pages, image_files=None):
    parser = _CountingParser(_FakeResult(pages, image_files))
    return OcrBackend(parser=parser, server=_FakeServer()), parser


def _pdf(tmp_path, data: bytes = b"%PDF-1.4 fake"):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(data)
    return pdf


_PAGES = [[_region("doc_title", "A Paper", index=0), _region("text", "Body text.", index=1)]]


def test_second_run_hits_cache_and_skips_ocr(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)
    cfg = ParseConfig()

    r1 = parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 1
    r2 = parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 1  # warm run made zero OCR calls
    assert r2.markdown == r1.markdown
    assert "# A Paper" in r2.markdown


def test_force_bypasses_the_cache(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)
    cfg = ParseConfig()

    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend, force=True)
    assert parser.calls == 2


def test_pdf_content_change_invalidates(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)
    cfg = ParseConfig()

    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    pdf.write_bytes(b"%PDF-1.4 DIFFERENT")
    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 2


def test_output_affecting_config_change_invalidates(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)

    parse_pdf_cached(pdf, tmp_path, ParseConfig(), backend=backend)
    # crop margin changes OCR output -> part of the parse signature -> miss
    parse_pdf_cached(pdf, tmp_path, ParseConfig(figure_crop_margin=1.5), backend=backend)
    assert parser.calls == 2


def test_perf_only_config_change_keeps_cache(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)

    parse_pdf_cached(pdf, tmp_path, ParseConfig(), backend=backend)
    # ports / gpu layers / timeouts are performance, not output -> still a hit
    cfg2 = ParseConfig(port=9999, n_gpu_layers=1, parse_timeout_s=60.0)
    parse_pdf_cached(pdf, tmp_path, cfg2, backend=backend)
    assert parser.calls == 1


def test_checkpoint_version_bump_invalidates(tmp_path, monkeypatch):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)
    cfg = ParseConfig()

    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    monkeypatch.setattr(parse_cache, "CHECKPOINT_VERSION", parse_cache.CHECKPOINT_VERSION + 1)
    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 2


def test_corrupt_checkpoint_is_a_miss(tmp_path):
    pdf = _pdf(tmp_path)
    backend, parser = _backend(_PAGES)
    cfg = ParseConfig()

    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    (tmp_path / "parse_cache" / "parse.json").write_text("{ not valid json")
    parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 2


def test_hit_restores_raw_crops_after_enrich_renamed_them(tmp_path):
    # the crux: enrich moves page_N_fig_i.png -> fig_*.png in place, so a warm run's
    # reloaded ParseResult would point at a file that no longer exists. The checkpoint
    # keeps an immutable copy and restores it into figures/ on a hit.
    pdf = _pdf(tmp_path)
    img = Image.new("RGB", (4, 4), "white")
    pages = [[_region("image", "", bbox_2d=[0, 0, 4, 4], image_path="imgs/cropped_page0_idx0.jpg")]]
    backend, parser = _backend(pages, image_files={"cropped_page0_idx0.jpg": img})
    cfg = ParseConfig()

    r1 = parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    figures = tmp_path / "figures"
    raw = figures / "page_1_fig_0.png"
    assert raw.exists()
    assert r1.figure_crops[1][0].path == raw
    assert r1.figure_crops[1][0].bbox == (0.0, 0.0, 4.0, 4.0)

    # simulate enrich renaming the crop away
    raw.replace(figures / "fig_1.png")
    assert not raw.exists()

    r2 = parse_pdf_cached(pdf, tmp_path, cfg, backend=backend)
    assert parser.calls == 1  # no OCR
    assert raw.exists()  # raw crop restored from the snapshot
    assert not (figures / "fig_1.png").exists()  # stale renamed crop cleared
    assert r2.figure_crops[1][0].path == raw
    assert r2.figure_crops[1][0].bbox == (0.0, 0.0, 4.0, 4.0)
    assert str(raw) in r2.markdown  # placeholder link matches the restored crop
