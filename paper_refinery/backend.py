"""OCR backend lifecycle: our own llama-server (inference) + the glmocr SDK bound to it.

Split out of parse.py so the expensive fixed cost -- spawning llama-server, loading the
GLM-OCR GGUF weights, initializing PP-DocLayout-V3 -- is paid once per *session*, not
once per PDF. ``parse.parse_pdf`` spawns its own backend when not given one (the CLI's
single-PDF case); a batch caller holds one open across many PDFs::

    with ocr_backend(cfg) as backend:
        for pdf in pdfs:
            parse_pdf(pdf, image_dir, cfg, backend=backend)

Nothing in here interprets OCR output -- region dispatch and markdown assembly stay in
parse.py; this module only owns process/connection lifetime.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ParseConfig, load_config

if TYPE_CHECKING:
    from glmocr import GlmOcr

# "chart"/"image" in glmocr's id2label (config.yaml); re-check on upgrade. Lives here
# (not parse.py) because it's only used to build the backend's layout-config overrides.
_FIGURE_CLASS_IDS = (3, 14)


@dataclass
class OcrBackend:
    """A live backend: a glmocr parser bound to our own running llama-server.

    ``server`` is exposed (not just the parser) so the parse watchdog can kill the
    process when a run wedges -- killing the server is what breaks glmocr's in-flight
    HTTP calls loose.
    """

    parser: GlmOcr  # the entered glmocr.GlmOcr (imported lazily; typed via TYPE_CHECKING)
    server: subprocess.Popen


def _ensure_port_free(cfg: ParseConfig) -> None:
    """Fail loudly if something is already listening on the configured port.

    Without this there's a race: _wait_for_health polls ``/health`` while our own
    process is still starting -- a leftover llama-server (e.g. orphaned by an earlier
    interrupted run) on the same port answers 200 first, our process dies with a bind
    error a moment later, and the whole PDF silently parses against the stale server.
    A raw socket connect, not an HTTP call: a half-loaded server answering 503 is
    still "occupied".
    """
    with socket.socket() as s:
        s.settimeout(1.0)
        if s.connect_ex((cfg.host, cfg.port)) == 0:
            raise RuntimeError(
                f"something is already listening on {cfg.host}:{cfg.port} "
                "(an orphaned llama-server?) -- kill it or change ParseConfig.port"
            )


@contextmanager
def _llama_server(cfg: ParseConfig) -> Iterator[subprocess.Popen]:
    """Spawn llama-server serving GLM-OCR, wait for it to become healthy, tear it down."""
    if not cfg.model_path:
        raise RuntimeError("ParseConfig.model_path is not set (GLM-OCR GGUF weights)")
    if not cfg.mmproj_path:
        raise RuntimeError("ParseConfig.mmproj_path is not set (GLM-OCR GGUF vision projector)")
    _ensure_port_free(cfg)

    cmd = [
        cfg.llama_server_bin,
        "-m",
        cfg.model_path,
        "--mmproj",
        cfg.mmproj_path,
        "--host",
        cfg.host,
        "--port",
        str(cfg.port),
        "-ngl",
        str(cfg.n_gpu_layers),
        *cfg.extra_server_args,
    ]
    fd, log_path_str = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
    log_path = Path(log_path_str)
    try:
        with open(fd, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            try:
                _wait_for_health(proc, cfg, log_path)
                yield proc
            finally:
                # sole place that owns process teardown, for every exit path (including
                # a health-check timeout/early-exit raised from _wait_for_health)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
    finally:
        log_path.unlink(missing_ok=True)


def _read_log(log_path: Path) -> str:
    return log_path.read_text(errors="replace")


def _wait_for_health(proc: subprocess.Popen, cfg: ParseConfig, log_path: Path) -> None:
    url = f"http://{cfg.host}:{cfg.port}/health"
    deadline = time.monotonic() + cfg.startup_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}); log:\n{_read_log(log_path)}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"llama-server did not become healthy within {cfg.startup_timeout_s}s; log:\n"
        f"{_read_log(log_path)}"
    )


def _dotted_overrides(cfg: ParseConfig) -> dict:
    """Build glmocr's ``_dotted`` config-override dict for this run.

    Widens the layout-detection box only for figure/chart classes (``cfg.figure_crop_margin``,
    per-class via ``pipeline.layout.layout_unclip_ratio``) before layering on
    ``cfg.glmocr_config_overrides`` -- an explicit user override always wins over our
    computed default for the same dotted path.
    """
    margin = (cfg.figure_crop_margin, cfg.figure_crop_margin)
    dotted = {
        "pipeline.layout.layout_unclip_ratio": dict.fromkeys(_FIGURE_CLASS_IDS, margin),
    }
    dotted.update(cfg.glmocr_config_overrides)
    return dotted


@contextmanager
def ocr_backend(cfg: ParseConfig | None = None) -> Iterator[OcrBackend]:
    """Spawn llama-server + GlmOcr once and yield a live ``OcrBackend``.

    The one place the glmocr SDK is imported/constructed. Reuse the yielded backend
    across as many ``parse_pdf`` calls as needed; everything is torn down on exit.
    """
    from glmocr import GlmOcr

    cfg = cfg or load_config().parse
    with (
        _llama_server(cfg) as proc,
        GlmOcr(
            mode="selfhosted",
            ocr_api_host=cfg.host,
            ocr_api_port=cfg.port,
            layout_device=cfg.layout_device,
            _dotted=_dotted_overrides(cfg),
        ) as parser,
    ):
        yield OcrBackend(parser=parser, server=proc)
