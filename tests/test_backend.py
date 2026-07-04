"""Tests for the OCR backend lifecycle (llama-server spawn/health/teardown, glmocr
config overrides). Process-free: the server is faked or never reached."""

from __future__ import annotations

import socket
import subprocess

import pytest

from paper_refinery.backend import (
    _dotted_overrides,
    _ensure_port_free,
    _llama_server,
)
from paper_refinery.config import ParseConfig

# ---------------------------------------------------------------------------
# _ensure_port_free
# ---------------------------------------------------------------------------


def test_ensure_port_free_raises_when_port_occupied():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="already listening"):
            _ensure_port_free(ParseConfig(port=port))


def test_ensure_port_free_passes_on_free_port():
    # bind-then-close to find a port that's definitely free right now
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    _ensure_port_free(ParseConfig(port=port))  # must not raise


# ---------------------------------------------------------------------------
# _llama_server
# ---------------------------------------------------------------------------


def test_llama_server_requires_model_path():
    with pytest.raises(RuntimeError, match="model_path"):
        with _llama_server(ParseConfig(mmproj_path="/x/mmproj.gguf")):
            pass


def test_llama_server_requires_mmproj_path():
    with pytest.raises(RuntimeError, match="mmproj_path"):
        with _llama_server(ParseConfig(model_path="/x/model.gguf")):
            pass


def test_llama_server_raises_cleanly_on_early_exit(monkeypatch):
    class FakeProc:
        def __init__(self, *a, **k):
            self.returncode = 1

        def poll(self):
            return self.returncode

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

        def kill(self):
            pass

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: FakeProc())
    # uncommon port: the pre-flight _ensure_port_free must not trip over a real
    # llama-server that happens to be running on the default 8080 during tests
    cfg = ParseConfig(
        model_path="/x/model.gguf",
        mmproj_path="/x/mmproj.gguf",
        startup_timeout_s=5,
        port=59173,
    )
    with pytest.raises(RuntimeError, match="exited early"):
        with _llama_server(cfg):
            pass


# ---------------------------------------------------------------------------
# _dotted_overrides
# ---------------------------------------------------------------------------


def test_dotted_overrides_widens_only_figure_class_ids():
    dotted = _dotted_overrides(ParseConfig(figure_crop_margin=1.1))
    ratios = dotted["pipeline.layout.layout_unclip_ratio"]
    assert ratios == {3: (1.1, 1.1), 14: (1.1, 1.1)}


def test_dotted_overrides_lets_explicit_override_win():
    cfg = ParseConfig(
        figure_crop_margin=1.1,
        glmocr_config_overrides={"pipeline.layout.layout_unclip_ratio": 1.0},
    )
    dotted = _dotted_overrides(cfg)
    assert dotted["pipeline.layout.layout_unclip_ratio"] == 1.0


def test_dotted_overrides_keeps_unrelated_user_overrides():
    cfg = ParseConfig(glmocr_config_overrides={"pipeline.max_workers": 1})
    dotted = _dotted_overrides(cfg)
    assert dotted["pipeline.max_workers"] == 1
    assert "pipeline.layout.layout_unclip_ratio" in dotted
