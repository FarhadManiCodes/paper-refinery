"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Every "~/..." default (the api, figure and extraction caches, the config and its
    secrets) points into a fresh temporary home, so no test reads real caches or keys."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    # and the real config and secrets are never loaded: these win over HOME
    for var in (
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "PAPER_REFINERY_CONFIG",
        "PAPER_REFINERY_SECRETS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
