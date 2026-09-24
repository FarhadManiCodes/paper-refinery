"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Every "~/..." default (the api, figure and extraction caches) points into a fresh
    temporary home, so no test reads a real cache or writes into one."""
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
