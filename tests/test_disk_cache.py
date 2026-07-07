"""Tests for the shared disk-cache primitives (path building, tolerant reads, atomic writes)."""

import pytest

from paper_refinery.disk_cache import cache_path, read_json, write_json


def test_cache_path_builds_digest_filename(tmp_path):
    assert cache_path(str(tmp_path), "abc123") == tmp_path / "abc123.json"


def test_cache_path_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cache_path("~/cache", "d") == tmp_path / "cache" / "d.json"


def test_cache_path_none_when_caching_disabled():
    # an empty cache_dir is the "caching off" signal -- no path, no writes
    assert cache_path("", "abc") is None


def test_write_then_read_roundtrips(tmp_path):
    path = tmp_path / "e.json"
    payload = {"a": 1, "nested": ["x", None, 2.5], "u": "café"}
    write_json(path, payload)
    assert read_json(path) == payload


def test_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "e.json"
    write_json(path, {"ok": True})
    assert path.exists()
    assert read_json(path) == {"ok": True}


def test_write_overwrites_existing(tmp_path):
    path = tmp_path / "e.json"
    write_json(path, {"v": 1})
    write_json(path, {"v": 2})
    assert read_json(path) == {"v": 2}


def test_write_leaves_no_temp_behind(tmp_path):
    path = tmp_path / "e.json"
    write_json(path, {"v": 1})
    # the atomic temp (mkstemp suffix ".tmp") must be renamed into place, never left over
    assert list(tmp_path.glob("*.tmp")) == []
    assert [p.name for p in tmp_path.iterdir()] == ["e.json"]


def test_read_missing_returns_none(tmp_path):
    assert read_json(tmp_path / "nope.json") is None


def test_read_corrupt_returns_none_never_raises(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is not: valid json")
    assert read_json(path) is None


def test_write_of_unserializable_raises_and_leaks_no_temp(tmp_path):
    path = tmp_path / "e.json"
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})  # not JSON-serializable
    assert not path.exists()  # nothing partial written at the target
    assert list(tmp_path.glob("*.tmp")) == []  # temp cleaned up on the failure path


def test_failed_write_does_not_corrupt_existing_entry(tmp_path):
    path = tmp_path / "e.json"
    write_json(path, {"good": 1})
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})
    # the previous good entry is untouched: os.replace only happens after a clean serialize
    assert read_json(path) == {"good": 1}
    assert list(tmp_path.glob("*.tmp")) == []
