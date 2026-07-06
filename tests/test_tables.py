"""Tests for GLM-OCR HTML-table -> markdown conversion (``tables.py``)."""

from __future__ import annotations

from paper_refinery.tables import html_table_to_markdown


def test_html_table_simple_roundtrip():
    html = "<table><tr><th>Name</th><th>Value</th></tr><tr><td>a</td><td>1</td></tr></table>"
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| a | 1 |"


def test_html_table_duplicates_colspan_value():
    html = "<table><tr><td colspan='2'>Header</td></tr><tr><td>a</td><td>b</td></tr></table>"
    md = html_table_to_markdown(html)
    assert md.splitlines()[0] == "| Header | Header |"


def test_html_table_duplicates_rowspan_value():
    html = "<table><tr><td rowspan='2'>Method</td><td>1</td></tr><tr><td>2</td></tr></table>"
    md = html_table_to_markdown(html)
    lines = md.splitlines()
    assert lines[0] == "| Method | 1 |"
    assert lines[2] == "| Method | 2 |"


def test_html_table_tolerates_malformed_span_attribute():
    # GLM-OCR's own model-generated HTML, not hand-authored -- a garbled rowspan
    # shouldn't crash table conversion (and by extension, the whole page/PDF)
    html = "<table><tr><td rowspan='not-a-number'>x</td><td>y</td></tr></table>"
    md = html_table_to_markdown(html)
    assert md.splitlines()[0] == "| x | y |"


def test_html_table_missing_table_tag_falls_back_to_stripped_text():
    assert html_table_to_markdown("just some text") == "just some text"
