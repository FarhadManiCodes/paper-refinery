"""refinery CLI -- parse -> enrich (figures) -> chunk -> write a chunks manifest.

`refinery paper.pdf` produces `paper.chunks.json`, the hand-off papis-ask ingests via
paper-qa's ``Docs.aadd_texts``.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
from .config import RefineryConfig, load_config
from .enrich import enrich_markdown
from .figures import describe_page_figures, make_client
from .parse import parse_pdf

_IMAGE_LINK_RE = re.compile(r"(!\[[^\]]*\]\()([^)\s]+)(\))")


def _relativize_image_links(md: str, base: Path) -> str:
    """Rewrite absolute image-link paths in markdown to be relative to ``base``.

    The .md file and its figure crops are meant to travel together (as siblings under
    ``image_dir``) -- an absolute path breaks the moment either is moved, renamed, or
    shared with someone else.
    """

    def _rel(m: re.Match) -> str:
        path = Path(m.group(2))
        if not path.is_absolute():
            return m.group(0)
        return f"{m.group(1)}{os.path.relpath(path, start=base)}{m.group(3)}"

    return _IMAGE_LINK_RE.sub(_rel, md)


def write_chunks(chunks: list[Chunk], docname: str, source_pdf: str, out_path: Path) -> None:
    """Serialize chunks to the hand-off JSON that papis-ask ingests via aadd_texts."""
    payload = {
        "source_pdf": source_pdf,
        "docname": docname,
        "parser": "paper-refinery",
        "chunks": [c.to_dict() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _refine(
    pdf: Path, out: Path, md_out: Path, image_dir: Path, cfg: RefineryConfig
) -> tuple[int, Path | None]:
    """Run the pipeline; returns (chunk count, raw references markdown path or None).
    Figure crops go to ``image_dir``, which persists alongside ``md_out`` (not cleaned
    up) so the enriched markdown's image links keep resolving after the run."""
    parsed = parse_pdf(pdf, image_dir, cfg.parse)

    refs_path: Path | None = None
    if parsed.references_markdown:
        refs_path = pdf.with_suffix(".references.md")
        refs_path.write_text(parsed.references_markdown)

    # one Gemini client, reused across pages (the page calls run concurrently in enrich).
    # Only built when there's actually a figure to describe -- a figure-less paper must
    # not require GOOGLE_API_KEY to be set.
    describe = None
    if parsed.figure_crops:
        client = make_client(cfg.figure)
        describe = functools.partial(describe_page_figures, client=client)
    enriched = enrich_markdown(parsed, cfg.figure, describe=describe)
    enriched = _relativize_image_links(enriched, md_out.parent)

    # keep the enriched markdown as a reviewable artifact, before chunking
    md_out.write_text(enriched)

    chunks = chunk_markdown(enriched, cfg.chunk)
    write_chunks(chunks, pdf.stem, str(pdf), out)
    return len(chunks), refs_path


@click.command()
@click.argument("pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output chunks JSON (default: <pdf>.chunks.json next to the PDF).",
)
@click.option(
    "--md-out",
    type=click.Path(path_type=Path),
    default=None,
    help="Enriched markdown artifact (default: <pdf>.refinery.md next to the PDF).",
)
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to keep figure/chart crops (default: alongside --md-out).",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GLM-OCR GGUF weights (default: 'parse.model_path' in config.toml).",
)
@click.option(
    "--mmproj-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="GLM-OCR GGUF vision projector (default: 'parse.mmproj_path' in config.toml).",
)
def main(
    pdf: Path,
    out: Path | None,
    md_out: Path | None,
    image_dir: Path | None,
    model_path: Path | None,
    mmproj_path: Path | None,
) -> None:
    """Parse, figure-enrich, and chunk PDF into a chunks manifest for papis-ask."""
    cfg = load_config()
    if model_path is not None:
        cfg.parse.model_path = str(model_path)
    if mmproj_path is not None:
        cfg.parse.mmproj_path = str(mmproj_path)
    out = out or pdf.with_suffix(".chunks.json")
    md_out = md_out or pdf.with_suffix(".refinery.md")
    image_dir = image_dir or md_out.parent

    n, refs_path = _refine(pdf, out, md_out, image_dir, cfg)

    summary = f"enriched markdown -> {md_out}\nwrote {n} chunks -> {out}"
    if refs_path is not None:
        summary += f"\nreferences -> {refs_path}"
    click.echo(summary)


if __name__ == "__main__":
    main()
