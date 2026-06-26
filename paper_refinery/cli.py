"""refinery CLI -- parse -> enrich (figures) -> chunk -> write a chunks manifest.

`refinery paper.pdf` produces `paper.chunks.json`, the hand-off papis-ask ingests via
paper-qa's ``Docs.aadd_texts``.
"""

from __future__ import annotations

import functools
import json
import tempfile
from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
from .config import RefineryConfig
from .enrich import enrich_markdown
from .figures import describe_page_figures, make_client
from .parse import parse_pdf


def write_chunks(chunks: list[Chunk], docname: str, source_pdf: str, out_path: Path) -> None:
    """Serialize chunks to the hand-off JSON that papis-ask ingests via aadd_texts."""
    payload = {
        "source_pdf": source_pdf,
        "docname": docname,
        "parser": "paper-refinery",
        "chunks": [c.to_dict() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _refine(pdf: Path, out: Path, md_out: Path, image_dir: Path, cfg: RefineryConfig) -> int:
    """Run the pipeline; returns the number of chunks written. Page renders go to
    ``image_dir`` (kept only for the duration of the run)."""
    parsed = parse_pdf(pdf, image_dir, cfg.parse)

    # one Gemini client, reused across pages (the page calls run concurrently in enrich)
    client = make_client(cfg.figure)
    describe = functools.partial(describe_page_figures, client=client)
    enriched = enrich_markdown(parsed, cfg.figure, describe=describe)

    # keep the enriched markdown as a reviewable artifact, before chunking
    md_out.write_text(enriched)

    chunks = chunk_markdown(enriched, cfg.chunk)
    write_chunks(chunks, pdf.stem, str(pdf), out)
    return len(chunks)


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
    help="Where to keep page renders (default: a temp dir cleaned up after the run).",
)
def main(pdf: Path, out: Path | None, md_out: Path | None, image_dir: Path | None) -> None:
    """Parse, figure-enrich, and chunk PDF into a chunks manifest for papis-ask."""
    cfg = RefineryConfig()
    out = out or pdf.with_suffix(".chunks.json")
    md_out = md_out or pdf.with_suffix(".refinery.md")

    if image_dir is not None:
        n = _refine(pdf, out, md_out, image_dir, cfg)
    else:
        with tempfile.TemporaryDirectory(prefix="refinery-") as td:
            n = _refine(pdf, out, md_out, Path(td), cfg)

    click.echo(f"enriched markdown -> {md_out}\nwrote {n} chunks -> {out}")


if __name__ == "__main__":
    main()
