"""refinery CLI — orchestrates parse -> describe figures -> enrich -> chunk -> write."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .chunker import Chunk, chunk_markdown
from .config import RefineryConfig


def write_chunks(chunks: list[Chunk], docname: str, source_pdf: str, out_path: Path) -> None:
    """Serialize chunks to the hand-off JSON that papis-ask ingests via aadd_texts."""
    payload = {
        "source_pdf": source_pdf,
        "docname": docname,
        "parser": "paper-refinery",
        "chunks": [c.to_dict() for c in chunks],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


@click.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Output chunks JSON (default: <pdf>.chunks.json next to the PDF).",
)
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to save extracted figure images (default: <pdf-dir>/figures).",
)
def main(pdf: Path, out: Path | None, image_dir: Path | None) -> None:
    """Parse, figure-enrich, and chunk PDF into a chunks manifest for papis-ask."""
    cfg = RefineryConfig()
    out = out or pdf.with_suffix(".chunks.json")
    image_dir = image_dir or pdf.parent / "figures"

    # TODO orchestration (fill in as parse/figures/enrich land):
    #   from .parse import parse_pdf
    #   from .figures import describe_figure
    #   from .enrich import enrich_markdown
    #   parsed = parse_pdf(pdf, image_dir, cfg.parse)
    #   descriptions = {f.image_path: describe_figure(f.image_path, f.caption, cfg.figure)
    #                   for f in parsed.figures}
    #   markdown = enrich_markdown(parsed, descriptions)
    #   chunks = chunk_markdown(markdown, cfg.chunk)
    #   write_chunks(chunks, pdf.stem, str(pdf), out)
    raise NotImplementedError("CLI orchestration not wired up yet")


if __name__ == "__main__":
    main()
