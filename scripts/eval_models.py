"""Compare Gemini models for refinery's stages on the real papis library before switching.

Three subcommands, each read-only for the library (nothing is written next to a PDF):

  extraction  A/B citation extraction: the same reference batches through each model with
              retries off; counts lines left unplaced and fields not present in the
              printed reference (year, DOI, volume, page, author, title).
  figures     Describe the fixed figures in scripts/eval_figures.json with each model, the
              figure cache bypassed (its key ignores the model), and write a Markdown report
              with the image and both descriptions for judging by eye.
  check       List the Gemini models the API offers next to the ones pinned in refinery's
              config and the papis-ask config, flagging newer versions of a pinned family.

Every subcommand except ``check`` makes paid Gemini calls (a few cents in total). Run from
the repo root with the project venv, e.g.:

  .venv/bin/python scripts/eval_models.py extraction gemini-3.5-flash-lite gemini-3.6-flash-lite
  .venv/bin/python scripts/eval_models.py figures gemini-3.8-flash gemini-3.9-flash
  .venv/bin/python scripts/eval_models.py check

First used on 2026-09-24 to move extraction to gemini-3.5-flash-lite and figures to
gemini-3.8-flash; the numbers behind those choices are in the config.toml comments.
"""

from __future__ import annotations

import argparse
import configparser
import copy
import glob
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from paper_refinery.text_utils import fold_name

LIBRARY = Path("~/.local/share/papis/papers").expanduser()
FIGURE_PICKS = Path(__file__).with_name("eval_figures.json")
EXTRACTION_DOCS = (
    "hastie-2009",  # author-year
    "nocedal-2006",  # numbered, ditto marks
    "dong-2024",  # long IEEE entries
    "brunton-2022",  # "Title, by Author" suggested reading
    "kirk-2004",  # scanned book
    "nosrati-2026",
    "zou-2023",
    "gottschling-2021",
)


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def invented_fields(item: dict, raw: str) -> list[str]:
    """Extracted fields whose value is not printed in the raw reference -- the
    hallucination signal. Lenient on formatting (case, accents, punctuation), strict on
    substance: a year, DOI, volume/page digits, surname or most of a title must appear."""
    printed = fold_name(raw)
    bad = []
    if item.get("year") and str(item["year"]) not in raw:
        bad.append("year")
    doi = fold_name(str(item.get("doi") or "")).replace("https://doi.org/", "")
    if doi and doi not in printed:
        bad.append("doi")
    for field in ("volume", "page"):
        digits = _digits(str(item.get(field) or ""))[:4]
        if digits and digits not in _digits(raw):
            bad.append(field)
    families = [fold_name(a.get("family") or "") for a in item.get("authors") or []]
    if any(len(f) > 2 and f.split()[-1] not in printed for f in families):
        bad.append("author")
    words = re.findall(r"[a-z]{4,}", fold_name(item.get("title") or ""))[:6]
    if words and sum(w in printed for w in words) < 0.6 * len(words):
        bad.append("title")
    return bad


def _version(name: str) -> tuple[float, str] | None:
    """``gemini-3.8-flash`` -> (3.8, "flash"); the family is what follows the version."""
    m = re.match(r"(?:gemini/)?gemini-(\d+(?:\.\d+)?)-([a-z-]+?)(?:-preview|-latest|-\d+)*$", name)
    return (float(m.group(1)), m.group(2)) if m else None


def newer_in_family(pinned: str, available: list[str]) -> list[str]:
    """Available models of the same family (flash, flash-lite, ...) with a higher version."""
    mine = _version(pinned)
    if mine is None:
        return []
    return sorted(
        name for name in available if (v := _version(name)) and v[1] == mine[1] and v[0] > mine[0]
    )


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def run_extraction(models: list[str], library: Path, seed: int, size: int) -> None:
    from paper_refinery.citation_extraction import _extract_batch, make_client
    from paper_refinery.config import load_config

    rng = random.Random(seed)
    batches = []
    for doc in EXTRACTION_DOCS:
        found = glob.glob(str(library / doc / "*.citations.json"))
        if not found:
            print(f"skip {doc}: no citations.json")
            continue
        refs = json.loads(Path(found[0]).read_text())["references"]
        raws = [r["raw_text"] for r in refs if r.get("raw_text")]
        start = rng.randrange(0, max(1, len(raws) - size))
        batches.append((doc, raws[start : start + size]))
    base = load_config().citation
    for model in models:
        cfg = copy.deepcopy(base)
        cfg.model = model
        client = make_client(cfg)
        total = unplaced = 0
        invented: Counter = Counter()
        per_doc = []
        for doc, raws in batches:
            out = _extract_batch(raws, cfg, client, retry_missing=False)
            empty = sum(not item for item in out)
            total, unplaced = total + len(raws), unplaced + empty
            invented.update(
                f
                for item, raw in zip(out, raws, strict=True)
                if item
                for f in invented_fields(item, raw)
            )
            per_doc.append(f"{doc}:{empty}")
        filled = max(1, total - unplaced)
        fields = (
            ", ".join(f"{k} {100 * v / filled:.1f}" for k, v in sorted(invented.items())) or "none"
        )
        print(f"{model:26} refs={total} unplaced={unplaced} ({unplaced / total:.1%})")
        print(f"{'':26} invented per 100 refs: {fields} | unplaced by doc: {' '.join(per_doc)}")


def run_figures(models: list[str], library: Path, out: Path) -> None:
    from paper_refinery.config import load_config
    from paper_refinery.figures import describe_figure, make_client

    picks = json.loads(FIGURE_PICKS.read_text())
    chosen = []
    for pick in picks:
        work = next(iter(glob.glob(str(library / pick["doc"] / "*.refinery"))), None)
        if work is None:
            print(f"skip {pick['doc']}: no work directory")
            continue
        md = Path(work, "refinery.md").read_text()
        num = re.escape(pick["figure"])
        link = re.search(rf"!\[FIGURE {num}\]\((figures/[^)]+)\)", md)
        if link is None:
            print(f"skip {pick['doc']} figure {pick['figure']}: not in refinery.md")
            continue
        cap = re.search(rf"(?m)^\**\s*(?:FIGURE|Figure|FIG|Fig)\.?\s*{num}\b[.:]?\s*(.*)$", md)
        chosen.append(
            {**pick, "path": str(Path(work, link.group(1))), "caption": cap.group(1) if cap else ""}
        )
    base = load_config().figure
    for model in models:
        cfg = copy.deepcopy(base)
        cfg.model = model
        cfg.figure_cache_dir = ""  # the cache key ignores the model; bypass it
        client = make_client(cfg)
        for c in chosen:
            try:
                c[model] = describe_figure(
                    [Path(c["path"])], c["figure"], c["caption"], {}, cfg, client
                )
            except Exception as exc:  # report, keep going
                c[model] = {"error": repr(exc)[:200]}
    lines = [
        f"# Figure descriptions: {' vs '.join(models)}",
        "",
        "Judge each claim against the image: invented labels, values or trends count as "
        "hallucinations.",
        "",
    ]
    for c in chosen:
        lines += [
            f"## {c['doc']} figure {c['figure']} ({c['kind']})",
            "",
            f"![]({c['path']})",
            "",
            f"Caption: {c['caption'][:300]}",
            "",
        ]
        for model in models:
            r = c.get(model) or {}
            label = r.get("figure_type") or r.get("error") or "non-figure"
            lines += [f"**{model}** ({label}): {r.get('description', '')}", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({len(chosen)} figures x {len(models)} models)")


def _papis_models() -> dict[str, str]:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(os.path.expanduser("~/.config/papis/config"))
    ask = parser["ask"] if parser.has_section("ask") else {}
    return {f"papis-ask {k}": ask[k] for k in ("llm", "summary-llm", "embedding") if k in ask}


def run_check() -> None:
    from paper_refinery.config import load_config
    from paper_refinery.figures import make_client

    cfg = load_config()
    pinned = {
        "refinery figure": cfg.figure.model,
        "refinery citation": cfg.citation.model,
        **_papis_models(),
    }
    client = make_client(cfg.figure)  # keep a reference: the pager reads lazily
    available = sorted(m.name.removeprefix("models/") for m in client.models.list())
    gemini = [m for m in available if m.startswith("gemini")]
    print(f"{len(gemini)} Gemini models listed by the API\n")
    for role, model in pinned.items():
        bare = model.removeprefix("gemini/")
        listed = "listed" if bare in available else "NOT LISTED (retired or renamed?)"
        newer = newer_in_family(bare, gemini)
        print(f"{role:22} {bare:28} {listed}{'  newer: ' + ', '.join(newer) if newer else ''}")
    print(
        "\nDeprecation dates are not in the API: https://ai.google.dev/gemini-api/docs/deprecations"
    )
    print("Never switch the embedding model casually: it forces re-embedding the whole library.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extraction", help="A/B citation extraction")
    ex.add_argument("models", nargs="+")
    ex.add_argument("--seed", type=int, default=11)
    ex.add_argument("--batch", type=int, default=50)
    fig = sub.add_parser("figures", help="A/B figure descriptions, judged by eye")
    fig.add_argument("models", nargs="+")
    fig.add_argument("--out", type=Path, default=Path("figure-eval.md"))
    sub.add_parser("check", help="pinned models vs what the API lists")
    for p in (ex, fig):
        p.add_argument("--library", type=Path, default=LIBRARY)
    args = parser.parse_args()
    if args.command == "extraction":
        run_extraction(args.models, args.library, args.seed, args.batch)
    elif args.command == "figures":
        run_figures(args.models, args.library, args.out)
    else:
        run_check()


if __name__ == "__main__":
    main()
