# paper-refinery

Parse, figure-enrich, citation-verify, and chunk scientific papers into RAG-ready chunks for
**papis-ask**.

It is a standalone preprocessor: it turns a PDF into a set of clean, section-aware,
overlapping text chunks — with figure descriptions spliced in, page numbers, a
verified/enriched bibliography, and in-text citations standardized to `[surname_year]`
citekeys — that papis-ask ingests directly, bypassing pypdf's blind char-window chunking. The
chunks are plain data, so it works as a front-end for any RAG pipeline, not just papis-ask.

## Requirements

- **Python ≥ 3.11** (3.12 recommended and used in development).
- **A local OCR backend** — `llama-server` (from `llama.cpp`) + the GLM-OCR GGUF weights.
  Parsing is local; see [Local OCR backend setup](#local-ocr-backend-setup).
- **API keys** — `GOOGLE_API_KEY` (Gemini: figure descriptions + citation extraction) and
  `HF_TOKEN` (one-time layout-model download on first parse); see [API keys](#api-keys). The
  citation-resolution providers (CrossRef / Semantic Scholar / OpenAlex) are keyless.

## Pipeline

A PDF flows through four stages. **Figure-enrichment and citation-resolution run
concurrently** — both are network-bound and independent — then the citekey rewrite is applied
once both finish:

```
                                ┌─ figures / enrich  (Gemini) ─┐
PDF ── parse (local GLM-OCR) ───┤                              ├── chunk ──▶ outputs
                                └─ citations (Gemini + web) ───┘
```

| Stage | Tool | What it does |
| --- | --- | --- |
| **parse** | local GLM-OCR (`llama-server` + `glmocr`, PP-DocLayout-V3 layout + per-region OCR) | PDF → clean body markdown (LaTeX + markdown tables), figure crops, and page markers. Boilerplate (headers/footers/page numbers) is dropped; the bibliography is routed out to its own raw markdown. |
| **figures → enrich** | Gemini | Describe each figure crop — one call per figure (what's compared, trends; never invents numbers) — then splice each description in next to its caption. |
| **citations** | Gemini + web APIs | Extract raw references, verify/enrich them against CrossRef / Semantic Scholar / OpenAlex (disk-cached), detect in-text markers, and rewrite them to `[surname_year]` citekeys. |
| **chunk** | — | Section-aware split with guaranteed soft-overlap and page ranges. |

### Outputs

Two finals land next to the PDF, plus a work directory holding everything reviewable:

| Path | Contents |
| --- | --- |
| `<pdf>.chunks.json` | The hand-off papis-ask ingests via `aadd_texts` — chunks with page ranges. |
| `<pdf>.citations.json` | Verified/enriched bibliography + the in-text linking map. |
| `<pdf>.refinery/` | `refinery.md` (enriched markdown, citekeys already rewritten — the last human-readable form before chunking), `references.md` (raw bibliography), `resolution_report.txt` (per-reference verification diff), `figures/`, and `parse_cache/` (the OCR checkpoint). |

## Why

pypdf mangles equations and emits glyph garbage for figures; paper-qa then
char-chunks the result blindly. paper-refinery produces faithful, semantically
chunked text instead. See [`CLAUDE.md`](CLAUDE.md) for the architecture and the validated
chunking policy.

Parsing runs entirely locally: a small (0.9B) OCR model served via `llama.cpp`, orchestrated
by the official `glmocr` SDK, replaces the cloud-based LlamaParse step -- no per-paper API
cost, no network dependency for parsing itself (figure description calls Gemini; citation
resolution calls CrossRef/Semantic Scholar/OpenAlex, all keyless and disk-cached).

## Development setup

Tooling lives in `.venv/` (no `uv.lock` -- this is a plain venv managed with `uv pip`,
not `uv sync`). `glmocr[selfhosted]` depends on `torch`/`torchvision`; on a machine
without an NVIDIA GPU, PyPI's default wheel still bundles the full CUDA toolkit
(~3.4GB of unused `nvidia-*`/`triton` packages) because pip has no hardware detection.
Install the CPU-only build first, then the project, so it never gets swapped back in:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --no-config \
  --default-index https://download.pytorch.org/whl/cpu \
  torch torchvision
uv pip install --python .venv/bin/python -e ".[dev]"
```

A `[tool.uv.pip]` index pin in `pyproject.toml` does NOT reliably take effect for
`uv pip install` (confirmed live: it silently let PyPI's CUDA build win over the
pinned index) -- the two-step CLI install above is the only version that has been
verified to stick. This only matters for hosts without a CUDA GPU; skip it if you
have one and want GPU-accelerated layout detection.

## Local OCR backend setup

Parsing requires a locally running `llama-server` (from `llama.cpp`) serving the GLM-OCR
model, plus the `glmocr` SDK to drive layout detection and per-region OCR against it.

1. **Install `llama-server`** with GPU support for your hardware (CUDA, Vulkan, or ROCm --
   any llama.cpp backend works, since it's just an OpenAI-compatible HTTP server to
   `glmocr`). On Arch, `pacman -S llama.cpp-vulkan` (or the CUDA/ROCm variant) provides it;
   otherwise build from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
   Confirm your build sees a GPU with `llama-server --list-devices`.

2. **Download the GLM-OCR GGUF weights** (two files -- the model and its vision
   projector) from [ggml-org/GLM-OCR-GGUF](https://huggingface.co/ggml-org/GLM-OCR-GGUF)
   to a local cache directory, e.g.:

   ```bash
   python -c "
   from huggingface_hub import hf_hub_download
   for fn in ['GLM-OCR-f16.gguf', 'mmproj-GLM-OCR-Q8_0.gguf']:
       print(hf_hub_download('ggml-org/GLM-OCR-GGUF', fn, local_dir='~/.cache/paper-refinery/models'))
   "
   ```

   (`Q8_0` main weights are also available and smaller, ~950MB vs ~1.8GB for F16, with
   negligible quality difference for a model this size.)

3. **Point `ParseConfig` at both files** via `~/.config/paper-refinery/config.toml`
   (XDG-style user config; not secrets, so it doesn't belong in your shell rc):

   ```toml
   [parse]
   model_path = "/home/you/.cache/paper-refinery/models/GLM-OCR-f16.gguf"
   mmproj_path = "/home/you/.cache/paper-refinery/models/mmproj-GLM-OCR-Q8_0.gguf"
   ```

   The file is optional and only needs to set what differs from the code defaults --
   see `paper_refinery.config.load_config` for the full mechanism (any `ParseConfig`/
   `FigureConfig`/`ChunkConfig` field can be overridden this way, e.g. `n_gpu_layers` or
   `glmocr_config_overrides` for hardware-specific tuning). `--model-path`/`--mmproj-path`
   CLI flags override the file per-run. `parse_pdf` spawns and tears down `llama-server`
   itself for the duration of each run, so you don't need to run it as a standing service;
   batch callers can instead hold one server open across many PDFs via
   `paper_refinery.backend.ocr_backend` (server spawn + model load is the dominant fixed
   cost of a parse).
   `llama-server` currently requires `--flash-attn off -fit off` for GLM-OCR ([llama.cpp
   discussion #19721](https://github.com/ggml-org/llama.cpp/discussions/19721));
   `ParseConfig.extra_server_args` defaults to that already -- re-check on a llama.cpp
   upgrade in case the constraint has been lifted.

4. **`glmocr[selfhosted]`** (already a project dependency) pulls PP-DocLayout-V3
   (torch/transformers, not PaddlePaddle despite the HF org name) automatically from
   Hugging Face Hub on first use -- this is a separate, automatic download the first time
   you parse a PDF, not a manual step, but it does mean the first run needs network egress
   even though every run after that is fully local.

## API keys

Two services need credentials. They load from **one file per service** under
`~/.config/paper-refinery/secrets/` (its own directory — deliberately never shared with any
other tool's env, so nothing can silently redirect glmocr's OpenAI-compatible client away
from your local `llama-server`):

```
~/.config/paper-refinery/secrets/
  google.env      GOOGLE_API_KEY=...     # Gemini: figure descriptions + citation extraction
  hf.env          HF_TOKEN=...           # one-time PP-DocLayout-V3 download on first parse
```

Each file is `KEY=value` (python-dotenv format), loaded automatically by `load_config()`. The
citation-resolution providers (CrossRef / Semantic Scholar / OpenAlex) are keyless, so no
credentials are needed for the citation-verification stage.

## Usage

### CLI

```bash
refinery path/to/paper.pdf            # -> path/to/paper.chunks.json, .citations.json, .refinery/
```

Options (all optional; outputs default next to the PDF):

| Flag | Effect |
| --- | --- |
| `--doi DOI` | Source paper DOI — enables the citation fast-path (one bulk reference fetch instead of a per-reference provider search). Without it the OCR'd title is tried. |
| `--force-parse` | Re-run OCR, bypassing the parse checkpoint in `<pdf>.refinery/parse_cache/`. |
| `--from chunk` | Re-chunk the saved `refinery.md` only (instant); for tuning chunk policy without re-running the expensive upstream stages. |
| `--out` / `--citations-out` / `--work-dir` | Override the individual output locations. |
| `--model-path` / `--mmproj-path` | Override the GLM-OCR GGUF paths from `config.toml`. |

### Programmatic API (what papis-ask integrates against)

This is the stable in-process surface — no paper-qa objects cross the boundary; the caller
owns turning chunks into whatever its indexer wants.

```python
from pathlib import Path
from paper_refinery import refine, refine_many, RefineResult

# one paper
result: RefineResult = refine(Path("paper.pdf"), doi="10.1234/abc")

# many papers — OCR runs serially on one shared GPU backend while each paper's
# network stages (figures, citations) overlap the next paper's OCR. Yields each
# RefineResult in COMPLETION order (not input order) as soon as it is ready, so a
# caller can index each paper the moment it finishes rather than blocking on the batch.
for result in refine_many(pdfs, dois=dois):   # dois aligned to pdfs, entries may be None
    index(result)
```

`refine(pdf, cfg=None, *, out=None, citations_out=None, work_dir=None, force_parse=False, doi=None) -> RefineResult`
reuses the parse checkpoint (pass `force_parse=True` to re-OCR). `cfg` defaults to
`load_config()`.

`refine_many(pdfs, cfg=None, *, dois=None, force_parse=False, network_workers=3) -> Iterator[RefineResult]`.
A paper whose OCR fails (corrupt PDF, or a wedged server the watchdog killed) is logged and
skipped, not fatal to the batch. `INFO` logs on the `paper_refinery` logger show the
OCR→pool handoff and each paper streaming out.

**`RefineResult`** — refinery's own types/paths only:

| field | meaning |
| --- | --- |
| `chunks: list[Chunk]` | the chunks in memory (also written to `chunks_path`) |
| `chunks_path: Path` | `<pdf>.chunks.json` — the papis-ask hand-off |
| `citations_path: Path` | `<pdf>.citations.json` (written only when the paper had references) |
| `work_dir: Path` | `<pdf>.refinery/` — refinery.md, references.md, figures/, parse_cache/ |

### Output shapes

`<pdf>.chunks.json`:

```jsonc
{
  "source_pdf": "…/paper.pdf",
  "docname": "paper",
  "parser": "paper-refinery",
  "chunks": [
    { "index": 0, "text": "…", "page_start": 1, "page_end": 2,
      "overlap_chars": 0, "overlap_mode": "-" }
  ]
}
```

`<pdf>.citations.json`:

```jsonc
{
  "source_pdf": "…/paper.pdf",
  "docname": "paper",
  "references": [ { "title": "…", "year": 2020, "authors": [...], "doi": "…",
                    "verified": true, "match": "crossref", "...": "..." } ],
  "linking": {
    "style": "numbered-bracket|numbered-paren|author-year",
    "markers": [ { "text": "[1]", "refs": [0] } ],
    "uncited": [ ], "ambiguous": [ ]
  }
}
```

## License

[GPL-3.0-or-later](LICENSE) — copyleft: works that build on paper-refinery must also be
released under a GPL-compatible open-source license. (The GLM-OCR model weights and
PP-DocLayout-V3 you download at runtime carry their own upstream licenses — check those
before redistributing the models themselves.)
