# paper-refinery

Parse, figure-enrich, citation-verify, and chunk scientific papers into RAG-ready chunks for
**papis-ask**.

It is a standalone preprocessor: it turns a PDF into a set of clean, section-aware,
overlapping text chunks — with figure descriptions spliced in, page numbers, a
verified/enriched bibliography, and in-text citations standardized to `[surname_year]`
citekeys — that papis-ask ingests directly, bypassing pypdf's blind char-window chunking. The
chunks are plain data, so it works as a front-end for any RAG pipeline, not just papis-ask.

## Requirements

- **Python ≥ 3.12** (3.14 used in development).
- **A Zhipu / z.ai API key** — OCR runs on the cloud GLM-OCR API by default (`mode="maas"`):
  no GPU, no model download, no `llama-server`, no local torch. Get a key at
  [z.ai](https://docs.z.ai/guides/vlm/glm-ocr) (~$0.03/M tokens ≈ pennies per paper); see
  [API keys](#api-keys).
- **`GOOGLE_API_KEY`** — Gemini, for figure descriptions + citation extraction; see
  [API keys](#api-keys). The citation-resolution providers (CrossRef / Semantic Scholar /
  OpenAlex) are keyless.
- **Optional — a local OCR backend** instead of the cloud: `llama-server` + GLM-OCR GGUF
  weights + a GPU + the `.[local]` install extra. See
  [Local (selfhosted) OCR backend (optional)](#local-selfhosted-ocr-backend-optional).

## Pipeline

A PDF flows through four stages. **Figure-enrichment and citation-resolution run
concurrently** — both are network-bound and independent — then the citekey rewrite is applied
once both finish:

```
                                  ┌─ figures / enrich  (Gemini) ─┐
PDF ── parse (GLM-OCR: cloud/local) ┤                            ├── chunk ──▶ outputs
                                  └─ citations (Gemini + web) ───┘
```

| Stage | Tool | What it does |
| --- | --- | --- |
| **parse** | GLM-OCR — Zhipu cloud API by default, or local `llama-server` + `glmocr` (PP-DocLayout-V3 layout + per-region OCR) | PDF → clean body markdown (LaTeX + markdown tables), figure crops, and page markers. Boilerplate (headers/footers/page numbers) is dropped; the bibliography is routed out to its own raw markdown. |
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

## OCR backend: cloud (default) or local

`parse.mode`, set in `~/.config/paper-refinery/config.toml`, selects the OCR backend:

```toml
[parse]
mode = "maas"          # Zhipu cloud GLM-OCR (default) -- needs ZHIPU_API_KEY, no GPU
# mode = "selfhosted"  # local llama-server + GLM-OCR GGUF -- needs a GPU + the .[local] extra
```

- **`maas` (default):** layout + OCR on Zhipu's cloud (`api.z.ai`). ~$0.03/M tokens
  (≈ 4k tokens/page ≈ $0.06 per 500 pages); no GPU, no model download, no local torch. Same
  GLM-OCR model as local.
- **`selfhosted`:** everything runs on your machine (no per-paper cost, no network for OCR),
  but needs a GPU, the GGUF weights, and `pip install -e .[local]` — see
  [Local (selfhosted) OCR backend (optional)](#local-selfhosted-ocr-backend-optional).

## Why

pypdf mangles equations and emits glyph garbage for figures; paper-qa then
char-chunks the result blindly. paper-refinery produces faithful, semantically
chunked text instead. See [`CLAUDE.md`](CLAUDE.md) for the architecture and the validated
chunking policy.

Parsing uses GLM-OCR — a small (0.9B) but top-ranked document OCR model — on Zhipu's cloud API
by default, or fully locally via `llama.cpp` + the `glmocr` SDK if you prefer no per-paper cost
and no network for OCR. Either way it replaces the cloud LlamaParse step this project began
with. Figure descriptions call Gemini; citation resolution calls CrossRef / Semantic Scholar /
OpenAlex (keyless, disk-cached).

## Development setup

Tooling lives in `.venv/` (no `uv.lock` -- a plain venv managed with `uv pip`, not
`uv sync`). The default install is **cloud-only and torch-free** -- one step:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Run the suite with `.venv/bin/pytest` (fully offline -- cloud OCR and Gemini are mocked, so
no key, GPU, or network is needed to test). Lint/format with `.venv/bin/ruff check .` /
`ruff format .`. The optional local OCR backend is a separate, heavier install
([below](#local-selfhosted-ocr-backend-optional)).

## Local (selfhosted) OCR backend (optional)

Only needed for `mode = "selfhosted"`; the default cloud (`maas`) mode needs none of this.
It requires a locally running `llama-server` (from `llama.cpp`) serving the GLM-OCR model,
plus the `glmocr` SDK's layout detection (`.[local]` extra -> `torch`/`torchvision` +
PP-DocLayout-V3). On a machine without an NVIDIA GPU, PyPI's default torch wheel still bundles
the full CUDA toolkit (~3.4GB of unused `nvidia-*`/`triton`), so install the CPU-only build
first:

```bash
uv pip install --python .venv/bin/python --no-config \
  --default-index https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/bin/python -e ".[dev,local]"
```

Then set up the server and weights:

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

4. **The `.[local]` extra** (`glmocr[selfhosted]`) pulls PP-DocLayout-V3 (torch/transformers,
   not PaddlePaddle despite the HF org name) automatically from Hugging Face Hub on first use
   -- a separate, automatic download the first time you parse a PDF (needs `HF_TOKEN` +
   network egress once), after which every selfhosted run is fully local.

## API keys

Credentials load from **one file per service** under `~/.config/paper-refinery/secrets/` (its
own directory, deliberately isolated from other tools' env so nothing can silently redirect a
client elsewhere), each `KEY=value` (python-dotenv), loaded automatically by `load_config()`:

```
~/.config/paper-refinery/secrets/
  zai.env         ZHIPU_API_KEY=...      # Zhipu / z.ai cloud GLM-OCR (maas mode -- the default)
  google.env      GOOGLE_API_KEY=...     # Gemini: figure descriptions + citation extraction
  hf.env          HF_TOKEN=...           # ONLY for selfhosted mode: PP-DocLayout-V3 download
```

`ZHIPU_API_KEY` is the SDK's env-var name (z.ai and Zhipu/BigModel are the same provider, same
key). The citation-resolution providers (CrossRef / Semantic Scholar / OpenAlex) are keyless.
In selfhosted mode `ZHIPU_API_KEY` isn't needed; in the default maas mode `HF_TOKEN` isn't.

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

# many papers — OCR runs serially on one shared OCR backend while each paper's
# network stages (figures, citations) overlap the next paper's OCR. Yields each
# RefineResult in COMPLETION order (not input order) as soon as it is ready, so a
# caller can index each paper the moment it finishes rather than blocking on the batch.
for result in refine_many(pdfs, dois=dois):   # dois aligned to pdfs, entries may be None
    index(result)
```

`refine(pdf, cfg=None, *, out=None, citations_out=None, work_dir=None, force_parse=False, doi=None) -> RefineResult`
reuses the parse checkpoint (pass `force_parse=True` to re-OCR). `cfg` defaults to
`load_config()`.

`refine_many(pdfs, cfg=None, *, dois=None, force_parse=False, workers=4, ocr_workers=2) -> Iterator[RefineResult]`.
In the default cloud (`maas`) mode each paper runs its whole pipeline concurrently on a pool
of `workers`, and results stream out in completion order. **OCR is gated separately** by
`ocr_workers` (default 2): the cloud OCR endpoint rate-limits concurrent requests (z.ai
returns 429 above ~2–3 at once), so only `ocr_workers` papers may be OCR-ing at any moment
while up to `workers` run the other network stages (Gemini figures, citation providers, which
hit different hosts). The moment a paper's OCR finishes it frees its slot and flows straight
into its figure/citation stages while the next paper OCRs. Keep `ocr_workers` at/below your
z.ai tier's concurrency (`z.ai/manage-apikey/rate-limits`); raise `workers` to overlap more
network tails. In `selfhosted` mode OCR is bound to one local llama-server, so it falls back
to a serial-OCR path (`ocr_workers` unused) that overlaps each paper's network stages with the
next paper's OCR. A paper that fails — corrupt PDF, provider outage, or an OCR call the cloud
never fulfilled (an empty parse is caught and treated as a failure, never written as a
0-chunk result) — is logged and skipped, not fatal to the batch.

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
