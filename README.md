# paper-refinery

Parse, figure-enrich, and chunk papers into RAG-ready chunks for **papis-ask**.

It is a standalone preprocessor: it turns a PDF into a set of clean, section-aware,
overlapping text chunks (with figure descriptions, page numbers, and standardized
`[surname_year]` in-text citekeys) that papis-ask ingests directly — bypassing pypdf's
blind char-window chunking.

## Pipeline

```
PDF
 └─ parse       local GLM-OCR -> clean body markdown (LaTeX + markdown tables) + figure crops + page markers
                (llama-server + glmocr SDK: PP-DocLayout-V3 layout, per-region OCR)
                boilerplate (headers/footers/page numbers) dropped; bibliography routed to its own raw markdown
 └─ figures     Gemini        -> describe each figure crop, one call per figure (what's compared, trends;
                                 never fabricated numbers)
 └─ enrich                    -> splice figure descriptions next to their captions
 └─ citations   Gemini + APIs -> extract raw references, verify/enrich against CrossRef/Semantic
                                 Scholar/OpenAlex (cached), detect + link in-text markers, rewrite
                                 them to `[surname_year]` citekeys -- runs concurrently with figures
 └─ chunker                   -> section-aware split + guaranteed soft-overlap + page numbers
 └─ paper.chunks.json         -> consumed by papis-ask via aadd_texts
 └─ paper.citations.json      -> verified/enriched bibliography + in-text linking map
 └─ paper.refinery/           -> everything reviewable: refinery.md (enriched markdown, citekeys
                                 already rewritten), references.md (raw bibliography),
                                 resolution_report.txt (per-reference verification diff), figures/
```

## Why

pypdf mangles equations and emits glyph garbage for figures; paper-qa then
char-chunks the result blindly. paper-refinery produces faithful, semantically
chunked text instead. See the design notes for the validated chunking policy.

Parsing runs entirely locally: a small (0.9B) OCR model served via `llama.cpp`, orchestrated
by the official `glmocr` SDK, replaces the cloud-based LlamaParse step -- no per-paper API
cost, no network dependency for parsing itself (figure description calls Gemini; citation
resolution calls CrossRef/Semantic Scholar/OpenAlex, all keyless and disk-cached).

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

## Usage

```bash
refinery path/to/paper.pdf            # -> path/to/paper.chunks.json, .citations.json, .refinery/
```
