"""Bibliography extraction: one batched Gemini call per paper's bibliography.

Turns each raw, OCR'd reference string into rough structured fields -- title, authors,
year, venue, volume, page, a DOI if visible, and the original citation marker. This is a
labeling guess, not ground truth: citation styles vary too much (numbered, author-year,
lettered) for a hand-rolled regex to reliably find title/venue boundaries, but an LLM's
guess can still be wrong. Verifying/resolving this against real bibliographic data is a
separate concern, deliberately not this file's job.
"""

from __future__ import annotations

import os
import warnings

from pydantic import BaseModel, Field

from .config import CitationConfig
from .retry import call_with_backoff


class Author(BaseModel):
    family: str = Field(description="Author's family/last name, exactly as printed.")
    given: str | None = Field(
        default=None, description="Author's given name(s) or initials, exactly as printed."
    )


class ExtractedReference(BaseModel):
    """One bibliography entry's rough fields. Field descriptions double as per-field
    prompting -- Google's own guidance for structured extraction: prefer schema
    descriptions over duplicating the schema in prose."""

    citation_key: str | None = Field(
        default=None,
        description='The exact citation marker printed for this entry, if any '
        '(e.g. "[1]", "[14]", "Smith, 2023"). Null if this bibliography style has no '
        "separate marker.",
    )
    title: str = Field(
        description="The work's title, exactly as printed, with residual Markdown "
        "artifacts and trailing punctuation stripped."
    )
    authors: list[Author] = Field(
        default_factory=list, description="Authors in the order printed."
    )
    year: int | None = Field(default=None, description="Publication year, if printed.")
    container_title: str | None = Field(
        default=None, description="Journal, conference, or book title."
    )
    volume: str | None = Field(
        default=None, description="Volume identifier, exactly as printed (may be alphanumeric)."
    )
    page: str | None = Field(
        default=None, description="Page range or article number, exactly as printed."
    )
    doi: str | None = Field(
        default=None, description="DOI, only if visibly printed in this exact line."
    )


_PROMPT = (
    "You extract structured bibliographic metadata from unstructured Markdown reference "
    "lists (parsed from academic PDFs). The response shape is enforced by a JSON schema, "
    "so focus entirely on extraction quality and preventing hallucinations:\n\n"
    "1. Preserve citation keys: capture the exact citation marker (e.g. \"[1]\", \"[14]\", "
    "or \"Smith, 2023\") associated with each entry, if one is present -- this is the key "
    "for downstream inline-citation resolution.\n"
    "2. Strict faithfulness: extract titles, authors, and venues exactly as written. If a "
    "nullable field (DOI, year, volume, page) is genuinely absent from the input text, "
    "use null. NEVER infer, guess, or use your own training-data knowledge of a paper to "
    "fill in a gap that isn't actually printed in this exact line -- a well-known paper's "
    "real-world metadata can differ from what this specific citation prints (a different "
    "version, a typo, a preprint vs. published date).\n"
    "3. Normalize for API search: strip residual Markdown artifacts (**, *, stray "
    "newlines) and trailing punctuation from the title (\"Attention is all you need.\" -> "
    "\"Attention is all you need\").\n"
    "4. Author formatting: keep the author list exactly as it appears. Do not abbreviate "
    "to \"et al.\" unless that is literally printed in the text.\n"
    "5. One entry per line: each numbered line below is meant to be one bibliography "
    "entry, but OCR noise occasionally blurs two entries together. If a line visibly "
    "contains more than one distinct entry (e.g. two citation markers), extract only the "
    "first one faithfully -- never merge fields from two different entries into one "
    "corrupted record.\n\n"
    "Return one JSON object per input line, in the same order, matching the schema "
    "exactly."
)


def make_client(cfg: CitationConfig):
    """Create a Gemini client. Build one and reuse it across the whole bibliography.

    Assumes API keys are already loaded into the environment -- see
    ``config.load_config``'s ``_load_secrets`` call.
    """
    from google import genai

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not set")
    return genai.Client(api_key=api_key)


def extract_references(
    raw_texts: list[str], cfg: CitationConfig, client=None
) -> list[dict]:
    """Extract rough structured fields from each raw reference string, in one batched,
    schema-enforced Gemini call.

    Order-preserving: ``output[i]`` corresponds to ``raw_texts[i]`` -- callers re-attach
    this to parse.py's own page/number metadata by position. An entry the model couldn't
    extract anything useful from comes back as ``{}``, never dropped (dropping a
    position would silently misalign every entry after it).
    """
    from google.genai import types

    if not raw_texts:
        return []
    client = client or make_client(cfg)

    listing = "\n\n".join(f"{i + 1}. {t}" for i, t in enumerate(raw_texts))
    prompt = f"{_PROMPT}\n\n{listing}"
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=list[ExtractedReference],
        # a deterministic labeling task, not creative generation -- matches glmocr's
        # own default (temperature=0.0) elsewhere in this pipeline
        temperature=0.0,
        # low-complexity, schema-constrained extraction doesn't benefit from reasoning --
        # MINIMAL is Gemini 3's equivalent of "thinking: disabled" (thinking can't be
        # fully turned off on Gemini 3, but MINIMAL uses as few tokens as possible,
        # documented as best for exactly this kind of task)
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    response = call_with_backoff(
        lambda: client.models.generate_content(model=cfg.model, contents=prompt, config=config),
        cfg.retry_attempts,
        cfg.retry_base_delay,
    )

    items = [r.model_dump(exclude_none=True) for r in response.parsed]
    if len(items) != len(raw_texts):
        warnings.warn(
            f"extract_references: got {len(items)} items for {len(raw_texts)} input "
            "lines; padding/truncating to align by position"
        )
    return (items + [{}] * len(raw_texts))[: len(raw_texts)]
