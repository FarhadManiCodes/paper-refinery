from paper_refinery.fences import fence_code, repair_doubled_fences


def _fence_count(text: str) -> int:
    return sum(1 for line in text.split("\n") if line.lstrip().startswith(("```", "~~~")))


def test_fence_code_plain_content_is_wrapped_once():
    assert fence_code("a\nb") == "```\na\nb\n```"


def test_fence_code_unwraps_existing_fence_keeping_language():
    assert fence_code("```python\nx = 1\n```") == "```python\nx = 1\n```"


def test_fence_code_closes_an_unclosed_fence():
    assert fence_code("```python\nx = 1") == "```python\nx = 1\n```"


def test_fence_code_embedded_fence_gets_longer_outer_fence():
    out = fence_code("Listing 1:\n```python\nx = 1\n```")
    assert out == "````\nListing 1:\n```python\nx = 1\n```\n````"


def test_repair_collapses_doubled_language_block():
    doubled = "intro\n\n```\n```python\nx = 1\n```\n```\n\nafter $a$"
    assert repair_doubled_fences(doubled) == "intro\n\n```python\nx = 1\n```\n\nafter $a$"


def test_repair_collapses_doubled_bare_block():
    doubled = "```\n```\nx = 1\n```\n```\nafter"
    assert repair_doubled_fences(doubled) == "```\nx = 1\n```\nafter"


def test_repair_wraps_region_with_text_before_embedded_fence():
    doubled = "```\nCodeSnip 6.e\n\n```python\nx = 1\n```\n```\n\n## Next\n\nprose"
    out = repair_doubled_fences(doubled)
    assert out == "````\nCodeSnip 6.e\n\n```python\nx = 1\n```\n````\n\n## Next\n\nprose"


def test_repair_leaves_plain_algorithm_block_alone():
    text = "before\n\n```\nfor i in range(n):\n    do(i)\n```\n\nafter"
    assert repair_doubled_fences(text) == text


def test_repair_is_idempotent_and_leaves_clean_blocks_alone():
    doubled = (
        "a\n\n```\n```python\nx = 1\n```\n```\n\nb\n\n```\nplain\n```\n\n"
        "```\nCap\n```matlab\ny\n```\n```\n\nc"
    )
    once = repair_doubled_fences(doubled)
    assert repair_doubled_fences(once) == once
    assert "```\n```" not in once.replace("````", "")


def test_repair_keeps_fence_parity_even_across_many_blocks():
    doubled = "\n\n".join(f"p{i} $x$\n\n```\n```python\nc{i}\n```\n```" for i in range(5))
    assert _fence_count(repair_doubled_fences(doubled)) % 2 == 0


def test_repair_unterminated_block_is_left_untouched():
    text = "```\n```python\nx = 1\n"
    assert repair_doubled_fences(text) == text


def test_repair_handles_a_tilde_inner_fence():
    # the inner tilde block must end at its own ~~~ closer, not at the outer wrapper's fence
    doubled = (
        "```\n~~~python\nx = 1\n~~~\n```\n\nprose $a$\n\n```\n```python\ny = 2\n```\n```\n\ntail"
    )
    assert repair_doubled_fences(doubled) == (
        "```python\nx = 1\n```\n\nprose $a$\n\n```python\ny = 2\n```\n\ntail"
    )


def test_repair_handles_a_longer_backtick_inner_fence():
    doubled = "```\n````\nx = 1\n````\n```\n\nprose $a$\n\n```\n```python\ny = 2\n```\n```\n\ntail"
    assert repair_doubled_fences(doubled) == (
        "```\nx = 1\n```\n\nprose $a$\n\n```python\ny = 2\n```\n\ntail"
    )


def test_repair_handles_text_before_a_longer_inner_fence():
    doubled = "```\nCap\n````python\nx = 1\n````\n```\n\nprose $a$\n\n```\nplain\n```\n\ntail"
    assert repair_doubled_fences(doubled) == (
        "`````\nCap\n````python\nx = 1\n````\n`````\n\nprose $a$\n\n```\nplain\n```\n\ntail"
    )


def test_repair_keeps_a_shorter_fence_inside_a_longer_inner_fence_literal():
    # the ``` lines are text inside the ```` block, not its closer
    doubled = "```\n````\n```\nnot a closer\n```\n````\n```\n\nafter $a$"
    assert repair_doubled_fences(doubled) == ("````\n```\nnot a closer\n```\n````\n\nafter $a$")


def test_repair_is_idempotent_for_tilde_and_long_fences():
    doubled = (
        "```\n~~~\nx\n~~~\n```\n\np\n\n"
        "```\n````\ny\n````\n```\n\nq\n\n"
        "```\n```python\nz\n```\n```\n"
    )
    once = repair_doubled_fences(doubled)
    assert repair_doubled_fences(once) == once
    assert _fence_count(once) % 2 == 0
