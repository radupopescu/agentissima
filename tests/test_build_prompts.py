"""The Stage 1 raw-inference corpus generator (doc/benchmark.md §9 Stage 1).

Unlike the workspace/testrepo fixtures, nothing here is graded, so there is
no oracle to solve it — these are sanity checks on the generator itself.
"""

from __future__ import annotations

from fixtures.build_prompts import CHARS_PER_TOKEN, TARGET_TOKENS, build_document


def test_generation_is_deterministic():
    assert build_document("8k", "primary") == build_document("8k", "primary")


def test_primary_and_alternate_differ():
    assert build_document("8k", "primary") != build_document("8k", "alternate")


def test_the_two_tiers_differ():
    assert build_document("8k", "primary") != build_document("16k", "primary")


def test_sizes_are_within_ten_percent_of_target():
    for tier, target_tokens in TARGET_TOKENS.items():
        target_chars = target_tokens * CHARS_PER_TOKEN
        for variant in ("primary", "alternate"):
            length = len(build_document(tier, variant))
            assert abs(length - target_chars) / target_chars < 0.10, (tier, variant, length)


def test_each_document_ends_with_an_instruction():
    for tier in TARGET_TOKENS:
        for variant in ("primary", "alternate"):
            text = build_document(tier, variant)
            assert text.rstrip().endswith((".", ":")), (tier, variant)
            assert "write" in text.lower().rsplit("---", 1)[-1].lower()


def test_documents_are_valid_utf8_text():
    for tier in TARGET_TOKENS:
        for variant in ("primary", "alternate"):
            text = build_document(tier, variant)
            text.encode("utf-8").decode("utf-8")
            assert "\x00" not in text
