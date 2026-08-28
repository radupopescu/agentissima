"""§2.2 memory admissibility.

The gate is deliberately thin. What can be decided from metadata is decided
here; what cannot is left to the load attempt and to §5.2 measurement. These
tests pin that division, because the failure being guarded against is a gate
that *looks* authoritative and quietly excludes runs the machine could perform.
"""

from __future__ import annotations

from harness.admissibility import (
    ADMISSIBLE,
    OVERSIZED,
    UNSUPPORTED,
    classify_declared,
    classify_load_failure,
)


# --- the free half: what the model itself declares ---------------------------


def test_a_context_above_the_advertised_maximum_is_unsupported():
    assert classify_declared(65536, 131072) == UNSUPPORTED


def test_the_advertised_maximum_itself_is_allowed():
    assert classify_declared(65536, 65536) == ADMISSIBLE


def test_a_context_within_the_maximum_is_provisionally_admissible():
    assert classify_declared(131072, 8192) == ADMISSIBLE


def test_the_declared_check_never_returns_oversized():
    """Fit is not knowable from metadata. Bonsai's MLX build loads at 65536 in
    7.5 s on a 16 GiB machine that cannot hold its full 64K cache, because MLX
    allocates lazily — so any arithmetic verdict here would be a guess."""
    verdicts = {classify_declared(131072, ctx) for ctx in (1, 8192, 131072)}
    assert OVERSIZED not in verdicts


# --- the other half: what the load says --------------------------------------


def test_a_memory_refusal_is_oversized():
    assert classify_load_failure("Error: not enough memory to load model") == OVERSIZED
    assert classify_load_failure("failed to allocate KV cache") == OVERSIZED
    assert classify_load_failure("blocked by resource guardrail") == OVERSIZED


def test_an_unrelated_failure_is_not_oversized():
    """A missing artefact or a crashed backend must not be recorded as a
    machine that was too small — that would misattribute a harness fault to a
    §2.2 verdict."""
    assert classify_load_failure("No model found matching 'xyz'") != OVERSIZED
    assert classify_load_failure("backend exited with signal 6") != OVERSIZED


def test_the_match_is_case_insensitive():
    assert classify_load_failure("OUT OF MEMORY") == OVERSIZED
