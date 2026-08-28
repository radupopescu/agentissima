"""Memory admissibility, per doc/benchmark.md §2.2.

A `(configuration, context)` pair is admissible when the machine can actually
run it. That is decided as cheaply as the backend allows, and — critically — the
two backends allow different things:

- **llama.cpp allocates KV eagerly**, sized to the declared context, at load.
  A load that cannot fit fails, so attempting the load *is* the gate, and it is
  exact.
- **MLX allocates KV lazily**, on first touch, sized to the actual sequence. The
  load always succeeds regardless of declared context, so **no pre-flight gate
  exists**. Ternary-Bonsai-8B loads at 65536 in 7.5 s reporting 2.16 GiB on a
  machine that could not hold its full 64K cache.

An earlier revision predicted the KV size arithmetically and gated on that. It
was expensive (a multi-minute probe per configuration) and wrong in the damaging
direction: it excluded BON-M2 at 32K and 64K, while its llama.cpp twin — same
model, same geometry — remained admissible, which would have deleted the runtime
comparison of §1 for that model at those contexts. See §2.2.

What replaces it: refuse what the model itself declares unsupported, let the load
refuse what it cannot hold, and **measure** the rest. `peak_memory_bytes` and
`swap_flag` (§5.2) record what a run actually cost, so a lazily-allocated run
that did not fit is detected rather than predicted.
"""

from __future__ import annotations

ADMISSIBLE = "admissible"
UNSUPPORTED = "unsupported"
OVERSIZED = "oversized"


def classify_declared(advertised_max_context: int, context_len: int) -> str:
    """The free half of the check, from recorded metadata alone (§2.1).

    A context above what the model advertises is `unsupported` — skipped and
    recorded as such, never clamped or estimated. Everything else is provisional
    until the load is attempted; this function never returns `oversized`.
    """
    if context_len > advertised_max_context:
        return UNSUPPORTED
    return ADMISSIBLE


def classify_load_failure(message: str) -> str:
    """Whether a failed load means `oversized` or something else.

    Only a memory refusal is `oversized`. A missing artefact or a crashed
    backend is a different problem and must not be recorded as one the machine
    was too small for.
    """
    lowered = message.lower()
    memory_signals = (
        "out of memory",
        "insufficient memory",
        "not enough memory",
        "memory required",
        "failed to allocate",
        "guardrail",
        "too large",
    )
    return OVERSIZED if any(s in lowered for s in memory_signals) else ""
