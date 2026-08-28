"""Backend runtime identity for environment.json (§3).

The inference runtime is a property of the resident process, whose name and
command line differ between MLX and llama.cpp builds. It is discovered from
the resident process via the §5.2 discovery contract in `harness/metrics` —
never hardcoded, and never a second discovery implementation. A version is
recorded when the command line exposes one, and `None` otherwise: a missing
version is recorded, not estimated (§5.3).
"""

from __future__ import annotations

import re

from harness.metrics import find_inference_pid

_VERSION_IN_COMMAND = re.compile(r"(\d+\.\d+(?:\.\d+)?)")


def discover_runtime() -> tuple[str | None, str | None]:
    """Name and version of the resident inference runtime.

    The name is inferred from the command line: MLX backends run under an
    `mlx`-named engine, llama.cpp under a `llama`-named one. The version is
    taken from the first dotted number in the command line when one is present;
    most LM Studio backend launches do not embed one, so it is usually ``None``.
    """
    found = find_inference_pid()
    if found is None:
        return None, None
    _, command = found
    lowered = command.lower()
    if "mlx" in lowered:
        name = "mlx"
    elif "llama" in lowered:
        name = "llama.cpp"
    else:
        name = None

    match = _VERSION_IN_COMMAND.search(command)
    return name, match.group(1) if match else None