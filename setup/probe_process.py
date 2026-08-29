"""Backend runtime identity for environment.json (§3).

The resident inference process's command line does not reliably name its
engine or embed a version — verified against a live session, where LM
Studio's `llmworker.js` worker gives no hint at all of MLX vs llama.cpp.
Runtime identity instead comes from two calls that are both already exposed
by `lms`: which model *format* is resident (`lms ps`), and which engine is
*selected* for that format (`lms runtime ls`). Process discovery
(`harness.metrics.find_inference_pid`) remains the only source for *memory*,
which is a property of the process, not of engine selection.
"""

from __future__ import annotations

from harness import lmstudio

_FORMAT_LABELS = {"safetensors": "MLX", "gguf": "GGUF"}


def _normalized_name(engine_name: str) -> str | None:
    if engine_name.startswith("mlx"):
        return "mlx"
    if engine_name.startswith("llama.cpp"):
        return "llama.cpp"
    return None


def discover_runtime() -> tuple[str | None, str | None]:
    """Name and version of the engine backing the resident model.

    `None, None` when no model is resident, its format is unrecognised, or no
    engine is selected for that format — a missing value is recorded, not
    estimated (§5.3).
    """
    resident = lmstudio.list_loaded()
    if not resident:
        return None, None

    model_format = _FORMAT_LABELS.get(resident[0].format or "")
    if model_format is None:
        return None, None

    for engine in lmstudio.runtime_engines():
        if engine.selected and engine.model_format == model_format:
            return _normalized_name(engine.name), engine.version
    return None, None
