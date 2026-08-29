"""Backend runtime identity (§3, `backend_runtime`).

Verified against a live session that the resident process's command line
gives no hint of MLX vs llama.cpp for the current LM Studio build (the
worker is a generic `llmworker.js`). These tests pin the replacement: the
resident model's format from `lms ps`, matched against the engine `lms
runtime ls` reports selected for that format.
"""

from __future__ import annotations

from harness import lmstudio
from setup import probe_process


def _loaded(model_format: str) -> lmstudio.LoadedModel:
    return lmstudio.LoadedModel(
        identifier="bench",
        model_key="k",
        path="p",
        context_length=8192,
        format=model_format,
    )


def _engine(name: str, version: str, model_format: str, selected: bool):
    return lmstudio.RuntimeEngine(name, version, model_format, selected)


def test_mlx_resident_model_reports_mlx(monkeypatch):
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [_loaded("safetensors")])
    monkeypatch.setattr(
        lmstudio,
        "runtime_engines",
        lambda: [
            _engine("llama.cpp-mac-arm64-apple-metal-advsimd", "2.29.1", "GGUF", True),
            _engine("mlx-llm-mac-arm64-apple-metal-advsimd", "1.11.0", "MLX", True),
        ],
    )
    assert probe_process.discover_runtime() == ("mlx", "1.11.0")


def test_gguf_resident_model_reports_llama_cpp(monkeypatch):
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [_loaded("gguf")])
    monkeypatch.setattr(
        lmstudio,
        "runtime_engines",
        lambda: [
            _engine("llama.cpp-mac-arm64-apple-metal-advsimd", "2.29.1", "GGUF", True),
            _engine("mlx-llm-mac-arm64-apple-metal-advsimd", "1.11.0", "MLX", False),
        ],
    )
    assert probe_process.discover_runtime() == ("llama.cpp", "2.29.1")


def test_no_resident_model_is_none_not_a_guess(monkeypatch):
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [])

    def explode():
        raise AssertionError("must not shell out when nothing is resident")

    monkeypatch.setattr(lmstudio, "runtime_engines", explode)
    assert probe_process.discover_runtime() == (None, None)


def test_unrecognised_format_is_none_not_a_guess(monkeypatch):
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [_loaded("onnx")])

    def explode():
        raise AssertionError("must not shell out for an unrecognised format")

    monkeypatch.setattr(lmstudio, "runtime_engines", explode)
    assert probe_process.discover_runtime() == (None, None)


def test_no_engine_selected_for_the_format_is_none(monkeypatch):
    """Defensive: a format LM Studio reports but has no selected engine for."""
    monkeypatch.setattr(lmstudio, "list_loaded", lambda: [_loaded("safetensors")])
    monkeypatch.setattr(
        lmstudio,
        "runtime_engines",
        lambda: [
            _engine("llama.cpp-mac-arm64-apple-metal-advsimd", "2.29.1", "GGUF", True),
            _engine("mlx-llm-mac-arm64-apple-metal-advsimd", "1.11.0", "MLX", False),
        ],
    )
    assert probe_process.discover_runtime() == (None, None)
